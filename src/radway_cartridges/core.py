from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODELS = {
    "granite": "ibm-granite/granite-4.0-350m",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3": "Qwen/Qwen3-0.6B",
}

def validate_config(config):
    if config.model_type not in {"qwen2", "qwen3", "granitemoehybrid"}:
        raise ValueError(f"Unsupported architecture: {config.model_type}")
    kinds = getattr(config, "layer_types", None) or getattr(config, "layers_block_type", [])
    if any(k not in {"attention", "full_attention"} for k in kinds):
        raise ValueError("Only full-attention models are supported; no Mamba/sliding-window layers")
    if getattr(config, "use_sliding_window", False):
        raise ValueError("Sliding-window attention is unsupported")

def fingerprint(config):
    data = config.to_dict()
    for key in ["_name_or_path", "transformers_version", "_commit_hash"]:
        data.pop(key, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

def load_model(name, device="cuda", dtype="bfloat16", offline=False, revision=None):
    model_id = MODELS.get(name, name)
    config = AutoConfig.from_pretrained(model_id, local_files_only=offline, revision=revision)
    validate_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=offline, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, local_files_only=offline, revision=revision,
        dtype=getattr(torch, dtype), attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, model_id

class Cartridge(nn.Module):
    """FP32 optimizer parameters; each forward gets a fresh differentiable HF cache.

    Keys are stored after RoPE, exactly as produced by prefill. Query positions
    start at the cartridge length. Cache concatenation preserves gradients.
    """
    def __init__(self, keys, values):
        super().__init__()
        if not keys or len(keys) != len(values):
            raise ValueError("Expected nonempty per-layer K/V tensors")
        shape = keys[0].shape
        if len(shape) != 4 or shape[0] != 1 or shape[2] < 1:
            raise ValueError("Expected shape [1, kv_heads, tokens, head_dim]")
        if any(k.shape != shape or v.shape != shape for k, v in zip(keys, values)):
            raise ValueError("Inconsistent cache dimensions")
        self.keys = nn.ParameterList([nn.Parameter(k.detach().float().clone()) for k in keys])
        self.values = nn.ParameterList([nn.Parameter(v.detach().float().clone()) for v in values])

    @property
    def length(self):
        return self.keys[0].shape[2]

    def cache(self, dtype):
        # Never detach here: gradients must reach the cartridge parameters.
        return DynamicCache(ddp_cache_data=[
            (k.to(dtype=dtype), v.to(dtype=dtype)) for k, v in zip(self.keys, self.values)
        ])

    @classmethod
    @torch.no_grad()
    def initialize(cls, model, tokens):
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
            raise ValueError("Initialization requires a nonempty single token sequence")
        cache = model(input_ids=tokens, use_cache=True, logits_to_keep=1).past_key_values
        return cls([layer.keys for layer in cache.layers], [layer.values for layer in cache.layers])

    def save(self, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "metadata": metadata,
                    "keys": [k.detach().cpu() for k in self.keys],
                    "values": [v.detach().cpu() for v in self.values]}, path)

    @classmethod
    def load(cls, path, model, model_id):
        state = torch.load(path, map_location=model.device, weights_only=True)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint version")
        meta = state["metadata"]
        if meta["model_id"] != model_id or meta["config_hash"] != fingerprint(model.config):
            raise ValueError("Cartridge model/configuration mismatch")
        if meta.get("model_revision") != getattr(model.config, "_commit_hash", None):
            raise ValueError("Cartridge model revision mismatch")
        cartridge = cls(state["keys"], state["values"])
        cfg = model.config
        expected = (1, cfg.num_key_value_heads, cartridge.length,
                    getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
        if len(cartridge.keys) != cfg.num_hidden_layers or cartridge.keys[0].shape != expected:
            raise ValueError("Checkpoint cache geometry does not match model")
        return cartridge, meta

def prompt_ids(tokenizer, history, context=None):
    system = "Answer using the reference material. If the answer is not available, say so."
    if context is not None:
        system += "\n\nReference material:\n" + context
    messages = [{"role": "system", "content": system}] + history
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             enable_thinking=False, return_dict=False)
    # Plain-text fallback for tokenizers without a supplied chat template.
    text = system + "\n\n" + "\n".join(
        ("Question: " if m["role"] == "user" else "Answer: ") + m["content"] for m in history
    ) + "\nAnswer:"
    return tokenizer.encode(text, add_special_tokens=True)

def tensor(ids, model):
    return torch.tensor([ids], dtype=torch.long, device=model.device)

def check_length(model, count):
    if count > model.config.max_position_embeddings:
        raise ValueError(f"Sequence length {count} exceeds model context limit")

@torch.no_grad()
def decode(model, tokenizer, ids, max_new_tokens=128, cartridge=None, sample=False):
    if not ids or max_new_tokens < 1:
        raise ValueError("Nonempty prompt and positive generation length required")
    prefix = cartridge.length if cartridge is not None else 0
    check_length(model, prefix + len(ids) + max_new_tokens)
    cache = cartridge.cache(model.dtype) if cartridge is not None else None
    current = tensor(ids, model)
    generated = []
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for _ in range(max_new_tokens):
        out = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        logits = out.logits[:, -1].float()
        token = (torch.multinomial((logits / 0.8).softmax(-1), 1) if sample
                 else logits.argmax(-1, keepdim=True))
        generated.append(token.item())
        if token.item() in eos:
            break
        current = token
    return generated

def response_logits(model, prompt, response, cartridge=None):
    if not prompt or not response:
        raise ValueError("Nonempty prompt and response required")
    prefix = cartridge.length if cartridge is not None else 0
    check_length(model, prefix + len(prompt) + len(response))
    # Input ending one token before the final target aligns every response label.
    inputs = tensor(prompt + response[:-1], model)
    cache = cartridge.cache(model.dtype) if cartridge is not None else None
    return model(input_ids=inputs, past_key_values=cache, use_cache=True,
                 logits_to_keep=len(response)).logits[0].float()

def distillation_loss(student_logits, teacher_logits, top_k=20):
    """KL over teacher top-k tokens plus one bucket for the remaining mass.

    The tail bucket preserves normalization without pretending the top-k contains
    the whole distribution. top_k >= vocabulary size gives exact full-vocab KL.
    """
    teacher_logp = teacher_logits.detach().float().log_softmax(-1)
    student_logp = student_logits.float().log_softmax(-1)
    k = min(top_k, teacher_logp.shape[-1])
    if k < 1:
        raise ValueError("top_k must be positive")
    logp, indices = teacher_logp.topk(k, dim=-1)
    logq = student_logp.gather(-1, indices)
    p = logp.exp()
    loss = (p * (logp - logq)).sum(-1)
    if k < teacher_logp.shape[-1]:
        # Compute the complement directly to avoid cancellation near mass=1.
        keep = torch.zeros_like(teacher_logp, dtype=torch.bool).scatter_(-1, indices, True)
        tail_p = teacher_logp.masked_fill(keep, -torch.inf).logsumexp(-1)
        tail_q = student_logp.masked_fill(keep, -torch.inf).logsumexp(-1)
        loss = loss + tail_p.exp() * (tail_p - tail_q)
    return loss.mean()
