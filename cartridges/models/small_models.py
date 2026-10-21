"""Native HF architectures adapted to upstream Cartridges' packed-cache contract."""
import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers import AttentionInterface, Qwen2ForCausalLM, Qwen3ForCausalLM, GraniteMoeHybridForCausalLM
from transformers.cache_utils import DynamicCache
from cartridges.models.attention import create_block_mask_w_cache, flex_attention_forward, repeat_kv


def cartridge_attention(module, query, key, value, attention_mask=None,
                        scaling=None, dropout=0.0, cartridge_mode="train", **kwargs):
    if isinstance(attention_mask, BlockMask):
        return flex_attention_forward(module, query, key, value, attention_mask,
                                      scaling=scaling, mode=cartridge_mode), None
    # Reference backend for CPU correctness checks and un-packed teacher inference.
    key = repeat_kv(key, query.shape[1] // key.shape[1])
    value = repeat_kv(value, query.shape[1] // value.shape[1])
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        is_causal=attention_mask is None and query.shape[-2] > 1, scale=scaling)
    return output.transpose(1, 2).contiguous(), None

AttentionInterface.register("cartridges_packed", cartridge_attention)

class CacheBridge(DynamicCache):
    def __init__(self, cache, seq_ids, mode):
        super().__init__()
        self.cartridge_cache, self.seq_ids, self.mode = cache, seq_ids, mode

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return self.cartridge_cache.update(key_states, value_states, self.seq_ids,
                                            layer_idx, skip_append=self.mode == "train")

    def get_seq_length(self, layer_idx=0):
        return self.cartridge_cache.num_tokens()

class PackedModelMixin:
    def __init__(self, config):
        kinds = getattr(config, "layer_types", None) or []
        if any(k not in {"attention", "full_attention"} for k in kinds):
            raise ValueError("Cartridges supports full attention only, not Mamba/sliding windows")
        if getattr(config, "use_sliding_window", False):
            raise ValueError("Sliding-window models are unsupported")
        config._attn_implementation = "cartridges_packed"
        super().__init__(config)

    def forward(self, input_ids=None, seq_ids=None, position_ids=None,
                past_key_values=None, use_cache=None, mode="train", **kwargs):
        if seq_ids is None:
            return super().forward(input_ids=input_ids, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache, **kwargs)
        if mode not in {"train", "generate"}:
            raise ValueError("mode must be train or generate")
        if input_ids.ndim != 1 or seq_ids.shape != input_ids.shape or position_ids.shape != input_ids.shape:
            raise ValueError("Packed inputs, sequence IDs, and positions must be matching 1D tensors")
        cache = past_key_values
        cache_len = cache.num_tokens() if cache is not None else 0
        prefix_len = cache.num_cartridge_tokens() if cache is not None else 0
        if input_ids.is_cuda and getattr(self.config, "cartridges_backend", "flex") == "flex":
            mask = create_block_mask_w_cache(cache, seq_ids, input_ids.device)
        else:
            kv_seq_ids = torch.cat([cache.seq_ids(), seq_ids]) if cache_len else seq_ids
            q_idx = torch.arange(len(seq_ids), device=input_ids.device)[:, None]
            k_idx = torch.arange(len(kv_seq_ids), device=input_ids.device)[None, :]
            allowed = (kv_seq_ids[None, :] == -1) | (
                (seq_ids[:, None] == kv_seq_ids[None, :]) & (q_idx + cache_len >= k_idx))
            mask = torch.zeros(allowed.shape, dtype=self.dtype, device=input_ids.device)
            mask.masked_fill_(~allowed, torch.finfo(self.dtype).min)
            mask = mask[None, None, :, :]
        out = super().forward(input_ids=input_ids[None, :],
            position_ids=(position_ids + prefix_len)[None, :],
            attention_mask={"full_attention": mask},
            past_key_values=CacheBridge(cache, seq_ids, mode) if cache is not None else None,
            use_cache=cache is not None, cartridge_mode=mode, **kwargs)
        out.past_key_values = cache if use_cache else None
        return out

class FlexQwen2ForCausalLM(PackedModelMixin, Qwen2ForCausalLM):
    pass

class FlexQwen3ForCausalLM(PackedModelMixin, Qwen3ForCausalLM):
    pass

class FlexGraniteForCausalLM(PackedModelMixin, GraniteMoeHybridForCausalLM):
    pass

SMALL_MODELS = {
    "qwen2": ("Qwen/Qwen2.5-0.5B-Instruct", FlexQwen2ForCausalLM),
    "qwen3": ("Qwen/Qwen3-0.6B", FlexQwen3ForCausalLM),
    "granite": ("ibm-granite/granite-4.0-350m", FlexGraniteForCausalLM),
}
