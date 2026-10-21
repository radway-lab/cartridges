"""Single-GPU inference client for the unchanged upstream SelfStudySynthesizer."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cartridges.clients.base import Client, ClientConfig, ClientResponse, ClientSample, TopLogprobs
from cartridges.clients.usage import Usage

class LocalHFClient(Client):
    class Config(ClientConfig):
        device: str = "cuda"
        local_files_only: bool = False
        revision: str | None = None

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name,
            local_files_only=config.local_files_only, revision=config.revision)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name,
            local_files_only=config.local_files_only, revision=config.revision,
            dtype=torch.bfloat16, attn_implementation="sdpa").to(config.device).eval()

    async def chat(self, chats, temperature=0.6, stop=None, max_completion_tokens=128,
                   frequency_penalty=0.0, top_logprobs=None, logprobs_start_message=None,
                   modal_upstream_id=None, enable_thinking=False):
        if stop or frequency_penalty or logprobs_start_message is not None or enable_thinking:
            raise ValueError("Local client supports non-thinking generation without custom stops/penalties")
        samples, usage = [], Usage()
        with torch.inference_mode():
            for messages in chats:
                ids = self.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
                limit = max_completion_tokens or 128
                if len(ids) + limit > self.model.config.max_position_embeddings:
                    raise ValueError("Synthesis exceeds model context limit; reduce chunk/generation sizes")
                current = torch.tensor([ids], device=self.model.device)
                cache, generated, probabilities, indices = None, [], [], []
                eos = self.model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, list) else [eos])
                for _ in range(limit):
                    out = self.model(input_ids=current, past_key_values=cache,
                                     use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                    logits = out.logits[0, -1].float()
                    if top_logprobs:
                        lp, ix = logits.log_softmax(-1).topk(top_logprobs)
                        probabilities.append(lp.cpu().numpy())
                        indices.append(ix.cpu().numpy())
                    token = (torch.multinomial((logits / temperature).softmax(-1), 1).item()
                             if temperature > 0 else logits.argmax().item())
                    generated.append(token)
                    if token in eos:
                        break
                    current = torch.tensor([[token]], device=self.model.device)
                samples.append(ClientSample(
                    text=self.tokenizer.decode(generated, skip_special_tokens=True),
                    token_ids=generated,
                    top_logprobs=TopLogprobs(np.asarray(probabilities, dtype=np.float32),
                                            np.asarray(indices, dtype=np.int32)) if top_logprobs else None))
                usage += Usage(prompt_tokens=len(ids), completion_tokens=len(generated))
        return ClientResponse(samples=samples, usage=usage)
