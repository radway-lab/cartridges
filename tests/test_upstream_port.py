import os
import copy
import tempfile
import unittest
import numpy as np
import torch
from transformers import Qwen2Config, Qwen3Config, GraniteMoeHybridConfig, AutoModelForCausalLM
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.small_models import FlexQwen2ForCausalLM, FlexQwen3ForCausalLM, FlexGraniteForCausalLM
from cartridges.clients.base import TopLogprobs

class PortTests(unittest.TestCase):
    def test_probability_truncation_retains_top_k_if_mass_is_low(self):
        probs = TopLogprobs(np.log(np.array([[0.2, 0.1, 0.05]])), np.array([[1, 2, 3]]))
        self.assertEqual(len(probs.flatten(0.99).token_id), 3)

    @unittest.skipUnless(os.environ.get("PORT_TEST_TOKENIZERS"), "requires cached tokenizers")
    def test_teacher_student_token_alignment(self):
        from transformers import AutoTokenizer
        from cartridges.models.small_models import SMALL_MODELS
        from cartridges.datasets import MODEL_TO_MESSAGE_CONVERTER
        from cartridges.structs import Conversation
        for model_id, _ in SMALL_MODELS.values():
            with self.subTest(model=model_id):
                tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                messages = [{"role":"system","content":"A reference document."},
                            {"role":"user","content":"What is described?"}]
                prefix = tok.apply_chat_template(messages[:1], return_dict=False)
                prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                                 enable_thinking=False, return_dict=False)
                answer = tok.encode("A reference.", add_special_tokens=False)
                logp = TopLogprobs(np.zeros((len(answer),1)),np.array(answer)[:,None]).flatten()
                rows = [Conversation.Message(content=messages[1]["content"], role="user",token_ids=None),
                        Conversation.Message(content="A reference.",role="assistant",token_ids=answer,top_logprobs=logp)]
                element = MODEL_TO_MESSAGE_CONVERTER[model_id.lower()](rows,tokenizer=tok)
                self.assertEqual(element.input_ids.tolist(), prompt[len(prefix):] + answer)
                self.assertEqual(element.topk_token_idxs.tolist(),list(range(len(prompt)-len(prefix),len(prompt)-len(prefix)+len(answer))))

    def test_packed_attention_gradients_and_reload(self):
        device = os.environ.get("PORT_TEST_DEVICE", "cpu")
        torch.set_num_threads(1)
        common = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128)
        configs = [(Qwen2Config(**common), FlexQwen2ForCausalLM),
                   (Qwen3Config(**common, head_dim=8), FlexQwen3ForCausalLM),
                   (GraniteMoeHybridConfig(**common, layer_types=["full_attention"]*2,
                    num_local_experts=0, num_experts_per_tok=0, shared_intermediate_size=64,
                    mamba_n_heads=4, mamba_d_head=16, position_embedding_type="rope",
                    embedding_multiplier=2, attention_multiplier=0.125,
                    residual_multiplier=0.5, logits_scaling=4), FlexGraniteForCausalLM)]
        for cfg, cls in configs:
            with self.subTest(model=cfg.model_type):
                native = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").to(device).eval()
                port = cls(copy.deepcopy(cfg)).to(device).eval().requires_grad_(False)
                port.load_state_dict(native.state_dict())
                attn = AttnConfig(2, 2, 8)
                prefix = torch.tensor([3,4,5], device=device)
                empty = TrainableCache(attn)
                port(input_ids=prefix, seq_ids=torch.zeros_like(prefix),
                     position_ids=torch.arange(3,device=device), past_key_values=empty, mode="generate")
                for frozen in [0,1]:
                    cache = TrainableCache(attn, empty._keys, empty._values, frozen).to(device)
                    ids=torch.tensor([6,7,8,9],device=device)
                    seq=torch.tensor([0,0,1,1],device=device)
                    pos=torch.tensor([0,1,0,1],device=device)
                    packed=port(input_ids=ids,seq_ids=seq,position_ids=pos,past_key_values=cache).logits
                    expected=[]
                    for suffix in [ids[:2],ids[2:]]:
                        with torch.no_grad():
                            expected.append(native(input_ids=torch.cat([prefix,suffix])[None,:]).logits[:,3:])
                    torch.testing.assert_close(packed,torch.cat(expected,dim=1),atol=3e-5,rtol=3e-4)
                    self.assertEqual(cache.num_tokens(),3)  # training must not append
                    packed.square().mean().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in cache.trainable_keys))
                    self.assertTrue(all(p.grad is None for p in port.parameters()))
                    with tempfile.TemporaryDirectory() as tmp:
                        path=tmp+"/cache.pt";cache.save(path)
                        loaded=TrainableCache.from_pretrained(path,device=device)
                        self.assertEqual(loaded.num_tokens(),3)
                        other=port(input_ids=ids,seq_ids=seq,position_ids=pos,past_key_values=loaded).logits
                        torch.testing.assert_close(other,packed)

if __name__ == "__main__": unittest.main()
