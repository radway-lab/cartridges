import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen3Config, GraniteMoeHybridConfig
from radway_cartridges.core import Cartridge, distillation_loss, fingerprint, validate_config, prompt_ids

class CartridgeTests(unittest.TestCase):
    def configs(self):
        common = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, bos_token_id=1, eos_token_id=2)
        yield Qwen2Config(**common)
        yield Qwen3Config(**common, head_dim=8)
        yield GraniteMoeHybridConfig(**common, layer_types=["full_attention"] * 2,
            num_local_experts=0, num_experts_per_tok=0, shared_intermediate_size=64,
            mamba_n_heads=4, mamba_d_head=16, position_embedding_type="rope",
            attention_multiplier=0.125, embedding_multiplier=2.0,
            residual_multiplier=0.5, logits_scaling=4.0)

    def test_native_prefix_equivalence_gradients_and_roundtrip(self):
        torch.set_num_threads(1)
        for config in self.configs():
            with self.subTest(architecture=config.model_type):
                torch.manual_seed(7)
                validate_config(config)
                model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
                model.eval().requires_grad_(False)
                prefix = torch.tensor([[3, 4, 5, 6]])
                suffix = torch.tensor([[7, 8, 9]])
                cartridge = Cartridge.initialize(model, prefix)
                with torch.no_grad():
                    expected = model(input_ids=torch.cat([prefix, suffix], dim=1)).logits[:, 4:]
                actual = model(input_ids=suffix, past_key_values=cartridge.cache(model.dtype), use_cache=True).logits
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                # A fresh cache must reproduce the same result after a prior call.
                fresh = model(input_ids=suffix, past_key_values=cartridge.cache(model.dtype), use_cache=True).logits
                torch.testing.assert_close(fresh, actual)
                # Changing a future token must not change preceding logits.
                other = model(input_ids=torch.tensor([[7, 8, 20]]),
                              past_key_values=cartridge.cache(model.dtype), use_cache=True).logits
                torch.testing.assert_close(other[:, :2], actual[:, :2])
                before = [p.detach().clone() for p in cartridge.parameters()]
                optimizer = torch.optim.AdamW(cartridge.parameters(), lr=0.01)
                F.cross_entropy(actual.reshape(-1, 64), torch.tensor([12, 13, 14])).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in cartridge.parameters()))
                self.assertGreater(sum(p.grad.abs().sum().item() for p in cartridge.parameters()), 0)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                optimizer.step()
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, cartridge.parameters())))
                meta = {"model_id": "tiny", "config_hash": fingerprint(model.config),
                        "model_revision": getattr(model.config, "_commit_hash", None)}
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "cartridge.pt"
                    cartridge.save(path, meta)
                    loaded, _ = Cartridge.load(path, model, "tiny")
                    a = model(input_ids=suffix, past_key_values=cartridge.cache(model.dtype)).logits
                    b = model(input_ids=suffix, past_key_values=loaded.cache(model.dtype)).logits
                    torch.testing.assert_close(a, b)
                    with self.assertRaises(ValueError):
                        Cartridge.load(path, model, "wrong-model")
                # Incremental decoding must agree with a single causal forward.
                cache = cartridge.cache(model.dtype)
                pieces = []
                for i in range(suffix.shape[1]):
                    out = model(input_ids=suffix[:, i:i+1], past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    pieces.append(out.logits)
                torch.testing.assert_close(torch.cat(pieces, dim=1), a, atol=2e-5, rtol=2e-4)

    def test_kl_matches_full_distribution_and_identity(self):
        torch.manual_seed(3)
        teacher = torch.randn(3, 16)
        student = torch.randn(3, 16, requires_grad=True)
        expected = F.kl_div(student.log_softmax(-1), teacher.softmax(-1), reduction="batchmean")
        torch.testing.assert_close(distillation_loss(student, teacher, 16), expected)
        for k in [1, 5, 16]:
            self.assertAlmostEqual(distillation_loss(teacher, teacher, k).item(), 0, places=6)
            loss = distillation_loss(student, teacher, k)
            self.assertGreaterEqual(loss.item(), -1e-6)
            loss.backward()
            self.assertTrue(torch.isfinite(student.grad).all())
            student.grad = None

    def test_chat_template_returns_token_ids(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
        ids = prompt_ids(tokenizer, [{"role": "user", "content": "hello"}])
        self.assertIsInstance(ids, list)
        self.assertTrue(ids)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_reject_hybrid(self):
        config = list(self.configs())[-1]
        config.layer_types = ["linear_attention", "full_attention"]
        with self.assertRaises(ValueError):
            validate_config(config)

if __name__ == "__main__":
    unittest.main()
