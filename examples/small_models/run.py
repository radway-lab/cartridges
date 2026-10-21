"""Configure upstream Cartridges; no replacement synthesis or training algorithm."""
import argparse
import asyncio
import gc
import os
from pathlib import Path

# Upstream initializes its default text resource at import time.
os.environ.setdefault("CARTRIDGES_DIR", str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", str(Path.cwd() / "outputs"))
import torch
from cartridges.models import HFModelConfig
from cartridges.models.small_models import SMALL_MODELS
from cartridges.clients.local_hf import LocalHFClient
from cartridges.data.resources import TextFileResource
from cartridges.data.chunkers import TokenChunker
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.structs import write_conversations
from cartridges.initialization.text import KVFromText
from cartridges.train import TrainConfig
from cartridges.datasets import TrainDataset, DataSource

async def synthesize(args, model_id):
    config = SelfStudySynthesizer.Config(
        client=LocalHFClient.Config(model_name=model_id, local_files_only=args.offline),
        resources=[TextFileResource.Config(path=args.document,
            seed_prompts=["structuring", "summarization", "question"],
            chunker=TokenChunker.Config(tokenizer=model_id, min_tokens_per_chunk=256,
                                       max_tokens_per_chunk=1024))],
        tools=[], max_rounds=args.rounds, num_top_logprobs=20,
        max_completion_tokens_a=64, max_completion_tokens_b=128, prob_thinking=0.0)
    synthesizer = config.instantiate()
    rows = []
    async with synthesizer:
        for i in range(args.samples):
            rows.extend(await synthesizer.sample_convos(i, 1, args.samples))
    if len(rows) != args.samples:
        raise RuntimeError("Incomplete synthesis")
    write_conversations(rows, str(Path(args.output) / "dataset.parquet"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["synthesize", "train"])
    p.add_argument("--model", choices=SMALL_MODELS, required=True)
    p.add_argument("--document", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--cartridge-tokens", type=int, default=128)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if min(args.samples, args.rounds, args.cartridge_tokens, args.steps, args.batch_size) < 1:
        p.error("All sizes must be positive")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model_id, cls = SMALL_MODELS[args.model]
    if args.stage == "synthesize":
        if (out / "dataset.parquet").exists():
            raise FileExistsError(out / "dataset.parquet")
        asyncio.run(synthesize(args, model_id))
    else:
        config = TrainConfig(
            name=args.model, output_dir=str(out), run_dir=str(out / "train"), wandb=None,
            model=HFModelConfig(pretrained_model_name_or_path=model_id, model_cls=cls,
                load_kwargs={"local_files_only": args.offline}),
            kv_cache_initializer=KVFromText.Config(text_source=args.document,
                max_tokens=args.cartridge_tokens, num_frozen_tokens=1),
            dataset=TrainDataset.Config(data_sources=[DataSource(path=str(out / "dataset.parquet"), type="local")],
                top_k_logits=20, packed_seq_length=2048, packing_mode="truncate"),
            lr=0.02, epochs=100, global_batch_size=args.batch_size,
            max_optimizer_steps=args.steps, save_to_wandb=False,
            save_every_n_steps=64, save_after_training=True, keep_last_n_saved=2)
        Path(config.run_dir).mkdir(parents=True, exist_ok=False)
        (Path(config.run_dir) / "config.json").write_text(config.model_dump_json(indent=2))
        config.run()

if __name__ == "__main__":
    main()
