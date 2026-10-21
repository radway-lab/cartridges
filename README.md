# Cartridges for small Qwen and Granite models

Port of [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges),
retaining its self-study synthesizer, packed datasets, trainable KV cache,
FlexAttention, Adam optimizer, and top-k context-distillation loss.
Upstream revision: `ef34ba97a06049c34820506e2c283746284ae5f0`.

Models: `qwen2` = Qwen2.5-0.5B-Instruct, `qwen3` = Qwen3-0.6B,
`granite` = Granite-4.0-350m (attention-only configuration).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export CARTRIDGES_DIR="$PWD"
export CARTRIDGES_OUTPUT_DIR="$PWD/outputs"
```

Use Python 3.11+ and CUDA PyTorch. Transformers is pinned to 5.19.0 for the
native model adapters. Original Llama examples have not been validated with it.

## Run

Use a UTF-8 text document or start with the authors' paper example:

```bash
python -m examples.small_models.run synthesize --model qwen3 --document examples/arxiv/cartridges.tex --output outputs/qwen3
python -m examples.small_models.run train --model qwen3 --document examples/arxiv/cartridges.tex --output outputs/qwen3
```

Change `--model` and use a separate output directory for each model. Synthesis
uses the upstream two-agent recipe and writes upstream `dataset.parquet` files;
training invokes upstream `TrainConfig`. No external inference service or W&B
account is required. Add `--offline` to use cached models only.

Defaults: 512 conversations, two rounds, 128-token cartridge, up to 256 optimizer
steps, global batch size 4, learning rate 0.02. See `--help` to change them.
These are starting settings, not validated quality claims or paper-scale results.

## PARCC

After resource approval, submit `sbatch slurm/upstream-validation.sbatch` from
this directory. It requests one 45GB MIG GPU, 6 CPUs, 48GB RAM, and one hour.
It checks native-model equivalence and packed-sequence isolation, then exercises
the original synthesis/trainer on all three models. Cluster paths are in `slurm/`.

Validation status: port not yet GPU-tested. The earlier successful standalone
smoke run did not use the upstream trainer and does not validate this port.
Upstream source and its Apache-2.0 license are preserved. See `UPSTREAM_PORT.md`
for the compatibility changes.
