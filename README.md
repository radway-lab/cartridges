# Cartridges for Granite and Qwen

Train a small, reusable KV cache for a text corpus while keeping model weights frozen.
Implements self-study conversation synthesis and context distillation from
[Cartridges](https://arxiv.org/abs/2506.06266), following
[HazyResearch/cartridges](https://github.com/HazyResearch/cartridges).

Supported aliases: `granite` (Granite 4.0 350M), `qwen2` (Qwen2.5 0.5B Instruct),
`qwen3` (Qwen3 0.6B). HF IDs or local model paths also work for these architectures.
Granite uses plain-text prompts; Qwen uses its chat template with thinking disabled.

## Install

Python 3.11+ and a CUDA GPU are recommended. Install a PyTorch build suitable for your GPU.

```bash
git clone https://github.com/radway-lab/cartridges.git
cd cartridges
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Alternatively: `conda env create -f environment.yml`, then `conda activate radway-cartridges`.
Transformers is pinned because its cache API changes. PARCC versions are recorded in
`requirements-parcc.txt`. CPU: add `--device cpu --dtype float32` to commands.

## Use

Start with the included toy corpus, then replace it with your UTF-8 text file.
Use the same model and document for synthesis, training, and evaluation.

```bash
cartridges synthesize --model qwen3 --document examples/corpus.txt --output outputs/study.jsonl
cartridges train --model qwen3 --document examples/corpus.txt --data outputs/study.jsonl --output outputs/cartridge.pt
cartridges evaluate --model qwen3 --document examples/corpus.txt --data examples/eval.jsonl --checkpoint outputs/cartridge.pt --output outputs/eval.jsonl
cartridges chat --model qwen3 --checkpoint outputs/cartridge.pt --question "Who directs the station?"
```

Change `--model` for each model; keep its outputs separate. Outputs are not overwritten.
Evaluation JSONL contains `question` and `answer` fields; use questions excluded from
training. Results include reference-answer NLL, greedy responses, and strict exact match
for full context, no context, and cartridge. These are diagnostic metrics, not a claim
that a trained cartridge matches full-context quality.

Useful options: `--revision`, `--offline`, `--cartridge-tokens`, `--steps`, `--lr`,
`--samples`, `--turns`, `--chunk-tokens`. Run any subcommand with `--help`.
The whole document must fit the model context for full-context evaluation.

## PARCC and tests

`slurm/smoke.sbatch` runs unit tests and a tiny end-to-end pass on all three cached
models. Set `CARTRIDGES_ROOT`, `CARTRIDGES_PYTHON`, and `HF_HOME` for another installation.
It uses one B200 45GB MIG slice for at most 30 minutes; obtain allocation approval
before submitting. From the repository root:

```bash
sbatch slurm/smoke.sbatch
```

Outside PARCC: `python -m unittest discover -s tests -v` runs tiny random-model tests on CPU.
No downloaded weights are needed for these tests.

## Scope

This is a standalone implementation, not an upstream fork or checkpoint-compatible port.
It retains native model attention, trains an FP32 post-RoPE KV prefix with AdamW,
and distills teacher top-20 probabilities plus a residual-probability bucket.
Teacher distributions are recomputed during training. Batch size is one; no packed
training, distributed training, Mamba/sliding-window caches, or fast serving integration.
The included pilot checks functionality only. Validation status: pending SLURM approval.
