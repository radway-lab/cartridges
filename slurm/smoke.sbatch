#!/bin/bash
#SBATCH --job-name=cartridges-smoke
#SBATCH --account=radway-radway-lab
#SBATCH --partition=b200-mig45
#SBATCH --qos=mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cartridges-smoke-%j.out
#SBATCH --error=cartridges-smoke-%j.err
set -euo pipefail
ROOT=${CARTRIDGES_ROOT:-/vast/projects/radway/radway-lab/dangol/cartridges}
PYTHON=${CARTRIDGES_PYTHON:-/vast/projects/radway/radway-lab/dangol/envs/cartridges/bin/python}
export HF_HOME=${HF_HOME:-/vast/projects/radway/radway-lab/dangol/hf_cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$ROOT/.cache"
export TMPDIR="$ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
"$PYTHON" -m unittest discover -s tests -v
OUT="$ROOT/outputs/smoke-${SLURM_JOB_ID}"
mkdir -p "$OUT"
for MODEL in granite qwen2 qwen3; do
    "$PYTHON" -m radway_cartridges.cli synthesize --offline --model "$MODEL" --document examples/corpus.txt --samples 2 --turns 1 --question-tokens 24 --answer-tokens 32 --output "$OUT/$MODEL-study.jsonl"
    "$PYTHON" -m radway_cartridges.cli train --offline --model "$MODEL" --document examples/corpus.txt --data "$OUT/$MODEL-study.jsonl" --cartridge-tokens 32 --steps 3 --output "$OUT/$MODEL.pt"
    "$PYTHON" -m radway_cartridges.cli evaluate --offline --model "$MODEL" --document examples/corpus.txt --data examples/eval.jsonl --checkpoint "$OUT/$MODEL.pt" --answer-tokens 24 --output "$OUT/$MODEL-eval.jsonl"
done
