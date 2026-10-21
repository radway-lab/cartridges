#!/bin/bash
#SBATCH --job-name=cartridges-upstream
#SBATCH --account=radway-radway-lab
#SBATCH --partition=b200-mig45
#SBATCH --qos=mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cartridges-upstream-%j.out
#SBATCH --error=cartridges-upstream-%j.err
set -euo pipefail
ROOT=${CARTRIDGES_ROOT:-/vast/projects/radway/radway-lab/dangol/cartridges}
PYTHON=${CARTRIDGES_PYTHON:-/vast/projects/radway/radway-lab/dangol/envs/cartridges-upstream/bin/python}
export CARTRIDGES_DIR="$ROOT"
export CARTRIDGES_OUTPUT_DIR="$ROOT/outputs/upstream-${SLURM_JOB_ID}"
export HF_HOME=${HF_HOME:-/vast/projects/radway/radway-lab/dangol/hf_cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PORT_TEST_TOKENIZERS=1
export OMP_NUM_THREADS=6
export XDG_CACHE_HOME="$ROOT/.cache"
export TMPDIR="$ROOT/.cache/tmp"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
mkdir -p "$TMPDIR" "$CARTRIDGES_OUTPUT_DIR"
cd "$ROOT"
"$PYTHON" -m unittest discover -s tests -v
PORT_TEST_DEVICE=cuda "$PYTHON" -m unittest discover -s tests -v
for MODEL in qwen2 qwen3 granite; do
    OUT="$CARTRIDGES_OUTPUT_DIR/$MODEL"
    "$PYTHON" -m examples.small_models.run synthesize --offline --model "$MODEL" --document examples/arxiv/cartridges.tex --output "$OUT" --samples 8 --rounds 2
    "$PYTHON" -m examples.small_models.run train --offline --model "$MODEL" --document examples/arxiv/cartridges.tex --output "$OUT" --cartridge-tokens 128 --steps 4 --batch-size 1
done
