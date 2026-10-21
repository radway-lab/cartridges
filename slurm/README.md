# Validation job

Submit `sbatch slurm/smoke.sbatch` from the repository root after resource approval.
The script does not submit any further jobs. Outputs: `outputs/smoke-JOBID/`.

Allocation: 1 B200 45GB MIG slice, 6 CPUs, 48GB host RAM, 60-minute wall limit.
Estimate: 5-15 minutes, <12GB GPU memory, <100MB new outputs; no weight downloads.
Upper bound: 1 MIG-slice hour, 6 CPU-hours. Estimates are unmeasured.

Runs tiny CPU model tests, then two one-turn self-study conversations, three
optimizer steps with a 32-token cartridge, and three evaluation questions per model.
Models run sequentially. This checks functionality, not research quality.

The existing PARCC environment has torch 2.11.0+cu128 and transformers 5.19.0.
For a separate environment, create a venv and install `requirements-parcc.txt`,
then set `CARTRIDGES_PYTHON` to its Python binary. No existing environment is modified.
