Run upstream-validation.sbatch only after allocation approval. It requests
one B200 45GB MIG slice, 6 CPUs, 48GB host memory and one hour (upper bound:
1 MIG-slice hour and 6 CPU-hours). Estimate: 15-45 minutes including first-run
FlexAttention compilation; less than 20GB GPU memory and 5GB new caches/outputs.
No weights are downloaded. Uses envs/cartridges-upstream; original env unchanged.
This validates the upstream port, not compression quality at research scale.
