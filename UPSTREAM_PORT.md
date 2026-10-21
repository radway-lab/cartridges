# Upstream port

Base: HazyResearch/cartridges ef34ba97a06049c34820506e2c283746284ae5f0.
The original Git history and LICENSE.md are retained.

Unchanged algorithms: SelfStudySynthesizer's seed prompts/role flipping and
conversation loop; TrainDataset packing; TrainableCache parameterization;
TrainConfig's Adam, gradient accumulation, and top-k teacher cross-entropy;
FlexAttention mask semantics.

Changes: native Hugging Face architecture adapters for Qwen2, Qwen3, and Granite;
new model/tokenizer registrations; explicit tokenizer return types; a local
teacher inference client using the upstream response/log-probability schema;
small-model entry points and PARCC launcher. Qwen3 configurations use non-thinking
formatting consistently for generation and distillation. Granite Mamba variants
and sliding-window attention are rejected.

Required bug fixes: retain all top-k entries when their probability mass does not
reach the sparse cutoff; restore frozen cache length from the token axis and
support zero frozen tokens; guard W&B logs for local runs; infer missing head_dim
for generation. The local client is serial and prioritizes compatibility over
serving throughput. Original remote clients remain available.

Previous standalone implementation remains in Git history (through ade5d73).
Its four passed tests and smoke job 9259579 are not validation of this port.

Preflight: package sources parse, pip check passes, and upstream model/trainer/
synthesizer imports succeed. GPU and token-alignment tests are pending.
