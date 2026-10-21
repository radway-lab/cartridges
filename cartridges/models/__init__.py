from .config import HFModelConfig, PeftConfig, ModelConfig
from .small_models import FlexQwen2ForCausalLM, FlexQwen3ForCausalLM, FlexGraniteForCausalLM

# Keep legacy Llama available without eagerly importing its older HF internals.
def __getattr__(name):
    if name == "FlexLlamaForCausalLM":
        from .llama.modeling_llama import FlexLlamaForCausalLM
        return FlexLlamaForCausalLM
    raise AttributeError(name)
