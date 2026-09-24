"""GLM-5.3-Flash (``glm5_next``, ``Glm5NextForConditionalGeneration``), text-only.

Correctness-first port of ``transformers/models/glm5_next``: hybrid KDA linear attention
(``kda.py``) + NoPE MLA with the k-pool DSA indexer (``attention.py``), manifold-
constrained hyper-connections and clamped-SwiGLU MLP/MoE (``ops.py``/``moe.py``).

Known gaps (see the module docstrings): plain-PyTorch KDA prefill and DSA attention (no
fused kernels yet), no hybrid-radix prefix caching (engine forces ``--cache-type
naive``), no MTP (the ``layers.<N>`` MTP weights are skipped), TP=1 only, bf16 KV (the
checkpoint's fp8 KV hint is ignored).
"""

from .config import parse_config
from .model import Glm5NextForCausalLM
from .weight import iter_weights, load_nvfp4_expert_sources, load_nvfp4_expert_sources_parallel

__all__ = [
    "Glm5NextForCausalLM",
    "parse_config",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
