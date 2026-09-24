"""Engine-facing config for GLM-5.3-Flash (``glm5_next``).

The checkpoint is a multimodal wrapper (``Glm5NextForConditionalGeneration``): the text
tower lives in ``text_config`` and its weights under ``model.language_model.``; FreeToken
serves it text-only (the ``model.visual.`` tower is never built or loaded).

Attention groups (how the engine sizes caches and picks a backend):

* the MLA/DSA layers form ONE latent-KV group (``mla=True``, ``head_dim = kv_lora_rank``
  since the MLA is NoPE) carrying the index-slab dims, so the pool factory builds a
  ``DSAKVCache`` restricted to just those layers (``layer_ids``) and the engine picks the
  ``dsa`` backend (used here only for its per-batch metadata / decode row staging: the
  k-pool indexer and the sparse MLA are computed by the model module itself, see
  ``attention.py``). The index slab row is ``3 * index_head_dim`` wide
  (key | compress-gate | pooled key);
* the KDA layers form a ``LinearGatedDeltaGroupConfig``: KDA's state geometry (conv over
  q|k|v, a [K, V] recurrent state per head) is exactly GDN's with
  ``num_key_heads == num_value_heads``, so the stock ``LinearStatePool`` holds it.

Prefix caching: the hybrid-radix cache snapshots linear state at chunk boundaries through
the GDN op's track-snapshot hooks, which the KDA op does not implement yet, so this model
opts out (``linear_state_prefix_cache=False`` -> the engine forces ``--cache-type
naive``). Correct, just no cross-request prefix reuse.

Quantization (nvidia modelopt NVFP4): only the routed experts (-> offload cache, Triton
NVFP4 kernels: the clamped SwiGLU rules out marlin/b12x) and the dense MLP of the leading
dense layers (dequantized to bf16 at load) are FP4; attention, shared experts, router,
embeddings, lm_head and hyper-connection params are bf16. The fp8 KV-cache hint is
ignored (bf16 latent KV). The trailing MTP layer (``layers.<num_layers>``) is not served.
"""

from __future__ import annotations

from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_expert_quant,
)

from .args import load_args

# The routed/shared/dense SwiGLU of GLM-5.3 clamps gate (max) and up (+-limit) before
# silu(gate) * up. Exposed as the model's ``hidden_act`` so every generic consumer that
# keys off it (NVFP4 expert backend pick, CPU-MoE eligibility) sees a non-plain-silu
# activation instead of silently dropping the clamp.
MOE_ACTIVATION = "silu_clamp"


def _text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", None) or hf_config


def parse_config(hf_config: Any) -> ModelConfig:
    text = _text_config(hf_config)
    args = load_args(text)
    num_layers = args.num_layers

    # NoPE everywhere: rotary_dim 0 / base 0.0 is the repo's "no rope" marker.
    rotary_config = RotaryConfig(
        head_dim=args.latent_dim,
        rotary_dim=0,
        max_position=args.max_position,
        base=0.0,
        scaling=None,
    )
    dsa_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=args.dsa_layer_ids,
        num_kv_heads=1,
        head_dim=args.latent_dim,
        rotary_config=rotary_config,
        mla=True,
        index_head_dim=args.index_state_dim,
        num_index_layers=len(args.indexer_layer_ids),
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=args.linear_layer_ids,
        num_key_heads=args.linear_num_heads,
        num_value_heads=args.linear_num_heads,
        key_head_dim=args.linear_head_dim,
        value_head_dim=args.linear_head_dim,
        conv_kernel_dim=args.linear_conv_kernel_dim,
        output_gate=True,
    )
    groups = (dsa_group, linear_group) if args.linear_layer_ids else (dsa_group,)

    get = lambda k, d=None: (text.get(k, d) if isinstance(text, dict) else getattr(text, k, d))
    num_experts = int(get("n_routed_experts", None) or get("num_local_experts", 0) or 0)
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=1,  # one shared MLA latent
        head_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        vocab_size=int(get("vocab_size")),
        intermediate_size=int(get("intermediate_size")),
        hidden_act=MOE_ACTIVATION,
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        rotary_config=rotary_config,
        attention_groups=groups,
        num_experts=num_experts,
        num_experts_per_tok=int(get("num_experts_per_tok")),
        moe_intermediate_size=int(get("moe_intermediate_size")),
        norm_topk_prob=bool(get("norm_topk_prob", True)),
        model_type=str(getattr(hf_config, "model_type", "glm5_next")),
        architectures=list(getattr(hf_config, "architectures", None) or ["Glm5NextForConditionalGeneration"]),
        moe_enabled=True,
        expert_quant=detect_expert_quant(hf_config),
        first_k_dense_replace=args.first_k_dense_replace,
        n_shared_experts=int(get("n_shared_experts", 1)),
        routed_scaling_factor=float(get("routed_scaling_factor", 1.0)),
        n_group=int(get("n_group", 1) or 1),
        topk_group=int(get("topk_group", 1) or 1),
        attn_sm_scale=args.qk_head_dim**-0.5,
        swiglu_limit=args.swiglu_limit,
        glm_dsa_args=args,
        # KDA does not implement the hybrid-radix track snapshots yet -> naive cache.
        linear_state_prefix_cache=False,
    )


__all__ = ["parse_config", "MOE_ACTIVATION"]
