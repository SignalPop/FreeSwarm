"""Qwen4-Exp (``model_type: qwen4_exp``, Qwen3.8-Flash-Next) config parsing.

The checkpoint wraps a text tower (``text_config``, ``qwen4_exp_text``) and a vision tower;
FreeToken serves text-only. The text tower is Qwen3.5-style (GatedDeltaNet linear layers +
gated GQA + 512-expert MoE with a gated shared expert) plus three new pieces:

* **QSA** (Qwen Sparse Attention) on the attention layers (config ``full_attention`` /
  ``qwen_sparse_attention``): a 4-head / 1-key-head indexer scores mean-pooled 4-token key
  blocks and each query attends only the top ``indexer_budget / compress_ratio`` blocks plus
  its trailing partial block.
* **Gated residual hyper-connections**: the residual stream is ``hc_count`` (4) copies of
  ``hidden_size`` mixed by read/write gates around every sublayer.
* **PLE**: a hashed 2/3-gram embedding table (~320M rows) injected into the residual stream
  at ``ple_layer_ids`` (1-indexed); served from host memory.

The engine sees the attention layers as a plain FULL paged-KV group (short contexts are
exact dense attention, and the model handles sparse selection itself) plus a per-token side
slab for the indexer keys (``ModelConfig.qsa_index_head_dim``). Both CUDA graphs and prefix
reuse are disabled for correctness (see ``eager_only`` / ``prefix_reuse_supported``).
"""

from __future__ import annotations

import os
from typing import Any

from freetoken.utils import init_logger

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.qwen3_5_moe.config import _expert_quant, _fp8_block_quant

from .args import Qwen4ExpArgs, build_ple_layer_spec

logger = init_logger(__name__)

_ATTN_TYPES = ("full_attention", "qwen_sparse_attention")


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _layer_types(text: Any) -> list[str]:
    layer_types = _get(text, "layer_types")
    n = int(_get(text, "num_hidden_layers"))
    if layer_types is None:
        interval = int(_get(text, "full_attention_interval", 4) or 4)
        layer_types = [
            "linear_attention" if (i + 1) % interval else "qwen_sparse_attention" for i in range(n)
        ]
    out = []
    for t in layer_types:
        if t in _ATTN_TYPES:
            out.append("qsa")
        elif t == "linear_attention":
            out.append("linear")
        else:
            raise ValueError(f"Unsupported Qwen4-Exp layer type {t!r}")
    assert len(out) == n, f"layer_types has {len(out)} entries for {n} layers"
    return out


def _eos_id(text: Any, hf_config: Any) -> int:
    eos = _get(text, "eos_token_id")
    if eos is None:
        eos = _get(hf_config, "eos_token_id")
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    if eos is None:
        raise ValueError("Qwen4-Exp PLE needs eos_token_id (the n-gram boundary / pad token)")
    return int(eos)


def parse_config(hf_config: Any) -> ModelConfig:
    text = _get(hf_config, "text_config") or hf_config

    hidden = int(_get(text, "hidden_size"))
    num_heads = int(_get(text, "num_attention_heads"))
    head_dim = int(_get(text, "head_dim") or hidden // num_heads)
    num_kv_heads = int(_get(text, "num_key_value_heads") or num_heads)
    vocab = int(_get(text, "vocab_size"))
    num_layers = int(_get(text, "num_hidden_layers"))

    rope = _get(text, "rope_parameters") or {}
    rope_theta = float(_get(rope, "rope_theta") or _get(text, "rope_theta") or 10000.0)
    partial = float(
        _get(rope, "partial_rotary_factor") or _get(text, "partial_rotary_factor") or 1.0
    )
    rope_type = _get(rope, "rope_type", "default") or "default"
    if rope_type != "default":
        raise NotImplementedError(f"Qwen4-Exp rope_type {rope_type!r} is not supported")
    rotary_dim = int(head_dim * partial)  # HF: int(head_dim * partial_rotary_factor)
    max_pos = int(_get(text, "max_position_embeddings"))

    kinds = _layer_types(text)
    # Dev/testing only: serve a TRUNCATED model (first N decoder layers) -- e.g. to smoke the
    # real checkpoint's loaders/kernels without its full expert banks. Output is not meaningful.
    cap = os.environ.get("FREETOKEN_QWEN4_MAX_LAYERS")
    if cap and 0 < int(cap) < num_layers:
        logger.warning(
            f"FREETOKEN_QWEN4_MAX_LAYERS: serving a TRUNCATED Qwen4-Exp model "
            f"({int(cap)}/{num_layers} layers) -- dev/testing only"
        )
        num_layers = int(cap)
        kinds = kinds[:num_layers]
    qsa_ids = tuple(i for i, k in enumerate(kinds) if k == "qsa")
    linear_ids = tuple(i for i, k in enumerate(kinds) if k == "linear")

    idx_heads = _get(text, "indexer_n_heads")
    if idx_heads is None:
        raise NotImplementedError(
            "Qwen4-Exp without a QSA indexer (indexer_n_heads unset) is not supported"
        )
    idx_kv = int(_get(text, "indexer_kv_heads") or 1)
    if idx_kv != 1:
        raise ValueError(f"Qwen4-Exp QSA requires indexer_kv_heads=1, got {idx_kv}")
    idx_dim = int(_get(text, "indexer_head_dim"))
    budget = int(_get(text, "indexer_budget"))
    ratio = int(_get(text, "indexer_compress_ratio"))
    if budget % ratio:
        raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
    if rotary_dim > idx_dim:
        raise ValueError(f"rotary_dim {rotary_dim} exceeds indexer_head_dim {idx_dim}")

    gate_type = _get(text, "output_gate_type") or _get(text, "hidden_act") or "silu"
    if gate_type not in ("sigmoid", "silu"):
        raise ValueError(f"Unsupported Qwen4-Exp output gate activation {gate_type!r}")

    # PLE: one-indexed layer ids (must be linear-attention layers, HF validate_architecture).
    ple_ids_1 = sorted(set(_get(text, "ple_layer_ids") or []))
    ngram_size = int(_get(text, "ngram_size", 3))
    heads_per_ngram = int(_get(text, "heads_per_ngram", 8))
    ple_embed_dim = int(_get(text, "ple_embed_dim") or hidden)
    ple_layers = []
    eos = _eos_id(text, hf_config) if ple_ids_1 else int(_get(text, "eos_token_id") or 0)
    for ple_index, lid1 in enumerate(ple_ids_1):
        lid = lid1 - 1
        if lid >= num_layers and cap:
            continue  # truncated (dev) model: PLE layer beyond the cap
        if not 0 <= lid < num_layers or kinds[lid] != "linear":
            raise ValueError(f"PLE layer id {lid1} must be a linear-attention layer")
        ple_layers.append(build_ple_layer_spec(
            layer_id=lid, ple_index=ple_index, vocab_size=vocab, ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
            ngram_vocab_size_base=int(_get(text, "ngram_vocab_size_base", 20_000_000)),
            make_divisible_by=int(_get(text, "make_ngram_vocab_size_divisible_by", 128)),
            seed=int(_get(text, "seed", 1234)),
        ))
    if ple_layers:
        ngram_heads = (ngram_size - 1) * heads_per_ngram
        if ngram_heads <= 0 or ple_embed_dim % ngram_heads:
            raise ValueError(f"ple_embed_dim {ple_embed_dim} not divisible by {ngram_heads} heads")

    args = Qwen4ExpArgs(
        hidden_size=hidden,
        vocab_size=vocab,
        hc_count=int(_get(text, "hc_count", 4)),
        hc_lowrank=int(_get(text, "hc_lowrank", 320)),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
        max_position=max_pos,
        indexer_n_heads=int(idx_heads),
        indexer_head_dim=idx_dim,
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
        qsa_layer_ids=qsa_ids,
        output_gate_type=gate_type,
        ple_layers=tuple(ple_layers),
        ple_embed_dim=ple_embed_dim,
        ple_conv_kernel_size=int(_get(text, "ple_conv_kernel_size", 4)),
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        eos_token_id=eos,
        model_path=getattr(hf_config, "_name_or_path", None) or None,
    )
    if args.hc_count <= 1:
        raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {args.hc_count}")

    # Routed-expert quant only: block-fp8 (Qwen/...-FP8) or modelopt NVFP4 (RadixArk/...-NVFP4).
    # Both checkpoints keep EVERYTHING else bf16 (attention, GDN, shared expert, gates,
    # hyper-connections, PLE projections, embeddings, lm_head), so attn/dense/lm_head quant
    # stay "none" regardless of what the Qwen3.5 detectors would infer.
    expert_quant, block = _fp8_block_quant(hf_config)
    if expert_quant == "none":
        expert_quant = _expert_quant(hf_config)
    if expert_quant not in ("none", "nvfp4", "fp8_block"):
        raise NotImplementedError(f"Qwen4-Exp expert quant {expert_quant!r} is not supported")

    rotary = RotaryConfig(
        head_dim=head_dim, rotary_dim=rotary_dim, max_position=max_pos, base=rope_theta,
        scaling=None,
    )
    full_group = FullAttentionGroupConfig(
        name="full", layer_ids=qsa_ids, num_kv_heads=num_kv_heads, head_dim=head_dim,
        rotary_config=rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=int(_get(text, "linear_num_key_heads")),
        num_value_heads=int(_get(text, "linear_num_value_heads")),
        key_head_dim=int(_get(text, "linear_key_head_dim")),
        value_head_dim=int(_get(text, "linear_value_head_dim")),
        conv_kernel_dim=int(_get(text, "linear_conv_kernel_dim", 4)),
        output_gate=True,
    )
    groups = tuple(
        sorted(
            (g for g in (full_group, linear_group) if g.layer_ids),
            key=lambda g: g.layer_ids[0],
        )
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=vocab,
        intermediate_size=0,
        hidden_act=str(_get(text, "hidden_act", "silu")),
        rms_norm_eps=float(_get(text, "rms_norm_eps", 1e-6)),
        tie_word_embeddings=bool(
            _get(text, "tie_word_embeddings", False) or _get(hf_config, "tie_word_embeddings", False)
        ),
        rotary_config=rotary,
        num_experts=int(_get(text, "num_experts")),
        num_experts_per_tok=int(_get(text, "num_experts_per_tok")),
        moe_intermediate_size=int(_get(text, "moe_intermediate_size")),
        shared_expert_intermediate_size=int(_get(text, "shared_expert_intermediate_size")),
        norm_topk_prob=bool(_get(text, "norm_topk_prob", True)),
        moe_enabled=True,
        use_qk_norm=True,
        model_type=str(_get(hf_config, "model_type", "qwen4_exp")),
        architectures=list(
            _get(hf_config, "architectures") or ["Qwen4ExpForConditionalGeneration"]
        ),
        vision_config=None,  # text-only
        image_token_id=_get(hf_config, "image_token_id"),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=block,
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        qwen4_args=args,
        qsa_index_head_dim=idx_dim,
        eager_only=True,
        prefix_reuse_supported=False,
    )


__all__ = ["parse_config"]
