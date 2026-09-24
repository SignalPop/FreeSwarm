"""GLM-5.3-Flash (``glm5_next``) hyperparameters.

Ported from the HF reference (``transformers/models/glm5_next/configuration_glm5_next.py``
and ``modeling_glm5_next.py``). The text tower is a hybrid of

* **KDA** (Kimi Delta Attention) linear-attention layers (``layer_types ==
  "linear_attention"``) -- a gated delta rule with a per-CHANNEL decay, a depthwise
  causal short conv on q/k/v, and a sigmoid-gated RMSNorm on the output, and
* **MLA + DSA** layers (``"deepseek_sparse_attention"``) -- DeepSeek-V3 low-rank MLA
  with **NoPE** (``qk_rope_head_dim == 0``: no rotary anywhere in the text tower) plus
  a DSA indexer that scores *k-pools* (``index_kpool`` consecutive tokens compressed by
  a learned softmax gate) instead of single tokens, and always appends the current
  incomplete pool ("tail").

Every block is wrapped in manifold-constrained Hyper-Connections (``hc_mult`` residual
streams, Sinkhorn-projected mixing), the final streams collapse by an unweighted mean,
and every SwiGLU (dense MLP, shared and routed experts) clamps gate/up at
``swiglu_limit``. This payload rides ``ModelConfig.glm_dsa_args`` (so the generic ``dsa``
attention backend, which the engine picks for the DSA layer group, can read the MLA
dims it needs for its metadata/graph staging); the model module owns the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

LINEAR = "linear_attention"
DSA = "deepseek_sparse_attention"


def _get(cfg: Any, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    value = getattr(cfg, name, default)
    return default if value is None else value


@dataclass(frozen=True)
class Glm5NextArgs:
    hidden_size: int
    num_layers: int
    layer_types: Tuple[str, ...]
    mlp_layer_types: Tuple[str, ...]
    norm_eps: float
    # MLA (NoPE)
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    # DSA k-pool indexer
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_always_select_tail: bool
    indexer_types: Tuple[str, ...]
    # KDA linear attention
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float | None
    # Hyper-connections
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    # SwiGLU clamp (dense MLP + shared/routed experts)
    swiglu_limit: float
    max_position: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_dim(self) -> int:
        # NoPE MLA: the paged latent row is just the normalized c_kv.
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def index_state_dim(self) -> int:
        # Per-token row in the DSA index slab: indexer key | compress-gate scores | the
        # pooled (compressed) key of the pool this token CLOSES (only meaningful on rows
        # whose position % index_kpool == index_kpool - 1; see attention.py).
        return 3 * self.index_head_dim

    @property
    def rope_theta(self) -> float:  # NoPE; kept for code that reads GLM-5.2-shaped args
        return 0.0

    @property
    def dsa_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == DSA)

    @property
    def linear_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == LINEAR)

    @property
    def indexer_layer_ids(self) -> Tuple[int, ...]:
        """DSA layers that run their own indexer (and own an index-slab slot)."""
        return tuple(i for i in self.dsa_layer_ids if self.indexer_types[i] == "full")

    @property
    def first_k_dense_replace(self) -> int:
        n = 0
        for t in self.mlp_layer_types:
            if t != "dense":
                break
            n += 1
        return n


def _layer_types(text: Any, num_layers: int) -> list[str]:
    types = _get(text, "layer_types")
    if types is None:
        # HF default: every 4th layer (idx % 4 == 3) is DSA, the rest KDA.
        la = _get(text, "linear_attn_config") or {}
        kda = la.get("kda_layers") if isinstance(la, dict) else None
        if kda is not None:
            types = [LINEAR if i in set(kda) else DSA for i in range(num_layers)]
        else:
            types = [LINEAR if i % 4 != 3 else DSA for i in range(num_layers)]
    return [DSA if t == "full_attention" else t for t in list(types)[:num_layers]]


def _indexer_types(text: Any, num_layers: int) -> list[str]:
    types = _get(text, "indexer_types")
    if types is not None:
        return list(types)
    pattern = _get(text, "index_topk_pattern")
    if pattern is not None:
        return [{"F": "full", "S": "shared"}[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
    freq = max(int(_get(text, "index_topk_freq", 1)), 1)
    offset = int(_get(text, "index_skip_topk_offset", 2))
    return ["full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(num_layers)]


def load_args(text: Any, num_layers: int | None = None) -> Glm5NextArgs:
    """Build the args from the HF *text* config (object from AutoConfig, or the raw
    ``text_config`` dict/shim when the installed transformers predates glm5_next).
    Mirrors ``Glm5NextTextConfig.__post_init__``'s defaulting."""
    total_layers = int(_get(text, "num_hidden_layers"))
    num_layers = total_layers if num_layers is None else num_layers

    mlp_types = _get(text, "mlp_layer_types")
    if mlp_types is None:
        k = int(_get(text, "first_k_dense_replace", 3))
        mlp_types = ["dense"] * min(k, total_layers) + ["sparse"] * (total_layers - k)

    la = _get(text, "linear_attn_config") or {}
    if not isinstance(la, dict):
        la = {}
    linear_heads = int(la.get("num_heads", _get(text, "linear_num_heads", 64)))
    linear_head_dim = int(la.get("head_dim", _get(text, "linear_head_dim", 128)))
    conv_k = int(la.get("short_conv_kernel_size", _get(text, "linear_conv_kernel_dim", 4)))
    if "gate_lower_bound" in la:
        lower = la["gate_lower_bound"]
    elif isinstance(text, dict):
        lower = text.get("linear_lower_bound", -5.0)
    else:
        lower = getattr(text, "linear_lower_bound", -5.0)
    if lower is None and la and la.get("safe_gate", True):
        lower = -5.0

    qk_rope = int(_get(text, "qk_rope_head_dim", 0))
    if qk_rope != 0:
        raise ValueError(
            f"GLM-5.3 (glm5_next) MLA is NoPE; qk_rope_head_dim={qk_rope} is not supported"
        )
    index_topk = int(_get(text, "index_topk", 2048))
    kpool = int(_get(text, "index_kpool", 16))
    if kpool < 1 or index_topk % kpool:
        raise ValueError(f"index_topk ({index_topk}) must be a multiple of index_kpool ({kpool})")

    layer_types = _layer_types(text, total_layers)
    indexer_types = _indexer_types(text, total_layers)
    args = Glm5NextArgs(
        hidden_size=int(_get(text, "hidden_size")),
        num_layers=num_layers,
        layer_types=tuple(layer_types[:num_layers]),
        mlp_layer_types=tuple(list(mlp_types)[:num_layers]),
        norm_eps=float(_get(text, "rms_norm_eps", 1e-5)),
        num_heads=int(_get(text, "num_attention_heads")),
        q_lora_rank=int(_get(text, "q_lora_rank")),
        kv_lora_rank=int(_get(text, "kv_lora_rank")),
        qk_nope_head_dim=int(_get(text, "qk_nope_head_dim")),
        qk_rope_head_dim=qk_rope,
        v_head_dim=int(_get(text, "v_head_dim")),
        index_n_heads=int(_get(text, "index_n_heads", 32)),
        index_head_dim=int(_get(text, "index_head_dim", 128)),
        index_topk=index_topk,
        index_kpool=kpool,
        index_kpool_always_select_tail=bool(_get(text, "index_kpool_always_select_tail", True)),
        indexer_types=tuple(indexer_types[:num_layers]),
        linear_num_heads=linear_heads,
        linear_head_dim=linear_head_dim,
        linear_conv_kernel_dim=conv_k,
        linear_lower_bound=None if lower is None else float(lower),
        hc_mult=int(_get(text, "hc_mult", 4)),
        hc_eps=float(_get(text, "hc_eps", 1e-6)),
        hc_sinkhorn_iters=int(_get(text, "hc_sinkhorn_iters", 20)),
        swiglu_limit=float(_get(text, "swiglu_limit", 10.0)),
        max_position=int(_get(text, "max_position_embeddings", 1048576)),
    )
    _validate(args)
    return args


def _validate(args: Glm5NextArgs) -> None:
    if not args.dsa_layer_ids:
        raise ValueError("glm5_next: expected at least one deepseek_sparse_attention layer")
    for lid in args.dsa_layer_ids:
        if args.indexer_types[lid] == "shared":
            # The reference passes a "shared" layer only the top-k of the IMMEDIATELY
            # preceding layer (a KDA layer in between yields none and raises there).
            prev = lid - 1
            if prev < 0 or args.layer_types[prev] != DSA or args.indexer_types[prev] != "full":
                raise ValueError(
                    f"glm5_next: shared-indexer DSA layer {lid} must follow a full-indexer "
                    "DSA layer (HF reference semantics)"
                )
    k = args.first_k_dense_replace
    if any(t != "sparse" for t in args.mlp_layer_types[k:]):
        raise ValueError(
            f"glm5_next: dense MLP layers must be a contiguous prefix, got {args.mlp_layer_types}"
        )
    if args.index_head_dim % 2 or args.linear_head_dim <= 0:
        raise ValueError("glm5_next: bad indexer/linear head dims")


__all__ = ["Glm5NextArgs", "load_args", "LINEAR", "DSA"]
