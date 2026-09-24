"""Weight loading for GLM-5.3-Flash (``glm5_next``).

Checkpoint layout (nvidia/GLM-5.3-Flash-NVFP4, modelopt): text tower under
``model.language_model.``, vision under ``model.visual.`` (dropped), ``lm_head.weight`` at
the root, and the MTP layer at ``layers.<num_hidden_layers>`` (skipped). Renames follow
the transformers ``glm5_next`` conversion mapping, plus FreeToken's fusions:

* ``hc_{attn,ffn}_{fn,base,scale}``            -> ``{attn,ffn}_hc.{fn,base,scale}`` (fp32)
* KDA ``self_attn.{q,k,v}_proj``               -> ``linear_attn.qkv_proj`` (row concat)
* KDA ``self_attn.{q,k,v}_conv1d``             -> ``linear_attn.conv1d`` (fp32 concat)
* KDA ``self_attn.{f_a,f_b,g_a,g_b,b,o}_proj / dt_bias / A_log / o_norm``
                                               -> ``linear_attn.*``
* dense / shared-expert ``{gate,up}_proj``     -> ``gate_up_proj`` (row concat)
* ``mlp.gate.e_score_correction_bias``         -> ``mlp.e_score_correction_bias``

Quantized tensors (modelopt NVFP4: ``weight`` u8 + ``weight_scale`` fp8 + scalar
``weight_scale_2``) outside the routed experts -- the leading dense MLPs -- are
dequantized to bf16 here (W4A16 math, activation ``input_scale`` unused). Routed experts:
NVFP4 -> offload banks (``load_nvfp4_expert_sources``); bf16 (unquantized checkpoints,
e.g. the tiny test models) -> stacked resident banks when ``include_moe_experts``.

Both the original checkpoint names and the transformers-internal names (``forget_gate.*``,
``attn_hc.*``, fused ``conv1d``/``experts.gate_up_proj``, as written by a v5
``save_pretrained``) are accepted.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec, load_nvfp4_expert_source_banks
from freetoken.utils import cached_load_hf_config, download_hf_weight

from .args import LINEAR
from .config import parse_config

# Text-tower prefixes, most specific first ("" = a bare text-model state dict, e.g. a
# transformers Glm5NextTextModel in the tests).
_PREFIXES = ("model.language_model.", "model.", "language_model.model.", "language_model.", "")

_ROUTED_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_ROUTED_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    ),
    desc="GLM-5.3 NVFP4 experts",
)


class _Reader:
    """Name -> tensor over an index-mapped safetensors folder (lazy shard handles)."""

    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._map = weight_map
        self._device = device
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._map

    def get(self, name: str) -> torch.Tensor:
        shard = self._map[name]
        h = self._handles.get(shard)
        if h is None:
            h = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=str(self._device)
            ).__enter__()
            self._handles[shard] = h
        return h.get_tensor(name)

    def close(self) -> None:
        for shard, h in self._handles.items():
            try:
                h.__exit__(None, None, None)
            except Exception:  # pragma: no cover
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _open_reader(model_path: str, device: torch.device) -> _Reader:
    folder = download_hf_weight(model_path)
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    else:  # single-file checkpoint (tiny test models)
        weight_map = {}
        for fn in sorted(os.listdir(folder)):
            if fn.endswith(".safetensors"):
                with safetensors.safe_open(os.path.join(folder, fn), framework="pt") as f:
                    for k in f.keys():
                        weight_map[k] = fn
    return _Reader(folder, weight_map, device)


def _dequant_nvfp4(w, scale, scale_2) -> torch.Tensor:
    from freetoken.models.qwen3_5_moe.weight import _dequant_nvfp4_weight

    return _dequant_nvfp4_weight(w, scale, scale_2)


def iter_glm5_next_weights(
    has: Callable[[str], bool],
    get: Callable[[str], torch.Tensor],
    config,
    *,
    include_moe_experts: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Checkpoint (``has``/``get``) -> FreeToken state-dict items. Pure mapping logic,
    shared by the safetensors loader and the tests."""
    args = config.glm_dsa_args

    prefix = next(
        (p for p in _PREFIXES if has(f"{p}embed_tokens.weight")), None
    )
    assert prefix is not None, "glm5_next: cannot find the text tower (embed_tokens) in the checkpoint"

    def first(*names: str) -> str | None:
        return next((n for n in names if has(n)), None)

    def load(name: str) -> torch.Tensor:
        """Plain or NVFP4-quantized ``<base>.weight`` -> dense bf16/fp tensor."""
        if name.endswith(".weight"):
            base = name[: -len(".weight")]
            if has(base + ".weight_scale_2"):
                return _dequant_nvfp4(get(name), get(base + ".weight_scale"), get(base + ".weight_scale_2"))
            if has(base + ".weight_scale") and get(name).dtype == torch.float8_e4m3fn:
                return get(name).to(torch.bfloat16) * get(base + ".weight_scale").to(torch.bfloat16)
        return get(name)

    for layer in range(config.num_layers):
        src = f"{prefix}layers.{layer}."
        dst = f"model.layers.{layer}."
        # hyper-connections
        for site in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                n = first(f"{src}hc_{site}_{part}", f"{src}{site}_hc.{part}")
                assert n is not None, f"missing hc {site} {part} for layer {layer}"
                yield f"{dst}{site}_hc.{part}", get(n).float()
        for norm in ("input_layernorm", "post_attention_layernorm"):
            yield f"{dst}{norm}.weight", get(f"{src}{norm}.weight")

        a = f"{src}self_attn."
        if args.layer_types[layer] == LINEAR:
            o = f"{dst}linear_attn."
            yield o + "qkv_proj.weight", torch.cat(
                [load(f"{a}{p}_proj.weight") for p in ("q", "k", "v")], dim=0
            )
            if has(f"{a}conv1d.weight"):
                conv = get(f"{a}conv1d.weight")
            else:
                conv = torch.cat([get(f"{a}{p}_conv1d.weight") for p in ("q", "k", "v")], dim=0)
            yield o + "conv1d.weight", (conv.unsqueeze(1) if conv.dim() == 2 else conv).float()
            for p in ("f_a_proj", "f_b_proj"):
                n = first(f"{a}{p}.weight", f"{a}forget_gate.{p}.weight")
                yield f"{o}{p}.weight", load(n)
            for p in ("dt_bias", "A_log"):
                n = first(f"{a}{p}", f"{a}forget_gate.{p}")
                yield f"{o}{p}", get(n).float().reshape(-1)
            for p in ("b_proj", "g_a_proj", "g_b_proj", "o_proj"):
                yield f"{o}{p}.weight", load(f"{a}{p}.weight")
            yield o + "o_norm.weight", get(f"{a}o_norm.weight")
        else:
            o = f"{dst}self_attn."
            for p in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                yield f"{o}{p}.weight", load(f"{a}{p}.weight")
            for p in ("q_a_layernorm", "kv_a_layernorm"):
                yield f"{o}{p}.weight", get(f"{a}{p}.weight")
            if args.indexer_types[layer] == "full":
                i = f"{a}indexer."
                for p in ("wq_b", "wk", "weights_proj"):
                    yield f"{o}indexer.{p}.weight", load(f"{i}{p}.weight")
                yield f"{o}indexer.k_norm.weight", get(f"{i}k_norm.weight")
                yield f"{o}indexer.k_norm.bias", get(f"{i}k_norm.bias")
                for p in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
                    yield f"{o}indexer.{p}", get(f"{i}{p}")

        m = f"{src}mlp."
        om = f"{dst}mlp."
        if args.mlp_layer_types[layer] == "dense":
            yield om + "gate_up_proj.weight", torch.cat(
                [load(f"{m}gate_proj.weight"), load(f"{m}up_proj.weight")], dim=0
            )
            yield om + "down_proj.weight", load(f"{m}down_proj.weight")
            continue
        yield om + "gate.weight", get(f"{m}gate.weight")
        yield om + "e_score_correction_bias", get(f"{m}gate.e_score_correction_bias").float()
        s = f"{m}shared_experts."
        yield om + "shared_experts.gate_up_proj.weight", torch.cat(
            [load(f"{s}gate_proj.weight"), load(f"{s}up_proj.weight")], dim=0
        )
        yield om + "shared_experts.down_proj.weight", load(f"{s}down_proj.weight")
        if include_moe_experts:
            e = f"{m}experts."
            if has(e + "gate_up_proj"):  # transformers-internal fused layout
                yield om + "experts.gate_up_proj", get(e + "gate_up_proj")
                yield om + "experts.down_proj", get(e + "down_proj")
            else:
                if has(f"{e}0.gate_proj.weight_scale"):
                    raise NotImplementedError(
                        "glm5_next NVFP4 routed experts are served from the offload cache "
                        "(--moe-backend offload/auto), not as resident weights"
                    )
                n_exp = config.num_experts
                yield om + "experts.gate_up_proj", torch.stack([
                    torch.cat([get(f"{e}{x}.gate_proj.weight"), get(f"{e}{x}.up_proj.weight")], 0)
                    for x in range(n_exp)
                ])
                yield om + "experts.down_proj", torch.stack(
                    [get(f"{e}{x}.down_proj.weight") for x in range(n_exp)]
                )

    yield "model.embed_tokens.weight", get(f"{prefix}embed_tokens.weight")
    yield "model.norm.weight", get(f"{prefix}norm.weight")
    if not config.tie_word_embeddings:
        n = first("lm_head.weight", "language_model.lm_head.weight")
        assert n is not None, "glm5_next: missing lm_head.weight"
        yield "lm_head.weight", get(n)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    reader = _open_reader(model_path, device)
    try:
        yield from iter_glm5_next_weights(
            reader.has, reader.get, config, include_moe_experts=include_moe_experts
        )
    finally:
        reader.close()


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None) -> dict:
    """Pinned CPU NVFP4 banks for the routed experts (MoE layers
    ``[first_k_dense_replace, num_layers)``; the bf16 MTP layer is excluded)."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "iter_weights",
    "iter_glm5_next_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
