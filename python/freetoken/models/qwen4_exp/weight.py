"""Qwen4-Exp checkpoint -> FreeToken state dict.

Both released checkpoints store everything except the routed experts in bf16:

* ``Qwen/Qwen3.8-Flash-Next-FP8``: routed experts 128x128 block-fp8 (``weight_scale_inv``);
* ``RadixArk/Qwen3.8-Flash-Next-NVFP4``: routed experts modelopt NVFP4 (``weight`` uint8 +
  ``weight_scale`` fp8 + ``weight_scale_2``), everything else bf16.

The PLE n-gram table (``...ple.ple_embedding.ngram_embedding.shard_*``) and its hashing
buffers are NOT part of the state dict: the table is memory-mapped from the shards at
``prepare_for_runtime`` (see ``ple.py``) and the buffers are recomputed from the config.
MTP (``mtp.*``) and the vision tower are dropped.

Routed experts:
* resident (``--moe-backend fused``): bf16 per-expert tensors are stacked per layer into
  ``model.layers.N.mlp.experts.{gate_up_proj,down_proj}``; block-fp8 reuses the Qwen3.5
  stacked-bank reader (same per-expert key layout);
* offload: the Qwen3.5 bank providers (``setup_offload_expert_banks``; the NVFP4 source spec
  reuses the Qwen3.5 key regex) -- the per-expert key layout is identical. The
  ``FREETOKEN_QWEN4_MAX_LAYERS`` dev cap is honored by the dense pass, bf16 stacking and the
  NVFP4 banks (not by the block-fp8 bank reader).
"""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
    load_nvfp4_expert_source_banks_parallel,
)
from freetoken.models.qwen3_5_moe.weight import (
    _NVFP4_EXPERT_KEY_RE,
    _build_fp8_expert_banks,
    _dequant_fp8_weight,
    _dequant_nvfp4_weight,
    setup_offload_expert_banks,
)
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_PACKED_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<proj>gate_up_proj|down_proj)$"
)
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale", ".weight_scale_inv")
_LAYER_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.")

# NVFP4 routed experts (RadixArk modelopt checkpoint): same per-expert key layout as the
# Qwen3.5 modelopt checkpoints; layers past a (dev) FREETOKEN_QWEN4_MAX_LAYERS cap are skipped.
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer if layer < config.num_layers else None,
    desc="Qwen4-Exp NVFP4 experts",
)

# fused model buffer suffix -> ordered checkpoint parts (concatenated on dim 0)
_FUSIONS: dict[str, tuple[str, ...]] = {
    ".self_attn.qkv_proj.weight": (
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ),
    ".linear_attn.in_proj.weight": (
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ),
    ".mlp.shared_expert.gate_up_proj.weight": (
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ),
}


def _rename(raw: str) -> str | None:
    if raw.startswith(("mtp.", "model.mtp.", "model.visual.", "visual.")):
        return None
    if ".ple.ple_embedding." in raw:
        return None  # n-gram table (memory-mapped separately) + recomputed hash buffers
    if raw.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    if raw.startswith("model.language_model."):
        return "model." + raw[len("model.language_model."):]
    if raw.startswith("language_model."):
        return "model." + raw[len("language_model."):]
    return raw


def _load_dense(f, raw: str, keys: set[str]) -> torch.Tensor:
    """Load a dense tensor, dequantizing any quantized ``.weight`` to bf16 (defensive: the
    released checkpoints keep all dense weights bf16)."""
    t = f.get_tensor(raw)
    if not raw.endswith(".weight"):
        return t
    base = raw[: -len(".weight")]
    if base + ".weight_scale_2" in keys:
        return _dequant_nvfp4_weight(t, f.get_tensor(base + ".weight_scale"),
                                     f.get_tensor(base + ".weight_scale_2"))
    if base + ".weight_scale_inv" in keys:
        from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8

        dev = t.device if t.device.type == "cuda" else torch.device("cuda")
        return dequant_block_fp8(t.to(dev), f.get_tensor(base + ".weight_scale_inv").to(dev)).to(t.device)
    if base + ".weight_scale" in keys:
        return _dequant_fp8_weight(t, f.get_tensor(base + ".weight_scale"))
    return t


def _try_fuse(name: str, tensor: torch.Tensor, buf: dict):
    for fused, parts in _FUSIONS.items():
        for i, part in enumerate(parts):
            if name.endswith(part):
                key = name[: -len(part)] + fused
                slots = buf.setdefault(key, {})
                slots[i] = tensor
                if len(slots) == len(parts):
                    del buf[key]
                    return key, torch.cat([slots[j] for j in range(len(parts))], dim=0)
                return ()
    return None


def _beyond(raw: str, num_layers: int) -> bool:
    m = _LAYER_RE.match(raw)
    return m is not None and int(m.group(1)) >= num_layers


def _iter_dense(model_path: str, device: torch.device,
                num_layers: int) -> Iterator[tuple[str, torch.Tensor]]:
    fuse_buf: dict = {}
    for file in tqdm(iter_weight_files(model_path), desc="Loading weights",
                     disable=not get_tp_info().is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            keys = set(f.keys())
            for raw in f.keys():
                if ".mlp.experts." in raw:
                    continue  # routed experts: resident pass below / offload banks
                if raw.endswith(_SCALE_SUFFIXES):
                    continue  # consumed with their .weight
                if _beyond(raw, num_layers):
                    continue  # truncated (dev) model
                name = _rename(raw)
                if name is None:
                    continue
                tensor = _load_dense(f, raw, keys)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():
                        yield fused
                    continue
                yield name, tensor
    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"


def _iter_bf16_experts(model_path: str, config, device: torch.device):
    """Stack per-expert bf16 tensors (or pass through pre-packed ones) per layer."""
    E, I = config.num_experts, config.moe_intermediate_size
    pending: dict[int, dict] = {}
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw in f.keys():
                m = _PACKED_EXPERT_RE.match(raw)
                if m is not None:
                    if int(m["layer"]) >= config.num_layers:
                        continue
                    yield f"model.layers.{m['layer']}.mlp.experts.{m['proj']}", f.get_tensor(raw)
                    continue
                m = _EXPERT_RE.match(raw)
                if m is None:
                    continue
                li, e, proj = int(m["layer"]), int(m["expert"]), m["proj"]
                if li >= config.num_layers:
                    continue  # truncated (dev) model
                t = f.get_tensor(raw)
                slot = pending.get(li)
                if slot is None:
                    H = t.shape[1] if proj != "down_proj" else t.shape[0]
                    slot = pending[li] = {
                        "gate_up": torch.empty(E, 2 * I, H, dtype=t.dtype, device=device),
                        "down": torch.empty(E, H, I, dtype=t.dtype, device=device),
                        "n": 0,
                    }
                if proj == "gate_proj":
                    slot["gate_up"][e, :I].copy_(t)
                elif proj == "up_proj":
                    slot["gate_up"][e, I:].copy_(t)
                else:
                    slot["down"][e].copy_(t)
                slot["n"] += 1
                if slot["n"] == 3 * E:
                    del pending[li]
                    yield f"model.layers.{li}.mlp.experts.gate_up_proj", slot["gate_up"]
                    yield f"model.layers.{li}.mlp.experts.down_proj", slot["down"]
    assert not pending, f"incomplete expert layers: {sorted(pending)}"


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading currently supports TP=1 only")
    config = parse_config(cached_load_hf_config(model_path))
    if include_non_moe:
        yield from _iter_dense(model_path, device, config.num_layers)
    if not include_moe_experts:
        return
    if config.expert_quant == "fp8_block":
        banks = _build_fp8_expert_banks(model_path, config, dummy=False, pin=False)
        for li in range(config.num_layers):
            pre = f"model.layers.{li}.mlp.experts"
            yield f"{pre}.gate_up_proj", banks["gate_up"][li]
            yield f"{pre}.gate_up_scale_inv", banks["gate_up_scale"][li]
            yield f"{pre}.down_proj", banks["down"][li]
            yield f"{pre}.down_scale_inv", banks["down_scale"][li]
    elif config.expert_quant == "none":
        yield from _iter_bf16_experts(model_path, config, device)
    else:
        raise ValueError(f"{config.expert_quant} experts are served by the offload backends only")


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    """CPU NVFP4 expert source banks for the offload cache (gate/up fused on the output-row
    axis, down separate; ``weight_scale_2`` carried as the per-row global scale)."""
    return load_nvfp4_expert_source_banks(
        model_path, config, _NVFP4_SOURCE_SPEC, drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(), layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(model_path: str, config, *, workers: int = 8,
                                       chunk: int = 8 << 20, layer_sink=None):
    """Same NVFP4 source banks via the common chunked multi-threaded reader."""
    return load_nvfp4_expert_source_banks_parallel(
        model_path, config, _NVFP4_SOURCE_SPEC, drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(), workers=workers, chunk=chunk, layer_sink=layer_sink,
    )


__all__ = [
    "iter_weights",
    "setup_offload_expert_banks",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
]
