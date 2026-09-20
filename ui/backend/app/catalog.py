"""Discovery of local model checkpoints, and the path guard for launching them.

A checkpoint here is any directory holding a `config.json` plus weights (safetensors or
a GGUF file). We scan the configured roots shallowly -- a HuggingFace hub cache nests
weights under `models--org--name/snapshots/<sha>/`, so that layout gets a dedicated pass.

`resolve_model_path` is the security boundary for the whole control plane: the start
endpoint accepts a model *id* from the browser and turns it into a path only if that path
lands inside one of the configured roots. Without it, a POST to a loopback port would be
able to spawn a process pointed at any directory on the machine.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import functools

from .config import settings

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")


def _is_checkpoint(d: Path) -> bool:
    if not d.is_dir():
        return False
    try:
        if (d / "config.json").is_file():
            return any(
                child.suffix in _WEIGHT_SUFFIXES for child in d.iterdir() if child.is_file()
            )
        # GGUF single-file checkpoints carry their config inside the file.
        return any(child.suffix == ".gguf" for child in d.iterdir() if child.is_file())
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def supported_architectures() -> frozenset[str]:
    """Architecture names FreeToken can actually serve.

    Read from the engine's own registry rather than duplicated here, so the list cannot
    drift. Importing it costs a few seconds (it pulls the freetoken package), hence the
    cache; a failure is treated as "cannot tell" so discovery still works.
    """
    try:
        from freetoken.models.register import _MODEL_REGISTRY  # noqa: PLC0415

        return frozenset(_MODEL_REGISTRY)
    except Exception:  # noqa: BLE001 - discovery must not depend on the engine importing
        return frozenset()


def _lookup(cfg: dict, *keys: str):
    """First present key, searching the top level then any nested sub-config.

    Multimodal checkpoints (Qwen3.6-35B-A3B, Gemma-4, MiniMax-M3) put the language model's
    real fields under `text_config`, leaving the top level almost empty. Reading only the
    top level makes an MoE model look dense -- which then picks a backend that cannot fit.
    """
    for key in keys:
        if cfg.get(key) not in (None, ""):
            return cfg[key]
    for sub_key in ("text_config", "language_config", "llm_config"):
        sub_cfg = cfg.get(sub_key)
        if isinstance(sub_cfg, dict):
            for key in keys:
                if sub_cfg.get(key) not in (None, ""):
                    return sub_cfg[key]
    return None


def _read_meta(d: Path) -> dict:
    """Best-effort architecture/param info from config.json for the model card."""
    meta: dict = {
        "architecture": None, "num_experts": None, "quantization": None,
        "max_position_embeddings": None, "hidden_size": None,
        "is_moe": False, "supported": True, "unsupported_reason": None,
        # "llm" (served by the FreeToken engine) or "timeseries" (served by tsfm_server).
        "category": "llm", "ts_servable": False, "ts_note": None,
        "kv_bytes_per_token": None,
    }
    cfg_path = d / "config.json"
    if not cfg_path.is_file():
        # GGUF carries its metadata inside the file; assume serveable rather than
        # blocking it on a config.json it will never have.
        return meta
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return meta

    archs = cfg.get("architectures")
    arch = archs[0] if isinstance(archs, list) and archs else None
    meta["architecture"] = arch

    experts = _lookup(cfg, "num_experts", "num_local_experts", "n_routed_experts")
    if experts:
        meta["num_experts"] = int(experts)
        meta["is_moe"] = True

    quant = cfg.get("quantization_config") or _lookup(cfg, "quantization_config") or {}
    if isinstance(quant, dict):
        meta["quantization"] = quant.get("quant_method") or quant.get("fmt")
    meta["max_position_embeddings"] = _lookup(cfg, "max_position_embeddings")
    meta["kv_bytes_per_token"] = _kv_bytes_per_token(cfg)
    meta["hidden_size"] = _lookup(cfg, "hidden_size")

    # Time-series models are a category of their own, not "unsupported LLMs". They were
    # listed with a red "FreeToken has no implementation" error, which was true of the LLM
    # engine and misleading about the app -- they run, just on a different server.
    from .tsfm import classify

    ts = classify(cfg)
    if ts is not None:
        meta.update(ts)
        meta["supported"] = False  # not as an LLM: keeps it out of the LLM launch panel
        meta["unsupported_reason"] = None
        return meta

    known = supported_architectures()
    if known and arch and arch not in known:
        meta["supported"] = False
        meta["unsupported_reason"] = (
            f"FreeToken has no implementation for {arch}. It serves causal language "
            "models; this checkpoint is a different kind of model."
        )
    return meta


def _kv_bytes_per_token(cfg: dict) -> int | None:
    """KV-cache bytes one token of context costs (bf16), or None if the layout is unfamiliar.

    Only full-attention layers keep a per-token cache: the linear-attention (GDN) layers of
    Qwen3.5/3.6 keep a fixed-size state and the sliding-window layers of gpt-oss keep a bounded
    window, so a hybrid's per-token cost is a fraction of its layer count suggests. That is why
    a 64K context costs only 1-2 GiB on these models -- worth showing before someone accepts an
    8K window that an agent overflows after two tool calls.
    """
    t = cfg.get("text_config") or cfg
    try:
        layers = int(t["num_hidden_layers"])
        types = t.get("layer_types")
        full = sum(1 for x in types if x == "full_attention") if types else layers
        if t.get("kv_lora_rank"):  # MLA (DeepSeek): one compressed latent per layer
            return full * (int(t["kv_lora_rank"]) + int(t.get("qk_rope_head_dim") or 64)) * 2
        kv_heads = int(t.get("num_key_value_heads") or t["num_attention_heads"])
        head_dim = int(t.get("head_dim") or int(t["hidden_size"]) // int(t["num_attention_heads"]))
        return full * 2 * kv_heads * head_dim * 2
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _expert_bytes(d: Path) -> int:
    """Bytes of MoE expert weights, read from the safetensors headers (no tensor data).

    0 when there are no experts or the layout cannot be parsed -- callers treat that as
    "unknown" and fall back to the checkpoint size rather than assuming nothing is pinned.
    """
    import json
    import struct

    total = experts = 0
    try:
        shards = sorted(d.glob("*.safetensors"))
    except OSError:
        return 0
    for shard in shards:
        try:
            with open(shard, "rb") as fh:
                length = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(length))
        except (OSError, ValueError, struct.error):
            return 0
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            try:
                begin, end = meta["data_offsets"]
            except (KeyError, TypeError, ValueError):
                continue
            total += end - begin
            if ".experts." in name or ".mlp.expert" in name:
                experts += end - begin
    return experts


def _dir_size_bytes(d: Path) -> int:
    total = 0
    try:
        for child in d.iterdir():
            if child.is_file() and child.suffix in _WEIGHT_SUFFIXES:
                total += child.stat().st_size
    except OSError:
        pass
    return total


def _hub_entries(root: Path) -> list[Path]:
    """`models--org--name/snapshots/<sha>` directories inside a HF hub cache."""
    out: list[Path] = []
    try:
        for repo in root.iterdir():
            if not repo.is_dir() or not repo.name.startswith("models--"):
                continue
            snapshots = repo / "snapshots"
            if not snapshots.is_dir():
                continue
            try:
                for snap in snapshots.iterdir():
                    if _is_checkpoint(snap):
                        out.append(snap)
            except OSError:
                continue
    except OSError:
        pass
    return out


def _display_name(path: Path, root: Path) -> str:
    # HF cache: models--openai--gpt-oss-20b/snapshots/<sha> -> openai/gpt-oss-20b
    for part in path.parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


# A scan opens every shard of every checkpoint to read its safetensors header, so its cost
# scales with the weights on disk, not with the number of models: ~7s cold over a few
# hundred GB, and the console polls /api/models. Worse, the engines stream weights through
# host RAM, so the pages this scan warms get evicted and the next poll pays full price
# again -- the Models page sat on "Scanning..." indefinitely. The result only changes when
# a download finishes or a file is added by hand, so a short TTL costs nothing and a new
# checkpoint still appears within it. `invalidate_models()` forces it sooner.
_SCAN_TTL_S = 20.0
_scan_lock = threading.Lock()
_scan_cache: tuple[float, list[dict]] | None = None


def invalidate_models() -> None:
    """Drop the cached scan -- call after a download lands or a checkpoint is removed."""
    global _scan_cache
    with _scan_lock:
        _scan_cache = None


def list_models(force: bool = False) -> list[dict]:
    """Every checkpoint under the configured roots, de-duplicated by resolved path.

    Cached for `_SCAN_TTL_S`; `force=True` rescans now.
    """
    global _scan_cache
    if not force:
        with _scan_lock:
            if _scan_cache and time.monotonic() - _scan_cache[0] < _SCAN_TTL_S:
                return _scan_cache[1]
    out = _scan_models()
    with _scan_lock:
        _scan_cache = (time.monotonic(), out)
    return out


def _scan_models() -> list[dict]:
    seen: dict[str, dict] = {}
    for root in settings.model_roots:
        if not root.is_dir():
            continue
        candidates: list[Path] = []
        if _is_checkpoint(root):
            candidates.append(root)
        try:
            for child in root.iterdir():
                if _is_checkpoint(child):
                    candidates.append(child)
        except OSError:
            pass
        candidates.extend(_hub_entries(root))

        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            entry = {
                "id": _display_name(path, root),
                "path": key,
                "root": str(root),
                "size_bytes": _dir_size_bytes(path),
                # What an OFFLOADED engine page-locks in host RAM -- the experts only, not
                # the whole checkpoint. The console budgets the WDDM pinning ceiling against
                # this; using size_bytes there over-counted by the non-expert weights and
                # warned about models that actually fit (Qwen3.6: 67.0 GiB total but 61.5
                # GiB of experts, against a 66.5 GiB limit).
                "expert_bytes": _expert_bytes(path),
            }
            entry.update(_read_meta(path))
            seen[key] = entry
    return sorted(seen.values(), key=lambda m: m["id"].lower())


def model_meta_for(path: Path) -> dict:
    """Metadata for one checkpoint directory, for pre-launch checks."""
    return _read_meta(path)


def resolve_model_path(raw: str) -> Path:
    """Turn a client-supplied model id or path into a checkpoint directory.

    Raises ValueError unless the resolved path is a real checkpoint that sits inside one of
    the configured model roots. Both conditions matter: `is_relative_to` alone would still
    accept a non-model directory, and the checkpoint test alone would accept any path on
    the machine.
    """
    if not raw or not raw.strip():
        raise ValueError("model must not be empty")

    candidate: Path | None = None
    for entry in list_models():
        if raw == entry["id"] or raw == entry["path"]:
            candidate = Path(entry["path"])
            break
    if candidate is None:
        candidate = Path(raw).expanduser()
        try:
            candidate = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"unknown model: {raw!r}") from exc

    if not any(
        candidate == root.resolve() or candidate.is_relative_to(root.resolve())
        for root in settings.model_roots
        if root.exists()
    ):
        roots = ", ".join(str(r) for r in settings.model_roots)
        raise ValueError(
            f"{candidate} is outside the configured model roots ({roots}). "
            "Set FREESWARM_MODELS_DIR to add a root."
        )
    if not _is_checkpoint(candidate):
        raise ValueError(f"{candidate} does not look like a checkpoint (no config.json/weights)")
    return candidate
