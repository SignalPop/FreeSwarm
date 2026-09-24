"""Qwen4-Exp PLE (Per-Layer Embedding): hashed 2/3-gram embeddings injected into the
hyper-connection residual stream (port of HF ``Qwen4ExpTextNGramEmbedding`` +
``Qwen4ExpTextPLELayer``).

The n-gram table is huge (Qwen3.8-Flash-Next: 128 shards x 2.5M rows x 160 = ~320M rows,
fp8-e4m3 + a per-tensor bf16 ``weight_scale``, ~51 GB), so it never enters VRAM (nor, by
default, host RAM): each checkpoint shard is memory-mapped read-only and the ~16 rows a
token needs are gathered on the CPU, then copied to the GPU and dequantized. Set
``FREETOKEN_QWEN4_PLE_PRELOAD=1`` to copy the table into host RAM up front instead.

Per-request state (keyed by ``Req.table_idx``):

* the last ``ngram_size - 1`` token ids (host int64; HF keeps them in ``conv_states[2]``),
  reset to ``eos`` when a request starts (``cached_len == 0``), so chunked prefill and decode
  hash exactly like one full-sequence forward;
* the dilated depthwise-conv state: the last ``(kernel - 1) * ngram_size`` normalized gated
  values (GPU; HF ``conv_states[1]``), zeroed on request start.

Neither state is snapshotted by the hybrid radix cache, hence ``prefix_reuse_supported=False``
for this model (the engine forces ``--cache-type naive``).
"""

from __future__ import annotations

import json
import math
import os
import struct
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.utils import init_logger

from .layers import QRMSNorm

if TYPE_CHECKING:
    from .args import PLELayerSpec, Qwen4ExpArgs

logger = init_logger(__name__)

_ST_NP = {
    "BF16": (np.uint16, torch.bfloat16),
    "F16": (np.float16, torch.float16),
    "F32": (np.float32, torch.float32),
    "F8_E4M3": (np.uint8, torch.float8_e4m3fn),
    "F8_E5M2": (np.uint8, torch.float8_e5m2),
}


# ------------------------------------------------------------------------------------------
# n-gram hashing (HF-exact)
# ------------------------------------------------------------------------------------------
def shift_right_ignore_eos(token_ids: torch.Tensor, shift: int, eos: int) -> torch.Tensor:
    """HF ``_shift_right_ignore_eos`` on a ``[B, S]`` int64 tensor: the token ``shift`` places
    back, or ``eos`` when that would cross an eos boundary / the start of the history."""
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos))


def ngram_row_ids(history: torch.Tensor, new_ids: torch.Tensor, spec: "PLELayerSpec",
                  ngram_size: int, heads_per_ngram: int, eos: int) -> torch.Tensor:
    """Table rows ``[n, ngram_heads]`` (int64) for ``new_ids`` [n] given the preceding
    ``ngram_size - 1`` tokens ``history`` (eos-padded at a sequence start)."""
    tokens = torch.cat([history, new_ids]).long().unsqueeze(0)  # [1, ctx + n]
    shifted = [shift_right_ignore_eos(tokens, s, eos) for s in range(ngram_size)]
    mult = torch.tensor(spec.multipliers, dtype=torch.long)
    sizes = torch.tensor(spec.head_vocab_sizes, dtype=torch.long)
    offsets = torch.tensor(spec.head_offsets, dtype=torch.long)
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * mult[0]  # int64 multiply wraps like the reference
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * mult[position])
        ids = torch.remainder(mixed.unsqueeze(-1), sizes[start:end].view(1, 1, -1))
        blocks.append(ids + offsets[start:end].view(1, 1, -1))
    return torch.cat(blocks, dim=-1)[0, -new_ids.numel():]


# ------------------------------------------------------------------------------------------
# host-resident n-gram table
# ------------------------------------------------------------------------------------------
def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _weight_map(folder: str) -> dict[str, str]:
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as fh:
            return json.load(fh)["weight_map"]
    out = {}
    for f in sorted(os.listdir(folder)):
        if f.endswith(".safetensors"):
            hdr, _ = _read_header(os.path.join(folder, f))
            out.update({k: f for k in hdr if k != "__metadata__"})
    return out


class NGramTable:
    """Row-gather view of one PLE module's n-gram embedding over memory-mapped checkpoint
    shards (``...ngram_embedding.shard_{i}.weight`` concatenated on dim 0, or an unsharded
    ``...ngram_embedding.weight``) plus the optional per-tensor ``weight_scale``."""

    def __init__(self, parts: list[tuple[np.ndarray, torch.dtype]], scale: float | None,
                 row_dim: int):
        self._parts = [p for p, _ in parts]
        self._torch_dtype = parts[0][1]
        self._starts = np.cumsum([0] + [p.shape[0] for p in self._parts])
        self.num_rows = int(self._starts[-1])
        self.scale = scale
        self.row_dim = row_dim

    @classmethod
    def from_checkpoint(cls, model_path: str, prefix: str, row_dim: int) -> "NGramTable":
        from freetoken.utils import download_hf_weight

        folder = download_hf_weight(model_path)
        wmap = _weight_map(folder)
        base = f"{prefix}.ngram_embedding"
        names = []
        if f"{base}.weight" in wmap:
            names = [f"{base}.weight"]
        else:
            i = 0
            while f"{base}.shard_{i}.weight" in wmap:
                names.append(f"{base}.shard_{i}.weight")
                i += 1
        if not names:
            raise FileNotFoundError(f"no n-gram table under {base!r} in {folder}")
        preload = os.getenv("FREETOKEN_QWEN4_PLE_PRELOAD", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        headers: dict[str, tuple[dict, int]] = {}
        parts = []
        for name in names:
            path = os.path.join(folder, wmap[name])
            if path not in headers:
                headers[path] = _read_header(path)
            hdr, data_start = headers[path]
            meta = hdr[name]
            np_dtype, t_dtype = _ST_NP[meta["dtype"]]
            rows, cols = meta["shape"]
            if cols != row_dim:
                raise ValueError(f"{name}: row width {cols} != expected {row_dim}")
            b, _e = meta["data_offsets"]
            mm = np.memmap(path, dtype=np_dtype, mode="r", offset=data_start + b,
                           shape=(rows, cols))
            parts.append((np.array(mm) if preload else mm, t_dtype))
        scale = None
        sname = f"{base}.weight_scale"
        if sname in wmap:
            path = os.path.join(folder, wmap[sname])
            from safetensors import safe_open

            with safe_open(path, framework="pt") as fh:
                scale = float(fh.get_tensor(sname).float().reshape(-1)[0])
        return cls(parts, scale, row_dim)

    @classmethod
    def from_tensor(cls, weight: torch.Tensor, scale: float | None = None) -> "NGramTable":
        w = weight.detach().cpu().contiguous()
        np_view = w.view(torch.uint8).numpy() if w.dtype in (
            torch.float8_e4m3fn, torch.float8_e5m2) else (
            w.view(torch.int16).numpy().view(np.uint16) if w.dtype == torch.bfloat16 else w.numpy()
        )
        return cls([(np_view, w.dtype)], scale, int(w.shape[1]))

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        """Host rows ``[N]`` (int64) -> host tensor ``[N, row_dim]`` in the table dtype."""
        r = rows.numpy()
        out = np.empty((r.shape[0], self.row_dim), dtype=self._parts[0].dtype)
        part = np.searchsorted(self._starts, r, side="right") - 1
        for p in np.unique(part):
            sel = np.nonzero(part == p)[0]
            out[sel] = self._parts[p][r[sel] - self._starts[p]]
        t = torch.from_numpy(out)
        if self._torch_dtype == torch.bfloat16:
            return t.view(torch.bfloat16)
        if self._torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return t.view(self._torch_dtype)
        return t


# ------------------------------------------------------------------------------------------
# the PLE layer
# ------------------------------------------------------------------------------------------
class Qwen4ExpPLE(BaseOP):
    """``forward(h [T, hc*H], ids_cpu [T]) -> h + ple(h)`` for the current batch."""

    def __init__(self, args: "Qwen4ExpArgs", spec: "PLELayerSpec", eps: float):
        self._args = args
        self._spec = spec
        self.layer_id = spec.layer_id
        H, hc = args.hidden_size, args.hc_count
        self.hidden_size = H
        self.hc_count = hc
        hc_hidden = H * hc
        self.key_proj = LinearReplicated(args.ple_embed_dim, hc_hidden, has_bias=False)
        self.value_proj = LinearReplicated(args.ple_embed_dim, H, has_bias=False)
        self.norm_key = QRMSNorm(hc_hidden, eps=eps, group_size=H)
        self.norm_query = QRMSNorm(hc_hidden, eps=eps, group_size=H)
        self.norm_conv = QRMSNorm(hc_hidden, eps=eps, group_size=H)
        self.conv1d = _Conv1dWeight(hc_hidden, args.ple_conv_kernel_size)
        self._dilation = args.ngram_size
        self._state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        self._ctx_len = args.ngram_size - 1
        self._table: NGramTable | None = None
        self._conv_state: torch.Tensor | None = None  # [rows, state_len, hc*H]
        self._history: torch.Tensor | None = None  # [rows, ngram_size-1] int64 (host)

    # -- table ---------------------------------------------------------------------------
    @property
    def table_prefixes(self) -> tuple[str, ...]:
        # multimodal wrapper checkpoints nest the text tower under model.language_model.
        tail = f"layers.{self.layer_id}.ple.ple_embedding"
        return (f"model.language_model.{tail}", f"model.{tail}")

    def set_table(self, table: NGramTable) -> None:
        if table.num_rows < self._spec.total_vocab_size:
            raise ValueError(
                f"PLE layer {self.layer_id}: table has {table.num_rows} rows, hashing needs "
                f"{self._spec.total_vocab_size}"
            )
        self._table = table

    def load_table(self, model_path: str | None = None) -> None:
        path = model_path or self._args.model_path
        if path is None:
            raise RuntimeError("Qwen4-Exp PLE: no checkpoint path to load the n-gram table from")
        err: Exception | None = None
        for prefix in self.table_prefixes:
            try:
                table = NGramTable.from_checkpoint(path, prefix, self._args.ngram_head_dim)
                break
            except FileNotFoundError as e:
                err = e
        else:
            raise err  # type: ignore[misc]
        self.set_table(table)
        logger.info_rank0(
            f"PLE layer {self.layer_id}: n-gram table {self._table.num_rows} rows x "
            f"{self._table.row_dim} ({self._table._torch_dtype}), scale={self._table.scale}"
        )

    # -- per-request state ---------------------------------------------------------------
    def _ensure_state(self, rows: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._conv_state is None or self._conv_state.shape[0] < rows:
            self._conv_state = torch.zeros(
                rows, self._state_len, self.hc_count * self.hidden_size, dtype=dtype, device=device
            )
            self._history = torch.full(
                (rows, self._ctx_len), self._args.eos_token_id, dtype=torch.long
            )

    # -- forward -------------------------------------------------------------------------
    def _embed(self, ids_cpu: torch.Tensor, reqs, device, dtype) -> torch.Tensor:
        """n-gram embeddings ``[T, ple_embed_dim]`` for the batch; updates the id history."""
        a = self._args
        rows_all = []
        off = 0
        for r in reqs:
            n = r.extend_len
            seg = ids_cpu[off:off + n]
            off += n
            if r.cached_len == 0:
                self._history[r.table_idx].fill_(a.eos_token_id)
            hist = self._history[r.table_idx].clone()
            rows_all.append(ngram_row_ids(hist, seg, self._spec, a.ngram_size,
                                          a.heads_per_ngram, a.eos_token_id))
            full = torch.cat([hist, seg.long()])
            self._history[r.table_idx] = full[-self._ctx_len:]
        rows = torch.cat(rows_all).reshape(-1)  # [T * heads]
        emb = self._table.gather(rows)
        emb = emb.pin_memory() if torch.cuda.is_available() else emb
        emb = emb.to(device, non_blocking=True)
        if self._table.scale is not None or emb.dtype != dtype:
            emb = emb.float()
            if self._table.scale is not None:
                emb = emb * self._table.scale
            emb = emb.to(dtype)
        return emb.view(-1, a.ple_embed_dim)

    def _short_conv(self, x: torch.Tensor, reqs) -> torch.Tensor:
        """Dilated depthwise causal conv + silu per request, continuing each request's state."""
        w = self.conv1d.weight  # [C, 1, K]
        C = w.shape[0]
        out = torch.empty_like(x)
        off = 0
        for r in reqs:
            n = r.extend_len
            seg = x[off:off + n]
            if r.cached_len == 0:
                self._conv_state[r.table_idx].zero_()
            full = torch.cat([self._conv_state[r.table_idx], seg], dim=0)  # [S + n, C]
            y = F.conv1d(full.t().unsqueeze(0), w, None, groups=C, dilation=self._dilation)
            out[off:off + n] = F.silu(y[0].t())
            self._conv_state[r.table_idx].copy_(full[-self._state_len:])
            off += n
        return out

    def forward(self, h: torch.Tensor, ids_cpu: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        assert sum(r.extend_len for r in reqs) == h.shape[0], "PLE: batch/request length mismatch"
        if self._table is None:
            self.load_table()
        self._ensure_state(ctx.page_table.shape[0], h.device, h.dtype)
        H, hc = self.hidden_size, self.hc_count

        emb = self._embed(ids_cpu, reqs, h.device, h.dtype)
        key = self.norm_key.forward(self.key_proj.forward(emb)).unflatten(-1, (hc, H))
        value = self.value_proj.forward(emb)
        query = self.norm_query.forward(h).unflatten(-1, (hc, H))
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_normed = self.norm_conv.forward(gated.flatten(-2))
        out = gated.flatten(-2) + self._short_conv(gated_normed, reqs)
        return h + out


class _Conv1dWeight(BaseOP):
    """Holds the depthwise conv weight ``[C, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, channels: int, kernel: int):
        self.weight = torch.empty(channels, 1, kernel)


__all__ = ["Qwen4ExpPLE", "NGramTable", "ngram_row_ids", "shift_right_ignore_eos"]
