"""Qwen4-Exp QSA (Qwen Sparse Attention) layer -- port of HF ``Qwen4ExpTextAttention`` +
``Qwen4ExpTextQSAIndexer``.

Gated GQA exactly like Qwen3.5 (per-head (1+w) q/k norms, partial NeoX rope, sigmoid output
gate), except each query attends only a subset of its causal context chosen by an indexer:

* the indexer projects ``indexer_n_heads`` query heads and ONE raw key per token;
* the visible keys are split into complete ``compress_ratio``-token blocks, each block's raw
  keys are mean-pooled, (1+w)-normed and roped at the block's first position;
* ``score(q, b) = sum_h relu(q_h . k_b) / sqrt(d)``; the top ``budget / ratio`` blocks are
  selected, plus the trailing partial block (the last ``(pos+1) % ratio`` tokens).

When a request's context holds at most ``budget / ratio`` complete blocks every visible token
is selected, i.e. QSA == dense causal attention. That (the common short-context case) runs
through the engine's regular attention backend; longer contexts take a plain PyTorch
gather-attend path (correctness first -- not optimized).

The raw indexer keys of every token are kept in the KV pool's side slab
(``kvcache/qsa_pool.py``), addressed by the same token rows as K/V.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

from .layers import HFPartialRope, QRMSNorm

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Upper bound on the fp32 transients of the sparse path per query chunk (gathered K/V).
_SPARSE_CHUNK_BYTES = 512 << 20
# Debug/verification knob: route every QSA forward through the sparse gather path even when
# the dense backend fast path would be exact (lets tests pin the sparse path's engine
# integration against a dense-equivalent reference).
_FORCE_SPARSE = os.getenv("FREETOKEN_QWEN4_QSA_FORCE_SPARSE", "0").strip().lower() in (
    "1", "true", "yes", "on"
)


def qsa_block_keys(raw_keys: torch.Tensor, ratio: int, k_layernorm: "QRMSNorm",
                   rope: HFPartialRope) -> torch.Tensor:
    """Block keys of the complete ``ratio``-token blocks of ``raw_keys`` [L, Di] (bf16, the
    request's raw indexer keys in token order): fp32 mean -> activation dtype -> (1+w) norm ->
    rope at each block's first position. Returns fp32 ``[L // ratio, Di]`` (HF op order)."""
    nb = raw_keys.shape[0] // ratio
    if nb == 0:
        return raw_keys.new_zeros((0, raw_keys.shape[-1]), dtype=torch.float32)
    pooled = raw_keys[: nb * ratio].view(nb, ratio, -1).float().mean(dim=1).to(raw_keys.dtype)
    pooled = k_layernorm.forward(pooled)
    starts = torch.arange(nb, device=raw_keys.device) * ratio
    cos, sin = rope.cos_sin(starts, pooled.dtype)
    return rope.apply(pooled, cos, sin).float()


def qsa_select(index_q: torch.Tensor, block_k: torch.Tensor, pos: torch.Tensor, ratio: int,
               block_topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Token selection of HF ``Qwen4ExpTextQSAIndexer`` for queries at absolute positions
    ``pos`` [m] of one request (causal, no padding): the top-``block_topk`` complete visible
    blocks by ``sum_h relu(q_h . k_b) / sqrt(d)`` plus the trailing partial block.

    Returns ``(idx [m, S], ok [m, S])``: token indices (0 where not ``ok``) with
    ``S = min(block_topk, nb) * ratio + ratio - 1``."""
    dev = index_q.device
    m = index_q.shape[0]
    nb = block_k.shape[0]
    ncomp = (pos + 1) // ratio  # complete visible blocks per query
    ar = torch.arange(ratio, device=dev)
    k_top = min(block_topk, nb)
    idx_parts, ok_parts = [], []
    if k_top > 0:
        sc = torch.einsum("qhd,bd->qhb", index_q.float(), block_k)
        sc = torch.relu(sc).sum(dim=1) / math.sqrt(block_k.shape[-1])  # [m, nb]
        valid = torch.arange(nb, device=dev)[None, :] < ncomp[:, None]
        sc = sc.masked_fill(~valid, float("-inf"))
        top = sc.topk(k_top, dim=-1)
        top_ok = torch.isfinite(top.values)  # [m, k_top]
        idx_parts.append((top.indices[..., None] * ratio + ar).reshape(m, -1))
        ok_parts.append(top_ok[..., None].expand(m, k_top, ratio).reshape(m, -1))
    tail = ncomp[:, None] * ratio + ar[: ratio - 1][None, :]  # [m, ratio-1]
    idx_parts.append(tail)
    ok_parts.append(tail <= pos[:, None])
    idx = torch.cat(idx_parts, dim=1)
    ok = torch.cat(ok_parts, dim=1)
    return torch.where(ok, idx, torch.zeros_like(idx)), ok


class QSAIndexer(BaseOP):
    def __init__(self, config: "ModelConfig"):
        a = config.qwen4_args
        self.n_heads = a.indexer_n_heads
        self.head_dim = a.indexer_head_dim
        self.index_qk_proj = LinearReplicated(
            config.hidden_size, (self.n_heads + 1) * self.head_dim, has_bias=False
        )
        self.q_layernorm = QRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = QRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope: HFPartialRope):
        """-> (index_q [T, n_heads, d] normed+roped, raw index_k [T, d])."""
        qk = self.index_qk_proj.forward(x)
        q, k = torch.split(qk, [self.n_heads * self.head_dim, self.head_dim], dim=-1)
        q = self.q_layernorm.forward(q.reshape(-1, self.n_heads, self.head_dim))
        q = rope.apply(q, cos, sin)
        return q, k.contiguous()


class Qwen4ExpAttention(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        a = config.qwen4_args
        self.layer_id = layer_id
        self.num_q = a.num_heads
        self.num_kv = a.num_kv_heads
        self.head_dim = hd = a.head_dim
        self.qo_dim = self.num_q * hd
        self.kv_dim = self.num_kv * hd
        self._split = [self.num_q * hd * 2, self.kv_dim, self.kv_dim]
        # fused q(+gate) | k | v (bf16 in both released checkpoints)
        self.qkv_proj = LinearColParallelMerged(config.hidden_size, self._split, has_bias=False)
        self.q_norm = QRMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm = QRMSNorm(hd, eps=config.rms_norm_eps)
        self.o_proj = LinearReplicated(self.qo_dim, config.hidden_size, has_bias=False)
        self.rotary = get_rope(
            head_dim=hd, rotary_dim=a.rotary_dim, max_position=a.max_position, base=a.rope_theta,
        )
        self.indexer = QSAIndexer(config)
        self._index_rope = HFPartialRope(a.rotary_dim, a.rope_theta)
        self._ratio = a.indexer_compress_ratio
        self._block_topk = a.block_topk
        self._dense_limit = a.dense_kv_limit
        self._scale = hd ** -0.5

    # ------------------------------------------------------------------------------------
    def _sparse_request(self, q, index_q, rows, cached_len: int, k_rows, v_rows, ik_rows):
        """Sparse attention for one request. ``q`` [n, Hq, D], ``index_q`` [n, Hi, Di],
        ``rows`` [L] int64 KV rows of the request's tokens 0..L-1. Returns [n, Hq, D]."""
        R = self._ratio
        L = rows.numel()
        n = q.shape[0]
        dev = q.device
        nb = L // R
        pos = torch.arange(cached_len, cached_len + n, device=dev)
        block_k = qsa_block_keys(ik_rows[rows[: nb * R]], R, self.indexer.k_layernorm,
                                 self._index_rope)  # [nb, Di] fp32
        kvh = self.num_kv
        g = self.num_q // kvh
        D = self.head_dim
        S = min(self._block_topk, nb) * R + R - 1
        per_q = S * kvh * D * 4 * 2 + max(nb, 1) * index_q.shape[1] * 4
        chunk = max(1, min(n, _SPARSE_CHUNK_BYTES // per_q))
        out = torch.empty_like(q)
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            m = s1 - s0
            idx, ok = qsa_select(index_q[s0:s1], block_k, pos[s0:s1], R, self._block_topk)
            sel = rows[idx]  # [m, S] pool rows of the selected tokens
            Ks = k_rows[sel]  # [m, S, kvh, D]
            Vs = v_rows[sel]
            qm = q[s0:s1].view(m, kvh, g, D).float()
            att = torch.einsum("qkgd,qskd->qkgs", qm, Ks.float()) * self._scale
            att = att.masked_fill(~ok[:, None, None, :], float("-inf"))
            att = torch.softmax(att, dim=-1)
            o = torch.einsum("qkgs,qskd->qkgd", att, Vs.float())
            out[s0:s1] = o.reshape(m, self.num_q, D).to(out.dtype)
        return out

    def _sparse(self, q, k, v, index_q, batch) -> torch.Tensor:
        ctx = get_global_ctx()
        pool = ctx.kv_cache
        pool.store_kv(k, v, batch.out_loc, self.layer_id)
        kc = pool.k_cache(self.layer_id)
        vc = pool.v_cache(self.layer_id)
        k_rows = kc.view(-1, kc.shape[-2], kc.shape[-1])
        v_rows = vc.view(-1, vc.shape[-2], vc.shape[-1])
        ik_rows = pool.index_k_rows(self.layer_id)
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        out = torch.empty_like(q)
        off = 0
        for r in reqs:
            n = r.extend_len
            rows = ctx.page_table[r.table_idx, : r.device_len].long()
            out[off:off + n] = self._sparse_request(
                q[off:off + n], index_q[off:off + n], rows, r.cached_len, k_rows, v_rows, ik_rows
            )
            off += n
        return out

    @nvtx_annotate("QSA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        positions = batch.positions
        T = x.shape[0]
        hd = self.head_dim

        qkv = self.qkv_proj.forward(x)
        qg, k, v = torch.split(qkv, self._split, dim=-1)
        qg = qg.view(T, self.num_q, 2 * hd)
        q = qg[..., :hd]
        gate = qg[..., hd:].reshape(T, self.qo_dim)
        q = self.q_norm.forward(q).reshape(T, self.qo_dim).contiguous()
        k = self.k_norm.forward(k.view(T, self.num_kv, hd)).reshape(T, self.kv_dim).contiguous()
        v = v.contiguous()
        q, k = self.rotary.forward(positions, q, k)
        q = q.view(T, self.num_q, hd)

        cos, sin = self._index_rope.cos_sin(positions, x.dtype)
        index_q, index_k = self.indexer.forward(x, cos, sin, self._index_rope)
        ctx.kv_cache.store_index_k(index_k, batch.out_loc, self.layer_id)

        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        if not _FORCE_SPARSE and all(r.device_len <= self._dense_limit for r in reqs):
            # every visible token is selected: exact dense causal attention (backend fast path)
            o = ctx.attn_backend.forward(q, k, v, self.layer_id, batch)
        else:
            o = self._sparse(q, k, v, index_q, batch)
        o = o.reshape(T, self.qo_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(o)


__all__ = ["Qwen4ExpAttention", "QSAIndexer"]
