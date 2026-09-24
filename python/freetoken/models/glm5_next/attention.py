"""GLM-5.3 MLA (NoPE) + k-pool DSA sparse attention -- correctness-first PyTorch port.

Ported from ``Glm5NextTextAttention`` / ``Glm5NextTextIndexer`` in
``transformers/models/glm5_next/modeling_glm5_next.py``.

MLA: DeepSeek-V3 low-rank attention with ``qk_rope_head_dim == 0`` (NoPE). We run the
standard weight-absorbed form -- the paged pool stores only the normalized latent
``c_kv`` (``kv_lora_rank``) per token; ``kv_b``'s k-half is absorbed into the query and
its v-half onto the output. Mathematically identical to the reference's expanded K/V.

DSA k-pool indexer (the part that differs from GLM-5.2): per token the indexer produces a
key ``k = LayerNorm(wk(x))`` and compress-gate scores ``gate = x @ compress_gate^T``.
Tokens are grouped into consecutive pools of ``index_kpool`` (absolute positions
``[P*j, P*j + P)``); a COMPLETE pool's compressed key is
``sum_i softmax_i(gate_i + ape_i) * k_i`` (softmax over the pool's tokens, per channel).
Each query scores the complete pools it can see (pool end <= query position):
``sum_h w_h * relu(q_h . pool_key) * D^-0.5`` with ``q = wq_b(q_a_resid)`` and
``w = weights_proj(x) * H^-0.5``, keeps the top ``index_topk / index_kpool`` pools, and
always adds the "tail" (the tokens of its own incomplete pool). Attention then runs over
exactly that token set.

Storage (``DSAKVCache``, rows == page-table positions since page_size == 1):
  * latent slab: ``c_kv`` per token per DSA layer;
  * index slab (full-indexer layers): ``[k | gate | pooled_key]`` per token. The pooled
    key of pool ``j`` is computed once, when its last token (position ``P*j + P - 1``) is
    written, and stored on that token's row -- so scoring a query gathers one row per
    pool instead of re-pooling the whole history every step.

Regimes:
  * prefill (eager, per request): dense causal attention when every visible complete
    pool fits the top-k budget (exactly the reference's selection then), else the sparse
    selection above in query chunks;
  * decode: one static-shape path over the backend-staged per-request row snapshot
    (``DSAMetadata.rows`` / ``kvlen``) -- no host sync, CUDA-graph capturable. Its cost
    scales with the page-table width (the pooled keys of the whole row snapshot are
    gathered each step); see the package docstring for the perf follow-ups.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.utils import nvtx_annotate

from .ops import RMSNorm

# Transient budgets (bytes) for the eager prefill loops.
_PREFILL_SCORE_BYTES = 256 << 20
_PREFILL_ATTN_BYTES = 256 << 20


class _IdxLayerNorm(BaseOP):
    """LayerNorm with bias (the indexer's ``k_norm``, eps 1e-6)."""

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class Glm5NextIndexer(BaseOP):
    """k-pool DSA indexer weights + per-token projections (``full`` layers only)."""

    def __init__(self, args):
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.kpool = args.index_kpool
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = LinearReplicated(args.q_lora_rank, self.n_heads * self.head_dim, has_bias=False)
        self.wk = LinearReplicated(args.hidden_size, self.head_dim, has_bias=False)
        self.k_norm = _IdxLayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = LinearReplicated(args.hidden_size, self.n_heads, has_bias=False)
        self.index_kpool_compress_ape = torch.empty(self.kpool, self.head_dim)
        self.index_kpool_compress_gate = torch.empty(self.head_dim, args.hidden_size)

    def project(self, x: torch.Tensor, q_resid: torch.Tensor):
        """(q [T, Hi, D], weights [T, Hi] fp32, k [T, D], gate [T, D])."""
        t = x.shape[0]
        q = self.wq_b.forward(q_resid).view(t, self.n_heads, self.head_dim)
        k = self.k_norm.forward(self.wk.forward(x))
        gate = F.linear(x, self.index_kpool_compress_gate)
        w = self.weights_proj.forward(x).float() * (self.n_heads**-0.5)
        return q, w, k, gate

    def pool_keys(self, k: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Compressed keys of COMPLETE pools: ``k``/``gate`` [..., kpool, D] (position
        order within the pool) -> [..., D], matching the reference's
        ``(softmax(gate + ape).to(k.dtype) * k).sum(pool_dim)``."""
        logits = gate.float() + self.index_kpool_compress_ape.float()
        probs = logits.softmax(dim=-2).to(k.dtype)
        return (probs * k).sum(dim=-2)

    def scores(self, q: torch.Tensor, w: torch.Tensor, pk: torch.Tensor) -> torch.Tensor:
        """Head-reduced pool scores. ``q`` [..., Hi, D], ``w`` [..., Hi],
        ``pk`` [..., P, D] (broadcastable batch dims) -> [..., P] fp32."""
        s = torch.matmul(q.float(), pk.float().transpose(-1, -2))  # [..., Hi, P]
        s = F.relu(s * self.softmax_scale)
        return torch.matmul(w.unsqueeze(-2), s).squeeze(-2)


class Glm5NextAttention(BaseOP):
    def __init__(self, config, layer_id: int):
        args = config.glm_dsa_args
        self.args = args  # plain attribute (not a tensor/BaseOP): ignored by state_dict
        self.layer_id = layer_id
        self.num_heads = args.num_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.sm_scale = args.qk_head_dim**-0.5
        self.index_topk = args.index_topk
        self.kpool = args.index_kpool
        self.select_tail = args.index_kpool_always_select_tail
        self.index_dim = args.index_head_dim

        # "full": own indexer + index-slab slot; "shared": reuse the previous layer's
        # selection (only valid right after a full DSA layer; validated in args).
        self.shared = args.indexer_types[layer_id] == "shared"
        self.indexer = None if self.shared else Glm5NextIndexer(args)
        idx_layers = args.indexer_layer_ids
        self._idx_slot = idx_layers.index(layer_id) if not self.shared else idx_layers.index(layer_id - 1)

        hidden = args.hidden_size
        self.q_a_proj = LinearReplicated(hidden, args.q_lora_rank, has_bias=False)
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=args.norm_eps)
        self.q_b_proj = LinearReplicated(
            args.q_lora_rank, self.num_heads * args.qk_head_dim, has_bias=False
        )
        self.kv_a_proj_with_mqa = LinearReplicated(hidden, args.latent_dim, has_bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=args.norm_eps)
        # Consumed as bmm operands (absorption), never through a Linear forward.
        self.kv_b_proj = LinearReplicated(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            has_bias=False,
        )
        self.o_proj = LinearReplicated(self.num_heads * self.v_head_dim, hidden, has_bias=False)
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    # -- weights ------------------------------------------------------------------------
    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """W_uk [H, nope, lora] and W_uv^T [H, lora, v] (cached split of kv_b)."""
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(
                self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )
            self._w_uk = w[:, : self.qk_nope_head_dim, :].contiguous()
            self._w_uv = w[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None  # repacked forms serve from here on

    # -- index slab helpers -------------------------------------------------------------
    def _slab(self) -> torch.Tensor:
        return get_global_ctx().kv_cache.index_k_cache(self._idx_slot)

    def _store_index(self, k, gate, out_loc) -> None:
        slab = self._slab()
        d = self.index_dim
        slab[out_loc, :d] = k.to(slab.dtype)
        slab[out_loc, d : 2 * d] = gate.to(slab.dtype)

    def _store_pool_keys(self, pos: torch.Tensor, row_of, out_loc: torch.Tensor) -> None:
        """Write the pooled key for every new token (meaningful only where
        ``pos % kpool == kpool - 1``; other rows' pooled segment is never read).
        ``row_of(positions [N, P]) -> physical rows``."""
        slab = self._slab()
        d, P = self.index_dim, self.kpool
        member = pos.unsqueeze(-1) - (P - 1) + torch.arange(P, device=pos.device)  # [N, P]
        rows = row_of(member.clamp_min(0))
        st = slab[rows.long()]  # [N, P, 3D]
        pk = self.indexer.pool_keys(st[..., :d], st[..., d : 2 * d])
        slab[out_loc, 2 * d :] = pk.to(slab.dtype)

    def _pooled_keys_for(self, rows_pool_end: torch.Tensor) -> torch.Tensor:
        d = self.index_dim
        return self._slab()[rows_pool_end.long(), 2 * d :]

    # -- selection ----------------------------------------------------------------------
    def _select(
        self, scores: torch.Tensor, pos: torch.Tensor, num_pools: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token positions each query attends: ``scores`` [N, num_pools] over pools
        ``0..num_pools-1``, ``pos`` [N] query positions. Returns (positions [N, W] int64,
        valid [N, W] bool) with W = K*kpool (+ kpool-1 tail)."""
        P = self.kpool
        pool_end = torch.arange(num_pools, device=pos.device) * P + (P - 1)
        visible = pool_end.unsqueeze(0) <= pos.unsqueeze(1)  # [N, num_pools]
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        k_sel = min(self.index_topk // P, num_pools)
        sel = scores.topk(k_sel, dim=-1).indices  # [N, k_sel]
        sel_valid = visible.gather(-1, sel)
        tok = sel.unsqueeze(-1) * P + torch.arange(P, device=pos.device)  # [N, k_sel, P]
        tok_valid = sel_valid.unsqueeze(-1).expand_as(tok)
        tok, tok_valid = tok.flatten(-2), tok_valid.flatten(-2)
        if self.select_tail and P > 1:
            tail_count = (pos + 1) % P
            offs = torch.arange(P - 1, device=pos.device)
            tail = (pos + 1 - tail_count).unsqueeze(-1) + offs
            tail_valid = offs.unsqueeze(0) < tail_count.unsqueeze(-1)
            tok = torch.cat([tok, tail], dim=-1)
            tok_valid = torch.cat([tok_valid, tail_valid], dim=-1)
        return tok, tok_valid

    # -- attention core -----------------------------------------------------------------
    def _attend(self, q_abs, lat, valid) -> torch.Tensor:
        """``q_abs`` [N, H, L], ``lat`` [N, W, L] (per-query gathered latents) or
        [W, L] (shared), ``valid`` [N, W] bool. Returns o_latent [N, H, L] fp32."""
        lat_f = lat.float()
        if lat.dim() == 3:
            # Invalid slots gather rows that were never written (the pool is torch.empty):
            # a NaN there would survive p == 0 in the value product, so zero them.
            lat_f = lat_f.masked_fill(~valid.unsqueeze(-1), 0.0)
        if lat.dim() == 2:
            s = torch.einsum("nhl,wl->nhw", q_abs.float(), lat_f)
        else:
            s = torch.einsum("nhl,nwl->nhw", q_abs.float(), lat_f)
        s = (s * self.sm_scale).masked_fill(~valid.unsqueeze(1), float("-inf"))
        p = s.softmax(dim=-1)
        if lat.dim() == 2:
            return torch.einsum("nhw,wl->nhl", p, lat_f)
        return torch.einsum("nhw,nwl->nhl", p, lat_f)

    # -- forward ------------------------------------------------------------------------
    @nvtx_annotate("MLA-DSA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        kv = ctx.kv_cache
        t = x.shape[0]
        w_uk, w_uv = self._kv_b()

        q_resid = self.q_a_layernorm.forward(self.q_a_proj.forward(x))
        q = self.q_b_proj.forward(q_resid).view(t, self.num_heads, self.qk_nope_head_dim)
        c_kv = self.kv_a_layernorm.forward(self.kv_a_proj_with_mqa.forward(x))
        # absorb kv_b's k-half into the query: [H, T, nope] @ [H, nope, L] -> [T, H, L]
        q_abs = torch.bmm(q.transpose(0, 1), w_uk).transpose(0, 1)

        kv.store_kv(c_kv, c_kv.new_empty(t, 0), batch.out_loc, self.layer_id)
        proj = None
        if self.indexer is not None:
            q_i, w_i, k_i, gate_i = self.indexer.project(x, q_resid)
            self._store_index(k_i, gate_i, batch.out_loc)
            proj = (q_i, w_i)

        md = batch.attn_metadata
        if batch.is_decode:
            o_lat = self._decode(md, batch, q_abs, proj)
        else:
            o_lat = self._prefill(md, batch, q_abs, proj)

        # absorb kv_b's v-half onto the output: [H, T, L] @ [H, L, v] -> [T, H, v]
        o = torch.bmm(o_lat.to(x.dtype).transpose(0, 1), w_uv).transpose(0, 1)
        return self.o_proj.forward(o.reshape(t, self.num_heads * self.v_head_dim))

    # -- decode (static shapes, graph-capturable) ---------------------------------------
    def _decode(self, md, batch, q_abs, proj) -> torch.Tensor:
        ctx = get_global_ctx()
        if md.rows is None:
            # Eager decode: snapshot this step's page-table rows once (first DSA layer),
            # like the dsa backend's own mla_forward does.
            md.rows = ctx.page_table.index_select(
                0, batch.active_table_idx.to(torch.int64)
            ).to(torch.int32)
            md.kvlen = md.kv_len_cpu.to(q_abs.device, non_blocking=True)
        rows = md.rows.long()  # [B, W]
        bs, width = rows.shape
        pos = (md.kvlen.long() - 1).clamp_min(0)  # [B] query positions
        P = self.kpool

        if proj is not None:
            q_i, w_i = proj
            self._store_pool_keys(pos, lambda m: rows.gather(1, m), batch.out_loc)
            num_pools = width // P
            if num_pools > 0:
                pk = self._pooled_keys_for(rows[:, P - 1 : num_pools * P : P])  # [B, np, D]
                scores = self.indexer.scores(q_i, w_i, pk)  # [B, np]
            else:
                scores = q_abs.new_zeros(bs, 0, dtype=torch.float32)
            tok, valid = self._select(scores, pos, num_pools)
            md.glm5_sel = (tok, valid)
        tok, valid = md.glm5_sel
        valid = valid & (tok <= pos.unsqueeze(-1))
        phys = rows.gather(1, tok.clamp(0, width - 1))
        lat = get_global_ctx().kv_cache.latent_rows(self.layer_id)[phys]  # [B, W', L]
        return self._attend(q_abs, lat, valid)

    # -- prefill (eager, per request) ---------------------------------------------------
    def _prefill(self, md, batch, q_abs, proj) -> torch.Tensor:
        ctx = get_global_ctx()
        page_table = ctx.page_table
        latent = ctx.kv_cache.latent_rows(self.layer_id)
        qo = md.qo_indptr_cpu.tolist()
        reqs = batch.padded_reqs
        H, L = self.num_heads, self.kv_lora_rank
        P = self.kpool
        out = torch.empty(q_abs.shape[0], H, L, dtype=torch.float32, device=q_abs.device)
        sels = []
        for i, r in enumerate(reqs):
            s0, s1 = qo[i], qo[i + 1]
            m = s1 - s0
            if m == 0:
                sels.append(None)
                continue
            kv_len = r.device_len
            rows = page_table[r.table_idx, :kv_len].long()
            pos = batch.positions[s0:s1].long()
            if proj is not None:
                self._store_pool_keys(pos, lambda mm: rows[mm], batch.out_loc[s0:s1])
            # Dense is exact while every visible complete pool fits the top-k budget (and
            # the incomplete tail pool is attended too).
            dense = (kv_len // P) <= (self.index_topk // P) and (self.select_tail or P == 1)
            if dense:
                sels.append(None)
                lat = latent[rows]  # [kv, L]
                chunk = max(1, _PREFILL_ATTN_BYTES // max(H * kv_len * 4 * 2, 1))
                cols = torch.arange(kv_len, device=pos.device)
                for c0 in range(0, m, chunk):
                    c1 = min(m, c0 + chunk)
                    valid = cols.unsqueeze(0) <= pos[c0:c1].unsqueeze(1)
                    out[s0 + c0 : s0 + c1] = self._attend(q_abs[s0 + c0 : s0 + c1], lat, valid)
                continue
            if proj is not None:
                q_i, w_i = proj
                num_pools = kv_len // P
                pk = self._pooled_keys_for(rows[P - 1 : num_pools * P : P])  # [np, D]
                chunk = max(1, _PREFILL_SCORE_BYTES // max(self.indexer.n_heads * num_pools * 4 * 2, 1))
                toks, valids = [], []
                for c0 in range(0, m, chunk):
                    c1 = min(m, c0 + chunk)
                    sc = self.indexer.scores(q_i[s0 + c0 : s0 + c1], w_i[s0 + c0 : s0 + c1], pk)
                    tk, vd = self._select(sc, pos[c0:c1], num_pools)
                    toks.append(tk)
                    valids.append(vd)
                sel = (torch.cat(toks), torch.cat(valids))
            else:
                prev = md.glm5_prefill_sel[i]
                assert prev is not None, "shared DSA layer without a leader selection"
                sel = prev
            sels.append(sel)
            tok, valid = sel
            width = tok.shape[-1]
            chunk = max(1, _PREFILL_ATTN_BYTES // max(width * L * 4 * 2, 1))
            for c0 in range(0, m, chunk):
                c1 = min(m, c0 + chunk)
                tk = tok[c0:c1]
                vd = valid[c0:c1] & (tk <= pos[c0:c1].unsqueeze(-1))
                lat = latent[rows[tk.clamp(0, kv_len - 1)]]  # [c, W, L]
                out[s0 + c0 : s0 + c1] = self._attend(q_abs[s0 + c0 : s0 + c1], lat, vd)
        if proj is not None:
            md.glm5_prefill_sel = sels
        return out


__all__ = ["Glm5NextAttention", "Glm5NextIndexer"]
