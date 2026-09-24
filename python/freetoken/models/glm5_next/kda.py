"""Kimi Delta Attention (KDA) for GLM-5.3-Flash -- correctness-first PyTorch port.

Ported from ``Glm5NextTextLinearAttention`` / ``chunk_kimi_delta_attention`` /
``recurrent_kimi_delta_attention`` in ``transformers/models/glm5_next/modeling_glm5_next.py``.
KDA is a gated delta rule whose decay is per KEY CHANNEL (GDN's is per head)::

    S_t = diag(exp(g_t)) S_{t-1}                      # g_t: [H, K], per-channel log decay
    S_t = S_t + k_t (beta_t * (v_t - S_t^T k_t))^T    # delta-rule write
    o_t = S_t^T (q_t / sqrt(K))                       # q, k l2-normalized (fp32)

State ``S`` is ``[H, K, V]`` fp32 per request, held in the engine's ``LinearStatePool``
(``recurrent_states[layer, slot]``; KDA's q|k|v short conv has the GDN conv geometry,
``conv_states[layer, slot]``). Prefill runs the reference's WY-style chunk algorithm
(intra-chunk triangular solve + inter-chunk recurrence) per request in fp32; decode runs
the single-step recurrence batched over requests with device-side slot indices (no host
sync -> CUDA-graph capturable).

Performance: this is a plain PyTorch path (no fused FLA KDA kernel is vendored yet); it is
exact but prefill-slow for long prompts. See the package docstring for follow-ups.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.utils import nvtx_annotate

# Chunk length of the prefill algorithm (the reference's default). Math is chunk-size
# independent; 64 keeps the op order closest to the reference.
KDA_CHUNK = 64
# Transient budget for the intra-chunk [H, n, C, C, K] fp32 decay tensors.
_INTRA_BYTES = 512 << 20


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # FLA/HF convention: x / sqrt(sum(x^2) + eps) (not F.normalize's max(norm, eps)).
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def kda_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int = KDA_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked KDA over ONE sequence.

    ``q``/``k``/``v`` [T, H, D] (any float dtype), ``g`` [T, H, K] fp32 log-decay,
    ``beta`` [T, H], ``initial_state`` [H, K, V] fp32 or None. Returns
    (out [T, H, V] in q's dtype, final_state [H, K, V] fp32).
    """
    out_dtype = q.dtype
    T, H, K = k.shape
    V = v.shape[-1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = _l2norm(q) * (K**-0.5)
    k = _l2norm(k)

    C = chunk_size
    pad = (C - T % C) % C
    n = (T + pad) // C
    # [H, n, C, D]
    def chunked(x):
        x = F.pad(x, (0, 0, 0, 0, 0, pad)) if x.dim() == 3 else F.pad(x, (0, 0, 0, pad))
        return x.transpose(0, 1).reshape(H, n, C, *x.shape[2:])

    q, k, v, g = chunked(q), chunked(k), chunked(v), chunked(g)
    beta = chunked(beta)  # [H, n, C]
    k_beta = k * beta.unsqueeze(-1)
    v_beta = v * beta.unsqueeze(-1)
    g = g.cumsum(dim=-2)  # within-chunk cumulative log decay

    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))  # j <= i
    strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), -1)  # j < i
    eye = torch.eye(C, dtype=torch.float32, device=q.device)

    # Intra-chunk terms, in groups of chunks to bound the [.., C, C, K] transient:
    #   M[i, j]   = sum_d kb_i k_j exp(g_i - g_j)   (j < i)  -> T = (I + M)^-1
    #   QK[i, j]  = sum_d q_i  k_j exp(g_i - g_j)   (j <= i)
    #   u = T @ v_beta ; w = T @ (k_beta * exp(g))
    per_chunk = H * C * C * K * 4 * 3
    group = max(1, _INTRA_BYTES // max(per_chunk, 1))
    u = torch.empty_like(v)
    w = torch.empty_like(k)
    qk = torch.empty(H, n, C, C, dtype=torch.float32, device=q.device)
    for s in range(0, n, group):
        e = min(n, s + group)
        gs = g[:, s:e]
        decay = (gs.unsqueeze(-2) - gs.unsqueeze(-3))  # [H, m, i, j, K] = g_i - g_j
        decay = decay.masked_fill(~lower[:, :, None], float("-inf")).exp()
        m_ij = (k_beta[:, s:e].unsqueeze(-2) * k[:, s:e].unsqueeze(-3) * decay).sum(-1)
        m_ij = m_ij.masked_fill(~strict, 0.0)
        qk[:, s:e] = (q[:, s:e].unsqueeze(-2) * k[:, s:e].unsqueeze(-3) * decay).sum(-1)
        del decay
        tri = m_ij + eye  # unit lower triangular
        rhs = torch.cat([v_beta[:, s:e], k_beta[:, s:e] * gs.exp()], dim=-1)
        sol = torch.linalg.solve_triangular(tri, rhs, upper=False, unitriangular=True)
        u[:, s:e] = sol[..., :V]
        w[:, s:e] = sol[..., V:]
    qk = qk.masked_fill(~lower, 0.0)

    S = (
        torch.zeros(H, K, V, dtype=torch.float32, device=q.device)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty(H, n, C, V, dtype=torch.float32, device=q.device)
    for c in range(n):
        q_c, k_c, g_c = q[:, c], k[:, c], g[:, c]
        v_new = u[:, c] - w[:, c] @ S  # [H, C, V]
        out[:, c] = (q_c * g_c.exp()) @ S + qk[:, c] @ v_new
        g_last = g_c[:, -1]  # [H, K]
        S = S * g_last.exp().unsqueeze(-1) + (k_c * (g_last.unsqueeze(1) - g_c).exp()).transpose(
            -1, -2
        ) @ v_new
    out = out.reshape(H, n * C, V)[:, :T].transpose(0, 1).contiguous()
    return out.to(out_dtype), S


def kda_recurrent_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One decode step for a batch: ``q``/``k``/``v`` [B, H, D], ``g`` [B, H, K] fp32,
    ``beta`` [B, H], ``state`` [B, H, K, V] fp32. Returns (out [B, H, V] in q's dtype,
    new state fp32)."""
    out_dtype = q.dtype
    K = k.shape[-1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = _l2norm(q) * (K**-0.5)
    k = _l2norm(k)
    S = state.float() * g.exp().unsqueeze(-1)
    kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)  # [B, H, V]
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (S * q.unsqueeze(-1)).sum(dim=-2)
    return out.to(out_dtype), S


class _Param(BaseOP):
    """Holds one named tensor (``<name>.weight``)."""

    def __init__(self, *shape: int, dtype: torch.dtype | None = None):
        self.weight = torch.empty(*shape, dtype=dtype) if dtype else torch.empty(*shape)


class Glm5NextKDA(BaseOP):
    """GLM-5.3 KDA mixer. Parameter names follow the HF module after the checkpoint's
    renames, with q/k/v fused (``qkv_proj``) and the three per-projection short convs
    concatenated (``conv1d``, fp32 like the checkpoint)::

        qkv_proj, conv1d, f_a_proj, f_b_proj, dt_bias, A_log, b_proj,
        g_a_proj, g_b_proj, o_norm, o_proj
    """

    def __init__(self, config, layer_id: int):
        args = config.glm_dsa_args
        self.layer_id = layer_id
        self.num_heads = args.linear_num_heads
        self.head_dim = args.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_kernel = args.linear_conv_kernel_dim
        self.lower_bound = args.linear_lower_bound
        self.norm_eps = args.norm_eps
        hidden = args.hidden_size

        self.qkv_proj = LinearReplicated(hidden, 3 * self.qkv_dim, has_bias=False)
        self.conv1d = _Param(3 * self.qkv_dim, 1, self.conv_kernel, dtype=torch.float32)
        self.f_a_proj = LinearReplicated(hidden, self.head_dim, has_bias=False)
        self.f_b_proj = LinearReplicated(self.head_dim, self.qkv_dim, has_bias=False)
        self.dt_bias = torch.empty(self.qkv_dim, dtype=torch.float32)
        self.A_log = torch.empty(self.num_heads, dtype=torch.float32)
        self.b_proj = LinearReplicated(hidden, self.num_heads, has_bias=False)
        self.g_a_proj = LinearReplicated(hidden, self.head_dim, has_bias=False)
        self.g_b_proj = LinearReplicated(self.head_dim, self.qkv_dim, has_bias=False)
        self.o_norm = _Param(self.head_dim)
        self.o_proj = LinearReplicated(self.qkv_dim, hidden, has_bias=False)

    # -- gates --------------------------------------------------------------------------
    def _forget_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel log decay g [T, H, K] fp32 (``Glm5NextTextForgetGate``)."""
        t = x.shape[0]
        f = self.f_b_proj.forward(self.f_a_proj.forward(x))
        g = (f.float() + self.dt_bias.float()).view(t, self.num_heads, self.head_dim)
        decay_rate = self.A_log.float().exp().view(1, self.num_heads, 1)
        if self.lower_bound is not None:
            return self.lower_bound * torch.sigmoid(decay_rate * g)
        g_softplus = torch.where(g > 20.0, g, torch.log1p(torch.exp(g)))
        return -decay_rate * g_softplus

    def _gated_norm(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """fp32 RMSNorm (weight in fp32) * sigmoid(gate), cast back (``RMSNormGated``)."""
        dtype = core.dtype
        h = core.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        h = self.o_norm.weight.float() * h
        return (h * torch.sigmoid(gate.float())).to(dtype)

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, K]

    # -- forward ------------------------------------------------------------------------
    @nvtx_annotate("KDA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = x.shape[0]
        H, D = self.num_heads, self.head_dim

        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, x.device)
            batch.fla_metadata = fla
        assert fla.track_dst is None, (
            "glm5_next KDA does not implement hybrid-radix state snapshots "
            "(the engine forces --cache-type naive for this model)"
        )
        li = pool.local_index(self.layer_id)
        conv_w = self._conv_weight()

        qkv = self.qkv_proj.forward(x)  # [T, 3 * qkv_dim]
        g = self._forget_gate(x)  # [T, H, K] fp32
        beta = torch.sigmoid(self.b_proj.forward(x))  # [T, H] (model dtype, like HF)

        rec = pool.recurrent_states[li]
        if batch.is_decode:
            mixed = causal_conv1d_decode(qkv, pool.conv_states[li], conv_w, fla.cache_indices)
            q, k, v = mixed.split(self.qkv_dim, dim=-1)
            slots = fla.cache_indices.long()
            state = rec.index_select(0, slots)
            core, new_state = kda_recurrent_step(
                q.reshape(total, H, D), k.reshape(total, H, D), v.reshape(total, H, D),
                g, beta, state,
            )
            rec.index_copy_(0, slots, new_state.to(rec.dtype))
        else:
            conv_in = qkv.transpose(0, 1).contiguous()  # [conv_dim, T]
            mixed = causal_conv1d_varlen(
                conv_in, conv_w, pool.conv_states[li], fla.cu_seqlens, fla.cache_indices,
                fla.has_initial_state,
            ).transpose(0, 1)
            q, k, v = mixed.split(self.qkv_dim, dim=-1)
            q, k, v = (t.reshape(total, H, D) for t in (q, k, v))
            core = torch.empty_like(v)
            reqs = batch.padded_reqs
            offset = 0
            for r in reqs:
                n = r.extend_len
                if n == 0:
                    continue
                slot = r.linear_slot_idx if r.linear_slot_idx is not None else r.table_idx
                init = rec[slot] if r.cached_len > 0 else None
                sl = slice(offset, offset + n)
                core[sl], final = kda_chunk_prefill(q[sl], k[sl], v[sl], g[sl], beta[sl], init)
                rec[slot].copy_(final.to(rec.dtype))
                offset += n
            assert offset == total, (offset, total)

        gate = self.g_b_proj.forward(self.g_a_proj.forward(x)).view(total, H, D)
        out = self._gated_norm(core.view(total, H, D), gate).reshape(total, self.qkv_dim)
        return self.o_proj.forward(out)


__all__ = ["Glm5NextKDA", "kda_chunk_prefill", "kda_recurrent_step"]
