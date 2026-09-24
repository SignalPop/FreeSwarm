"""Small GLM-5.3 building blocks: HF-exact RMSNorm, manifold-constrained
Hyper-Connections, and the clamped-SwiGLU MLP (dense layers + shared experts).

Plain PyTorch on purpose (correctness first): each mirrors the reference module's op
order and precision (``Glm5NextTextRMSNorm``, ``Glm5NextTextHyperConnection``,
``Glm5NextTextMLP``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.utils import nvtx_annotate


class RMSNorm(BaseOP):
    """``weight * (x.float() * rsqrt(mean(x^2) + eps)).to(x.dtype)`` (HF op order)."""

    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dtype)


class HyperConnection(BaseOP):
    """mHC mapping for one sublayer site (``attn_hc`` / ``ffn_hc``).

    ``streams`` [T, hc, D] -> (``post`` [T, hc] fp32, ``comb`` [T, hc, hc] fp32,
    ``collapsed`` [T, D] in the stream dtype). ``fn``/``base``/``scale`` are bf16 in the
    checkpoint and held fp32 here (lossless; the reference upcasts them at use).
    """

    def __init__(self, hc_mult: int, hidden_size: int, eps: float, sinkhorn_iters: int,
                 norm_eps: float):
        self.hc = hc_mult
        self.eps = eps
        self.iters = sinkhorn_iters
        self.norm_eps = norm_eps
        mix = (2 + hc_mult) * hc_mult
        self.fn = torch.empty(mix, hc_mult * hidden_size, dtype=torch.float32)
        self.base = torch.empty(mix, dtype=torch.float32)
        self.scale = torch.empty(3, dtype=torch.float32)

    def forward(self, streams: torch.Tensor):
        hc = self.hc
        t = streams.shape[0]
        flat = streams.reshape(t, -1).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        pre_w, post_w, comb_w = F.linear(flat, self.fn).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_s, post_s, comb_s = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_s + pre_b) + self.eps
        post = 2 * torch.sigmoid(post_w * post_s + post_b)
        comb = comb_w.view(t, hc, hc) * comb_s + comb_b.view(hc, hc)
        comb = torch.softmax(comb, dim=-1) + self.eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        collapsed = (pre.unsqueeze(-1) * streams).sum(dim=1).to(streams.dtype)
        return post, comb, collapsed

    @staticmethod
    def expand(out: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
               comb: torch.Tensor) -> torch.Tensor:
        """New streams: ``post[h] * out + sum_j comb[j, h] * residual[j]`` (bf16, HF order)."""
        dtype = residual.dtype
        return post.to(dtype).unsqueeze(-1) * out.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )


class ClampedSwiGLUMLP(BaseOP):
    """``down(silu(min(gate, L)) * clamp(up, +-L))`` with fused gate|up (bf16)."""

    def __init__(self, hidden_size: int, intermediate_size: int, limit: float):
        self.limit = limit
        self.inter = intermediate_size
        self.gate_up_proj = LinearReplicated(hidden_size, 2 * intermediate_size, has_bias=False)
        self.down_proj = LinearReplicated(intermediate_size, hidden_size, has_bias=False)

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj.forward(x).split(self.inter, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.down_proj.forward(F.silu(gate) * up)


__all__ = ["RMSNorm", "HyperConnection", "ClampedSwiGLUMLP"]
