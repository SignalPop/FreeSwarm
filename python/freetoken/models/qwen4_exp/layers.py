"""Small Qwen4-Exp building blocks: the (1+w) RMSNorm (optionally grouped), the gated
residual hyper-connection, and an HF-exact partial NeoX rope for the QSA indexer.

These are plain PyTorch ports of ``modeling_qwen4_exp`` (correctness first); numerics follow
the reference op-for-op (fp32 norm statistics, bf16 elementwise math elsewhere).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated


class QRMSNorm(BaseOP):
    """HF ``Qwen4ExpTextRMSNorm``: ``x * rsqrt(mean(x^2) + eps) * (1 + w)`` in fp32, cast back
    to the input dtype. ``group_size`` normalizes each ``group_size`` chunk of the last dim
    independently (hyper-connection / PLE norms over the ``hc_count * hidden`` stream). The raw
    checkpoint weight is stored as-is (no +1 baked in)."""

    def __init__(self, dim: int, eps: float, group_size: int | None = None):
        if group_size is not None and dim % group_size:
            raise ValueError(f"dim {dim} not divisible by group_size {group_size}")
        self.weight = torch.empty(dim)
        self.eps = eps
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        if self.group_size is not None:
            xf = xf.reshape(*xf.shape[:-1], -1, self.group_size)
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            out = out.flatten(-2)
        out = out * (1.0 + self.weight.float())
        return out.to(x.dtype)


class GatedResidual(BaseOP):
    """HF ``Qwen4ExpTextGatedResidual`` over the widened residual ``[T, hc * H]``.

    ``forward(x) -> (mixed [T, H], x, injection_weights [T, hc])``; with ``use_combine=False``
    (the final ``hyper_connection_mixer``) only ``mixed`` is returned."""

    def __init__(self, hidden_size: int, hc_count: int, hc_lowrank: int, eps: float,
                 use_combine: bool = True):
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        hc_hidden = hc_count * hidden_size
        self.hc_norm = QRMSNorm(hc_hidden, eps=eps, group_size=hidden_size)
        self.input_mix_weight_down = LinearReplicated(hc_hidden, hc_lowrank, has_bias=False)
        self.input_mix_weight_up = LinearReplicated(hc_lowrank, hc_hidden, has_bias=False)
        self._use_combine = use_combine
        if use_combine:
            self.block_inject_weight = LinearReplicated(hc_hidden, hc_count, has_bias=False)

    def forward(self, x: torch.Tensor):
        xn = self.hc_norm.forward(x)
        w = F.silu(self.input_mix_weight_down.forward(xn) / self.hc_count)
        w = torch.sigmoid(self.input_mix_weight_up.forward(w))
        w = w.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = (w * xn.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        if not self._use_combine:
            return mixed
        inj = 2 * torch.sigmoid(self.block_inject_weight.forward(xn) / self.hc_count)
        return mixed, x, inj

    @staticmethod
    def combine(hyper_input: torch.Tensor, out: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
        """``hyper_input + flatten(out[:, None, :] * inj[..., None])`` (HF decoder layer)."""
        return hyper_input + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)


class HFPartialRope:
    """Partial NeoX rope computed exactly like the HF reference: fp32 ``inv_freq`` (built on
    CPU with the reference formula), fp32 angles, cos/sin cast to the activation dtype, then
    ``x*cos + rotate_half(x)*sin`` on the first ``rotary_dim`` dims in that dtype. Used for the
    QSA indexer, whose block selection is a discrete top-k and so benefits from bit-level
    agreement with the reference."""

    def __init__(self, rotary_dim: int, base: float):
        self.rotary_dim = rotary_dim
        # explicit device: models are constructed under ``torch.device("meta")``
        self._inv_freq_cpu = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float, device="cpu") / rotary_dim)
        )
        self._inv_freq: dict = {}

    def cos_sin(self, positions: torch.Tensor, dtype: torch.dtype):
        dev = positions.device
        inv = self._inv_freq.get(dev)
        if inv is None:
            inv = self._inv_freq[dev] = self._inv_freq_cpu.to(dev)
        freqs = positions.float()[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """``x`` [N, ..., D] with ``cos``/``sin`` [N, rotary_dim] (broadcast over middle dims)."""
        rd = self.rotary_dim
        while cos.dim() < x.dim():
            cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        x_rope, x_pass = x[..., :rd], x[..., rd:]
        half = rd // 2
        rot = torch.cat((-x_rope[..., half:], x_rope[..., :half]), dim=-1)
        return torch.cat((x_rope * cos + rot * sin, x_pass), dim=-1)


__all__ = ["QRMSNorm", "GatedResidual", "HFPartialRope"]
