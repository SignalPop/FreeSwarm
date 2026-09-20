"""Qwen3-Next / Qwen3.5-MoE Multi-Token-Prediction (MTP / "nextn") head + self-speculative
decode. NOTE: authored on a box that can't fit the 35B for a live run; the module structure
mirrors the tested Qwen3_5 blocks and the checkpoint's 19 ``mtp.*`` tensors exactly, but the
weight-load fusion and the engine decode integration need validation on real hardware (see
MTP.md). Nothing here runs unless a model is served with ``--mtp`` (off by default).

The nextn head predicts token t+2 from (hidden_t, embed(token_{t+1})):

    e  = pre_fc_norm_embedding(embed(next_id))     # Gemma RMSNorm
    h  = pre_fc_norm_hidden(prev_hidden)           # Gemma RMSNorm
    x  = fc(cat([e, h], dim=-1))                    # Linear 2H -> H  (mtp.fc)
    x  = mtp_layer(x)                               # one full-attn + MoE block (mtp.layers.0)
    x  = norm(x)                                    # Gemma RMSNorm   (mtp.norm)
    logits = lm_head(x)                             # SHARED lm_head

Run it k times (feeding its own argmax back in as next_id, advancing its own KV slot) to draft
k tokens, then the base model verifies all k in one forward and accepts the matching prefix.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearReplicated, OPList

from .attention import Qwen3_5Attention
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5MTPLayer(BaseOP):
    """One MTP transformer block. Structurally identical to a full-attention
    ``Qwen3_5DecoderLayer`` (the nextn layer is always full attention, never GDN):
    ``x = x + self_attn(input_layernorm(x)); x = x + mlp(post_attention_layernorm(x))``.

    ``layer_id`` is the block's global attention-layer id in the KV pool. The MTP layer needs
    its own KV storage (one extra full-attention layer appended after the base model's layers);
    the engine must size the KV pool for it -- see MTP.md "KV / state".
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self.self_attn = Qwen3_5Attention(config, layer_id)
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm.forward(hidden)
        hidden = self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden + residual


class Qwen3_5MTP(BaseOP):
    """The nextn / MTP head. State-dict keys match the checkpoint's ``mtp.*`` tensors:
    ``pre_fc_norm_embedding``, ``pre_fc_norm_hidden``, ``fc``, ``layers.0.*``, ``norm``.
    Embedding and lm_head are SHARED with the base model (passed into ``forward``)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        h = config.hidden_size
        self.pre_fc_norm_embedding = GemmaRMSNorm(h, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(h, eps=config.rms_norm_eps)
        # mtp.fc.weight is [H, 2H]: projects concat(normed_embed, normed_hidden) -> H.
        self.fc = LinearReplicated(2 * h, h, has_bias=False)
        # OPList so the state-dict key is ``layers.0.*`` exactly as in the checkpoint.
        self.layers = OPList([Qwen3_5MTPLayer(config, layer_id)])
        self.norm = GemmaRMSNorm(h, eps=config.rms_norm_eps)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        next_ids: torch.Tensor,
        embed_tokens,
        lm_head,
    ) -> torch.Tensor:
        """logits for the token AFTER ``next_ids`` given the base model's ``prev_hidden``.

        ``prev_hidden`` is the base model's final pre-lm_head hidden state at the position that
        produced ``next_ids`` (i.e. ``Qwen3_5Model.forward`` output, already post final norm).
        Positions/KV for the inner attention come from ``get_global_ctx().batch`` -- the caller
        sets the batch up for the MTP layer's KV slot before calling (see MTP.md).
        """
        e = self.pre_fc_norm_embedding.forward(embed_tokens.forward(next_ids))
        h = self.pre_fc_norm_hidden.forward(prev_hidden)
        x = self.fc.forward(torch.cat([e, h], dim=-1))
        x = self.layers.op_list[0].forward(x)
        x = self.norm.forward(x)
        return lm_head.forward(x)


__all__ = ["Qwen3_5MTP", "Qwen3_5MTPLayer"]
