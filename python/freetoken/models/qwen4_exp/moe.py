"""Qwen4-Exp sparse MoE block (HF ``Qwen4ExpTextSparseMoeBlock``): softmax top-k routed
experts (renormalized) + a sigmoid-gated shared expert. Same math as Qwen3.5's block, but the
shared expert is ALWAYS bf16 here -- both released checkpoints quantize only the routed
experts (the Qwen3.5 block would make the shared expert block-fp8 under ``fp8_block``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _SharedExpert(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen4ExpMoE(BaseOP):
    def __init__(self, config: "ModelConfig", layer_id: int):
        weight_format = "fp8_block" if config.expert_quant == "fp8_block" else "bf16"
        self.experts = make_moe_layer(
            config, layer_id=layer_id, renormalize=config.norm_topk_prob,
            weight_format=weight_format,
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(config.hidden_size, config.shared_expert_intermediate_size)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # router + shared expert first: the fused routed kernel may write ``x`` in place.
        router_logits = self.gate.forward(x)
        shared = self.shared_expert.forward(x)
        shared = torch.sigmoid(self.shared_expert_gate.forward(x)) * shared
        routed = self.experts.forward(hidden_states=x, router_logits=router_logits)
        return routed + shared


__all__ = ["Qwen4ExpMoE"]
