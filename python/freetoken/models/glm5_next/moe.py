"""GLM-5.3 sparse MoE block (``Glm5NextTextMoE``).

Routing is GLM/DeepSeek ``noaux_tc`` (fp32 sigmoid scores, selection-only
``e_score_correction_bias``, optional group-limited top-k, renormalize, scale by
``routed_scaling_factor``). Routed experts go through ``make_moe_layer`` with the
clamped SwiGLU (``silu_clamp``, limit = ``swiglu_limit``): the offload family (NVFP4
banks, Triton kernels) in serving, resident bf16 experts otherwise. The always-on shared
expert is a bf16 clamped SwiGLU MLP.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

from .config import MOE_ACTIVATION
from .ops import ClampedSwiGLUMLP


class Glm5NextMoE(BaseOP):
    def __init__(self, config, layer_id: int):
        args = config.glm_dsa_args
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        # fp32 in the checkpoint and in HF (_keep_in_fp32_modules_strict).
        self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)
        # Experts are indexed by MoE layer (global layer minus the dense prefix), matching
        # how the NVFP4 loader packs the offload banks.
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            activation=MOE_ACTIVATION,
            renormalize=config.norm_topk_prob,
            extra_attrs={"swiglu_limit": args.swiglu_limit},
        )
        self.shared_experts = ClampedSwiGLUMLP(
            config.hidden_size,
            config.moe_intermediate_size * max(1, config.n_shared_experts),
            args.swiglu_limit,
        )

    def _group_limited(self, scores_for_choice: torch.Tensor) -> torch.Tensor:
        m = scores_for_choice.shape[0]
        e, g = self.num_experts, self.n_group
        group_scores = scores_for_choice.view(m, g, e // g).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = group_mask.unsqueeze(-1).expand(m, g, e // g).reshape(m, e)
        return scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(x.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.float()
        if self.n_group > 1:
            scores_for_choice = self._group_limited(scores_for_choice)
        topk_ids = torch.topk(scores_for_choice, self.top_k, dim=-1)[1]
        topk_weights = scores.gather(-1, topk_ids)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.float().contiguous(), topk_ids.to(torch.int32).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The shared expert reads x BEFORE the routed kernels: the resident bf16 fused
        # path overwrites its input with the routed output.
        shared = self.shared_experts.forward(x)
        topk_weights, topk_ids = self.route(x)
        routed = self.experts.routed_forward(x.contiguous(), topk_weights, topk_ids)
        return routed + shared


__all__ = ["Glm5NextMoE"]
