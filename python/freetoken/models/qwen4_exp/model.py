from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .layers import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import Qwen4ExpPLE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpDecoderLayer(BaseOP):
    """HF ``Qwen4ExpTextDecoderLayer`` over the widened residual ``h`` [T, hc*H]:

        h = h + ple(h, ids)                                    (PLE layers only)
        x, h0, w = attn_hyper_connection(h); h = h0 + w * mixer(x)   (per stream)
        x, h0, w = mlp_hyper_connection(h);  h = h0 + w * moe(x)
    """

    def __init__(self, config: "ModelConfig", layer_id: int):
        a = config.qwen4_args
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant="none",  # GDN projections are bf16 in both checkpoints
                attn_quant="none",
                gate_activation=a.output_gate_type,
            )
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        spec = a.ple_for_layer(layer_id)
        self._has_ple = spec is not None
        if self._has_ple:
            self.ple = Qwen4ExpPLE(a, spec, eps=config.rms_norm_eps)
        hc_args = (config.hidden_size, a.hc_count, a.hc_lowrank, config.rms_norm_eps)
        self.attn_hyper_connection = GatedResidual(*hc_args)
        self.mlp_hyper_connection = GatedResidual(*hc_args)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, h: torch.Tensor, ids_cpu: torch.Tensor | None) -> torch.Tensor:
        if self._has_ple:
            h = self.ple.forward(h, ids_cpu)
        x, h0, w = self.attn_hyper_connection.forward(h)
        x = self.linear_attn.forward(x) if self._is_linear else self.self_attn.forward(x)
        h = GatedResidual.combine(h0, x, w)
        x, h0, w = self.mlp_hyper_connection.forward(h)
        x = self.mlp.forward(x)
        return GatedResidual.combine(h0, x, w)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: "ModelConfig"):
        a = config.qwen4_args
        self._hc = a.hc_count
        self._has_ple = bool(a.ple_layers)
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(
            config.hidden_size, a.hc_count, a.hc_lowrank, config.rms_norm_eps, use_combine=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # PLE hashes token ids on the host (the n-gram table lives in host memory).
        ids_cpu = input_ids.to("cpu") if self._has_ple else None
        h = self.embed_tokens.forward(input_ids).repeat(1, self._hc)
        for layer in self.layers.op_list:
            h = layer.forward(h, ids_cpu)
        return self.hyper_connection_mixer.forward(h)

    def ple_layers(self) -> list[Qwen4ExpPLE]:
        return [layer.ple for layer in self.layers.op_list if layer._has_ple]


class Qwen4ExpForCausalLM(BaseLLMModel):
    """Qwen3.8-Flash-Next (HF ``Qwen4ExpForConditionalGeneration``), text-only. There is no
    final RMSNorm: the final hyper-connection mixer's grouped norm plays that role."""

    def __init__(self, config: "ModelConfig"):
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        """Open the PLE n-gram tables (memory-mapped checkpoint shards, host side)."""
        for ple in self.model.ple_layers():
            ple.load_table()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen4ExpForCausalLM"]
