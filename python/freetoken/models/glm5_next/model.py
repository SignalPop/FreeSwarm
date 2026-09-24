"""GLM-5.3-Flash text model (``Glm5NextTextModel`` + lm_head), FreeToken layout.

Decoder block (``Glm5NextTextDecoderLayer``), with ``hc_mult`` residual streams
``S [T, hc, D]``::

    post, comb, h = attn_hc(S);  h = mixer(input_layernorm(h))
    S = post * h + comb^T @ S
    post, comb, h = ffn_hc(S);   h = mlp(post_attention_layernorm(h))
    S = post * h + comb^T @ S

where the mixer is KDA (linear layers) or MLA+DSA, and the MLP is the dense clamped
SwiGLU (leading ``first_k_dense_replace`` layers) or the MoE block. Embeddings are
broadcast to every stream; the output is ``norm(mean_streams(S))``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .args import LINEAR
from .attention import Glm5NextAttention
from .kda import Glm5NextKDA
from .moe import Glm5NextMoE
from .ops import ClampedSwiGLUMLP, HyperConnection, RMSNorm

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Glm5NextDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.glm_dsa_args
        self._layer_id = layer_id
        self._is_linear = args.layer_types[layer_id] == LINEAR
        if self._is_linear:
            self.linear_attn = Glm5NextKDA(config, layer_id)
        else:
            self.self_attn = Glm5NextAttention(config, layer_id)
        if args.mlp_layer_types[layer_id] == "sparse":
            self.mlp: BaseOP = Glm5NextMoE(config, layer_id)
        else:
            self.mlp = ClampedSwiGLUMLP(
                config.hidden_size, config.intermediate_size, args.swiglu_limit
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc = (args.hc_mult, config.hidden_size, args.hc_eps, args.hc_sinkhorn_iters,
              config.rms_norm_eps)
        self.attn_hc = HyperConnection(*hc)
        self.ffn_hc = HyperConnection(*hc)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        post, comb, h = self.attn_hc.forward(streams)
        h = self.input_layernorm.forward(h)
        h = self.linear_attn.forward(h) if self._is_linear else self.self_attn.forward(h)
        streams = HyperConnection.expand(h, streams, post, comb)

        post, comb, h = self.ffn_hc.forward(streams)
        h = self.post_attention_layernorm.forward(h)
        h = self.mlp.forward(h)
        return HyperConnection.expand(h, streams, post, comb)


class Glm5NextModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self._hc_mult = config.glm_dsa_args.hc_mult
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = OPList(
            [Glm5NextDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        streams = x.unsqueeze(1).expand(-1, self._hc_mult, -1).contiguous()
        for layer in self.layers.op_list:
            streams = layer.forward(streams)
        return self.norm.forward(streams.mean(dim=1))


class Glm5NextForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        from freetoken.distributed import get_tp_info

        if get_tp_info().size != 1:
            raise NotImplementedError(
                "glm5_next is served at TP=1 only (correctness-first port: the KDA/MLA/HC "
                "modules are replicated, not sharded)"
            )
        self.model = Glm5NextModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing: materialize the MLA kv_b split (and free the
        checkpoint layout) so the KV budget sees the final resident footprint."""
        for layer in self.model.layers.op_list:
            if not layer._is_linear:
                layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Glm5NextForCausalLM"]
