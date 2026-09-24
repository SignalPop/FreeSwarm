"""Shared helpers for the Qwen4-Exp tests: a tiny random-weight HF reference checkpoint
(``transformers`` ``Qwen4ExpForConditionalGeneration``) plus a word-level tokenizer, saved to
disk so the real FreeToken loader/engine path serves it."""

from __future__ import annotations

import os

import torch

TINY_TEXT = dict(
    vocab_size=512,
    hidden_size=128,
    num_hidden_layers=5,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    hidden_act="silu",
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    },
    linear_conv_kernel_dim=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    num_experts_per_tok=2,
    num_experts=8,
    # real checkpoints spell the QSA layers "full_attention"; HF rewrites them
    layer_types=[
        "linear_attention", "linear_attention", "full_attention", "linear_attention",
        "full_attention",
    ],
    hc_count=4,
    hc_lowrank=32,
    ple_layer_ids=[2],  # 1-indexed -> decoder layer 1 (a linear layer)
    ple_embed_dim=128,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=1000,
    make_ngram_vocab_size_divisible_by=128,
    seed=1234,
    split_ngram_parts=4,
    indexer_n_heads=4,
    indexer_kv_heads=1,
    indexer_head_dim=32,
    # budget 8 / ratio 4 -> 2 blocks: contexts > 11 tokens are genuinely sparse
    indexer_budget=8,
    indexer_compress_ratio=4,
    output_gate_type="sigmoid",
    eos_token_id=5,
    bos_token_id=5,
    tie_word_embeddings=False,
)
EOS = 5


def build_hf_model(seed: int = 0, **overrides):
    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    text = dict(TINY_TEXT, **overrides)
    cfg = Qwen4ExpConfig(
        text_config=text,
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=text["hidden_size"]),
    )
    torch.manual_seed(seed)
    model = Qwen4ExpForConditionalGeneration(cfg)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".visual." in name or name.startswith("model.visual"):
                continue
            if "ngram_embedding" in name:
                p.copy_(torch.randn(p.shape, generator=g))  # make PLE matter
            elif name.endswith("norm.weight") or "_layernorm" in name or "hc_norm" in name \
                    or "norm_key" in name or "norm_query" in name or "norm_conv" in name \
                    or "q_norm" in name or "k_norm" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)  # exercise the (1+w) form
            elif "ple.conv1d" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
            elif "linear_attn.conv1d" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif name.endswith("A_log") or name.endswith("dt_bias"):
                continue
            elif "embed_tokens" in name or "lm_head" in name:
                p.copy_(torch.randn(p.shape, generator=g))
            elif p.dim() >= 2:
                fan_in = p.shape[-1]
                p.copy_(torch.randn(p.shape, generator=g) / fan_in ** 0.5)
    return model.to(torch.bfloat16).eval()


def write_tokenizer(path: str, vocab_size: int) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(vocab_size)}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="t0"))
    tok.pre_tokenizer = WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=f"t{EOS}",
                                   bos_token=f"t{EOS}", unk_token="t0")
    fast.save_pretrained(path)


def save_tiny_checkpoint(path: str, seed: int = 0, **overrides):
    model = build_hf_model(seed, **overrides)
    model.save_pretrained(path)
    write_tokenizer(path, model.config.text_config.vocab_size)
    return model


def cuda_home_for_jit() -> None:
    """The engine JIT-compiles a few CUDA kernels; point it at the venv's CUDA 13 toolkit
    (matching torch cu130) when the system nvcc is a different major."""
    if os.environ.get("CUDA_HOME"):
        return
    import sys

    cand = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cu13")
    if os.path.exists(os.path.join(cand, "bin")):
        os.environ["CUDA_HOME"] = cand
        os.environ["CUDA_PATH"] = cand
