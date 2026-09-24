"""Qwen4-Exp (Qwen3.8-Flash-Next) vs the HF reference (``transformers.models.qwen4_exp``).

* config parsing against the real released config.json shapes (FP8 + NVFP4 checkpoints);
* PLE n-gram hashing constants (primes, multipliers) and ids/embeddings, incl. the eos
  boundary and chunked continuation, bit-exact vs ``Qwen4ExpTextNGramEmbedding``;
* QSA token selection vs ``Qwen4ExpTextQSAIndexer`` on identical inputs;
* pool / engine capability wiring (QSA side slab budgeted, eager-only, naive cache);
* end to end: a tiny random-weight checkpoint saved by transformers, served by the real
  FreeToken stack (subprocess), logits compared against HF at every prefill-last and decode
  position -- one-shot prefill and chunked prefill -- with contexts past the QSA budget so the
  sparse path (not just the dense-equivalent fast path) is exercised.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen4_exp_tiny import EOS, TINY_TEXT, build_hf_model, save_tiny_checkpoint  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# ---- the real checkpoints' text_config (Qwen/Qwen3.8-Flash-Next-FP8 and
# RadixArk/Qwen3.8-Flash-Next-NVFP4 ship the same text tower; only quantization differs) ----
_REAL_TEXT = {
    "attention_bias": False, "bos_token_id": 248044, "eos_token_id": 248044,
    "full_attention_interval": 4, "hc_count": 4, "hc_lowrank": 320, "head_dim": 256,
    "heads_per_ngram": 8, "hidden_act": "silu", "hidden_size": 2560, "indexer_budget": 2048,
    "indexer_compress_ratio": 4, "indexer_head_dim": 128, "indexer_kv_heads": 1,
    "indexer_n_heads": 4,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 48, "linear_value_head_dim": 128,
    "make_ngram_vocab_size_divisible_by": 128, "max_position_embeddings": 262144,
    "model_type": "qwen4_exp_text", "moe_intermediate_size": 640, "ngram_size": 3,
    "ngram_vocab_size_base": 20000000, "num_attention_heads": 24, "num_experts": 512,
    "num_experts_per_tok": 10, "num_hidden_layers": 48, "num_key_value_heads": 2,
    "output_gate_type": "sigmoid", "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4,
    "ple_embed_dim": 2560, "ple_layer_ids": [2], "rms_norm_eps": 1e-06,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                        "rope_type": "default"},
    "shared_expert_intermediate_size": 640, "split_ngram_parts": 128,
    "tie_word_embeddings": False, "vocab_size": 248320,
}
_FP8_QUANT = {
    "quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
    "modules_to_not_convert": ["lm_head", "model.language_model.embed_tokens"],
}
_NVFP4_QUANT = {
    "config_groups": {"group_0": {
        "input_activations": {"dynamic": False, "group_size": 16, "num_bits": 4, "type": "float"},
        "targets": ["Linear"],
        "weights": {"dynamic": False, "group_size": 16, "num_bits": 4, "type": "float"}}},
    "ignore": ["model.embed_tokens", "mtp.*", "*.self_attn.*", "*.linear_attn.*", "*.mlp.gate*",
               "*.mlp.shared_expert.*", "*.mlp.shared_expert_gate*", "*hyper_connection*",
               "*.ple.*", "model.visual.*", "lm_head"],
    "producer": {"name": "modelopt", "version": "0.46.0"},
    "quant_algo": "NVFP4", "quant_method": "modelopt",
}


class _Cfg:
    def __init__(self, data: dict):
        self._data = data
        for k, v in data.items():
            setattr(self, k, v)

    def to_dict(self):
        return self._data


def _real_hf_config(quant: dict) -> _Cfg:
    return _Cfg({
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "image_token_id": 248056,
        "text_config": _Cfg(dict(_REAL_TEXT)),
        "quantization_config": quant,
        "_name_or_path": "Qwen/Qwen3.8-Flash-Next-FP8",
    })


@pytest.mark.parametrize("quant,expert_quant", [(_FP8_QUANT, "fp8_block"), (_NVFP4_QUANT, "nvfp4")])
def test_parse_real_config(quant, expert_quant):
    from freetoken.attention import AttnType
    from freetoken.models.qwen4_exp import parse_config

    mc = parse_config(_real_hf_config(quant))
    a = mc.qwen4_args
    assert mc.num_layers == 48 and mc.hidden_size == 2560 and mc.vocab_size == 248320
    assert (mc.num_qo_heads, mc.num_kv_heads, mc.head_dim) == (24, 2, 256)
    assert mc.rotary_config.rotary_dim == 64 and mc.rotary_config.base == 10000000
    assert (mc.num_experts, mc.num_experts_per_tok, mc.moe_intermediate_size) == (512, 10, 640)
    assert mc.shared_expert_intermediate_size == 640 and mc.norm_topk_prob
    assert mc.expert_quant == expert_quant
    assert (mc.attn_quant, mc.dense_quant, mc.lm_head_quant) == ("none", "none", "none")
    assert mc.weight_block_size == ((128, 128) if expert_quant == "fp8_block" else None)
    # every 4th layer is QSA; the rest GDN (48 value heads over 16 key heads)
    assert a.qsa_layer_ids == tuple(range(3, 48, 4))
    g = mc.linear_attention_group()
    assert (g.num_key_heads, g.num_value_heads, g.key_head_dim, g.value_head_dim) == (16, 48, 128, 128)
    assert len(g.layer_ids) == 36
    assert [mc.attn_type_for_layer(i) for i in (0, 3)] == [AttnType.LINEAR, AttnType.FULL]
    # QSA geometry: budget 2048 tokens in 4-token blocks -> 512 blocks; dense up to 2051
    assert (a.indexer_n_heads, a.indexer_head_dim, a.block_topk, a.dense_kv_limit) == (4, 128, 512, 2051)
    assert mc.qsa_index_head_dim == 128
    assert a.output_gate_type == "sigmoid" and (a.hc_count, a.hc_lowrank) == (4, 320)
    # PLE on (1-indexed) layer 2 == decoder layer 1; 16 heads x 160 dims
    assert len(a.ple_layers) == 1 and a.ple_layers[0].layer_id == 1
    assert (a.ngram_heads, a.ngram_head_dim, a.eos_token_id) == (16, 160, 248044)
    spec = a.ple_layers[0]
    # checkpoint: 128 shards x 2_500_012 rows == the padded table
    assert spec.padded_vocab_size == 128 * 2_500_012
    assert mc.eager_only and not mc.prefix_reuse_supported
    assert a.model_path == "Qwen/Qwen3.8-Flash-Next-FP8"


def test_real_config_hash_constants_match_hf():
    """Primes / multipliers for the real 20M-row PLE heads == the HF reference helpers."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

    from freetoken.models.qwen4_exp import parse_config

    spec = parse_config(_real_hf_config(_FP8_QUANT)).qwen4_args.ple_layers[0]
    want_sizes = [hf._find_nth_prime_after(20_000_000 - 1, h + 1) for h in range(16)]
    assert list(spec.head_vocab_sizes) == want_sizes
    assert list(spec.multipliers) == hf._build_layer_multipliers(248320, 3, 0, 1234).tolist()


def test_registry_aot_and_parsers():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS
    from freetoken.models.register import get_model_spec

    for arch in ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"):
        assert get_model_spec(arch).module == "freetoken.models.qwen4_exp"
    claimed = {m.architecture for m in SUPPORTED_MODELS} | {
        a for m in SUPPORTED_MODELS for a in m.arch_aliases
    }
    assert {"Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"} <= claimed
    names = {m.name for m in SUPPORTED_MODELS}
    assert {"Qwen/Qwen3.8-Flash-Next-FP8", "RadixArk/Qwen3.8-Flash-Next-NVFP4"} <= names


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_parser_inference(model_type):
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    cfg = _Cfg({"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": model_type})
    with patch("freetoken.utils.cached_load_hf_config", lambda _p: cfg):
        args, _ = parse_args(["--model", "/models/anon"])
    assert (args.tool_call_parser, args.reasoning_parser) == ("qwen3_coder", "qwen3")


def test_pool_family_and_kv_cost():
    from types import SimpleNamespace

    from freetoken.kvcache import resolve_pool_class
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.models.qwen4_exp import parse_config

    mc = parse_config(_real_hf_config(_NVFP4_QUANT))
    assert resolve_pool_class(mc) is QSAKVCache
    cfg = SimpleNamespace(model_config=mc, page_size=1, dtype=torch.bfloat16,
                          tp_info=SimpleNamespace(size=1))
    per_page, *_ = QSAKVCache.kv_cost(cfg)
    base, *_ = MHAKVCache.kv_cost(cfg)
    # 12 QSA layers x (K+V: 2 x 2 heads x 256 x 2B) + 12 x 128-wide bf16 index keys
    assert base == 12 * 2 * 2 * 256 * 2
    assert per_page - base == 12 * 128 * 2


@needs_cuda
def test_qsa_pool_store_and_rebuild(monkeypatch):
    from freetoken.distributed.info import DistributedInfo
    from freetoken.kvcache.qsa_pool import QSAKVCache

    monkeypatch.setattr("freetoken.kvcache.mha_pool.get_tp_info",
                        lambda: DistributedInfo(rank=0, size=1))
    dev = torch.device("cuda")
    pool = QSAKVCache(num_kv_heads=2, num_layers=5, head_dim=64, num_pages=10, page_size=1,
                      dtype=torch.bfloat16, device=dev, index_head_dim=32, layer_ids=(2, 4))
    k = torch.randn(3, 32, device=dev, dtype=torch.bfloat16)
    loc = torch.tensor([1, 5, 7], device=dev, dtype=torch.int32)
    pool.store_index_k(k, loc, 4)
    assert torch.equal(pool.index_k_rows(4)[loc.long()], k)
    assert pool.index_k_rows(2).abs().sum() == 0
    pool.rebuild(20)
    assert pool.index_k_rows(2).shape == (20, 32)
    with pytest.raises(KeyError):
        pool.index_k_rows(0)  # GDN layer: no paged storage


def test_adjust_config_forces_eager_and_naive():
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config
    from freetoken.models.qwen4_exp import parse_config

    mc = parse_config(_real_hf_config(_NVFP4_QUANT))
    cfg = SimpleNamespace(
        model_config=mc, cuda_graph_bs=None, cuda_graph_max_bs=None, max_running_req=4,
        cache_type="radix", attention_backend="triton", moe_backend="offload",
        moe_cache_size=1024, moe_cache_rate=None, moe_cache_auto=False, moe_cpu_layers=None,
        swa_full_tokens_ratio=0.2, page_size=1, dtype=torch.bfloat16, nvfp4_backend="triton",
        num_token_override=None, num_page_override=None, max_seq_len_override=None,
        moe_prefill_overlap=True, moe_cpu_threads=0, moe_hybrid_max_fetch=-1,
        moe_prefill_hit_d2d=False, expert_load="auto", kv_reserve_tokens=8192,
        max_extend_tokens=8192,
    )
    try:
        _adjust_config(cfg)
    except Exception as e:  # later, unrelated duck-typing gaps must not mask the asserts
        if cfg.cuda_graph_max_bs != 0 or cfg.cache_type != "naive":
            raise e
    assert cfg.cuda_graph_bs == [] and cfg.cuda_graph_max_bs == 0
    assert cfg.cache_type == "naive"


# ------------------------------------------------------------------------------------------
# PLE n-gram hashing
# ------------------------------------------------------------------------------------------
def _tiny_model_config():
    from transformers import Qwen4ExpConfig

    from freetoken.models.qwen4_exp import parse_config

    cfg = Qwen4ExpConfig(text_config=dict(TINY_TEXT))
    return parse_config(cfg)


def test_tiny_hash_buffers_match_hf():
    hf = build_hf_model()
    emb = hf.model.language_model.layers[1].ple.ple_embedding
    spec = _tiny_model_config().qwen4_args.ple_layers[0]
    assert list(spec.multipliers) == emb.layer_multipliers.tolist()
    assert list(spec.head_vocab_sizes) == emb.ngram_heads_vocab_sizes.tolist()
    assert list(spec.head_offsets) == emb.ngram_heads_offsets.tolist()
    assert spec.padded_vocab_size == emb.ngram_embedding.num_embeddings


def test_ngram_embedding_bit_exact_chunked():
    """Our host gather over (history + new ids) == HF's full-sequence n-gram embedding, for
    a sequence with eos boundaries, split into arbitrary chunks (prefill chunks + decode)."""
    from transformers import DynamicCache

    from freetoken.models.qwen4_exp.ple import NGramTable, ngram_row_ids

    hf = build_hf_model()
    emb = hf.model.language_model.layers[1].ple.ple_embedding
    a = _tiny_model_config().qwen4_args
    spec = a.ple_layers[0]
    table = NGramTable.from_tensor(emb.ngram_embedding.weight.data)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(6, 512, (1, 41), generator=g)
    ids[0, [0, 9, 10, 25]] = EOS  # eos at start, back-to-back eos, mid-sequence eos
    with torch.no_grad():
        want = emb(ids, None)[0]  # [S, ple_dim]
        # HF's own cached path must agree with its full path (sanity of the reference)
        cache = DynamicCache(config=hf.config.text_config)
        pieces = [emb(ids[:, s:e], cache)[0] for s, e in ((0, 13), (13, 14), (14, 30), (30, 41))]
        assert torch.equal(torch.cat(pieces), want)

    hist = torch.full((2,), EOS, dtype=torch.long)
    got = []
    for s, e in ((0, 5), (5, 6), (6, 7), (7, 33), (33, 41)):
        seg = ids[0, s:e]
        rows = ngram_row_ids(hist, seg, spec, a.ngram_size, a.heads_per_ngram, EOS)
        got.append(table.gather(rows.reshape(-1)).view(e - s, -1))
        hist = torch.cat([hist, seg])[-2:]
    assert torch.equal(torch.cat(got), want)


# ------------------------------------------------------------------------------------------
# QSA selection
# ------------------------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("seq,budget,ratio", [(37, 8, 4), (64, 16, 4), (50, 12, 3)])
def test_qsa_selection_matches_hf_indexer(seq, budget, ratio):
    from transformers.masking_utils import create_causal_mask

    from freetoken.models.qwen4_exp.attention import qsa_block_keys, qsa_select
    from freetoken.models.qwen4_exp.layers import HFPartialRope, QRMSNorm

    hf = build_hf_model(indexer_budget=budget, indexer_compress_ratio=ratio).cuda()
    tcfg = hf.config.text_config
    attn = hf.model.language_model.layers[2].self_attn
    idx = attn.indexer
    torch.manual_seed(1)
    x = torch.randn(1, seq, tcfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(seq, device="cuda")
    rot = hf.model.language_model.rotary_emb
    pos3 = pos.view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rot(x, pos3)
    mask = create_causal_mask(config=tcfg, inputs_embeds=x, attention_mask=None,
                              past_key_values=None, position_ids=pos[None],
                              allow_is_causal_skip=False)
    with torch.no_grad():
        sel = idx(x, (cos, sin), mask, None)[0, 0]  # [S, S] bool / float (0 == selected)
        if sel.dtype != torch.bool:
            sel = sel == 0
        # ours: identical projection -> HF-exact rope -> selection
        rope = HFPartialRope(int(tcfg.head_dim * 0.25), 10000.0)
        qk = idx.index_qk_proj(x[0])
        d = tcfg.indexer_head_dim
        q, raw = torch.split(qk, [tcfg.indexer_n_heads * d, d], dim=-1)
        qn = QRMSNorm(d, eps=tcfg.rms_norm_eps)
        qn.weight = idx.q_layernorm.weight.data
        kn = QRMSNorm(d, eps=tcfg.rms_norm_eps)
        kn.weight = idx.k_layernorm.weight.data
        c, s = rope.cos_sin(pos, torch.bfloat16)
        iq = rope.apply(qn.forward(q.reshape(seq, -1, d)), c, s)
        bk = qsa_block_keys(raw.contiguous(), ratio, kn, rope)
        ti, ok = qsa_select(iq, bk, pos, ratio, budget // ratio)
    got = torch.zeros(seq, seq + 1, dtype=torch.bool, device="cuda")
    got.scatter_(1, torch.where(ok, ti, seq), True)  # invalid slots -> dropped column
    assert torch.equal(got[:, :seq], sel)
    # the case is genuinely sparse
    assert (sel.sum(-1) < pos + 1).any()


@needs_cuda
@pytest.mark.parametrize("cached", [0, 29])
def test_qsa_sparse_attention_matches_hf(cached):
    """The sparse gather-attend path (``_sparse_request``: pooled block keys from the raw-key
    slab rows, selection, K/V row gather, masked softmax) vs HF ``Qwen4ExpTextAttention`` on
    identical inputs; ``cached`` > 0 runs only the trailing queries against the full context,
    as a chunked-prefill continuation / decode does."""
    from types import SimpleNamespace

    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention
    from freetoken.models.qwen4_exp.layers import HFPartialRope, QRMSNorm

    seq = 45
    hf = build_hf_model().cuda()
    tcfg = hf.config.text_config
    attn = hf.model.language_model.layers[2].self_attn
    torch.manual_seed(2)
    x = torch.randn(1, seq, tcfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(seq, device="cuda")
    cos, sin = hf.model.language_model.rotary_emb(x, pos.view(1, 1, -1).expand(3, 1, -1))
    mask = create_causal_mask(config=tcfg, inputs_embeds=x, attention_mask=None,
                              past_key_values=None, position_ids=pos[None],
                              allow_is_causal_skip=False)
    nh, kvh, hd = tcfg.num_attention_heads, tcfg.num_key_value_heads, tcfg.head_dim
    hi, di = tcfg.indexer_n_heads, tcfg.indexer_head_dim
    with torch.no_grad():
        want = attn(x, (cos, sin), mask)[0][0]
        h = x[0]
        q, gate = torch.chunk(attn.q_proj(h).view(seq, nh, 2 * hd), 2, dim=-1)
        q = attn.q_norm(q).transpose(0, 1)[None]
        k = attn.k_norm(attn.k_proj(h).view(seq, kvh, hd)).transpose(0, 1)[None]
        v = attn.v_proj(h).view(seq, kvh, hd)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k = q[0].transpose(0, 1).contiguous(), k[0].transpose(0, 1).contiguous()
        iqk = attn.indexer.index_qk_proj(h)
        iq, raw = torch.split(iqk, [hi * di, di], dim=-1)
        rope = HFPartialRope(int(hd * 0.25), 10000.0)
        iq = attn.indexer.q_layernorm(iq.reshape(seq, hi, di))
        c, s = rope.cos_sin(pos, torch.bfloat16)
        iq = rope.apply(iq, c, s)
        kn = QRMSNorm(di, eps=tcfg.rms_norm_eps)
        kn.weight = attn.indexer.k_layernorm.weight.data
        fake = SimpleNamespace(
            _ratio=tcfg.indexer_compress_ratio, indexer=SimpleNamespace(k_layernorm=kn),
            _index_rope=rope, num_kv=kvh, num_q=nh, head_dim=hd,
            _block_topk=tcfg.indexer_budget // tcfg.indexer_compress_ratio, _scale=hd ** -0.5,
        )
        # scatter the request into non-contiguous pool rows, as the paged pool does
        perm = torch.randperm(3 * seq, device="cuda")[:seq]
        K = torch.zeros(3 * seq, kvh, hd, device="cuda", dtype=k.dtype)
        V = torch.zeros_like(K)
        IK = torch.zeros(3 * seq, di, device="cuda", dtype=raw.dtype)
        K[perm], V[perm], IK[perm] = k.view(seq, kvh, hd), v, raw
        core = Qwen4ExpAttention._sparse_request(
            fake, q[cached:], iq[cached:], perm, cached, K, V, IK
        )
        got = attn.o_proj(core.reshape(seq - cached, -1) * torch.sigmoid(gate[cached:].reshape(seq - cached, -1)))
    err = (got.float() - want[cached:].float()).abs().max().item()
    assert err < 2e-2 * want.float().abs().max().item(), err


# ------------------------------------------------------------------------------------------
# end to end through the real engine
# ------------------------------------------------------------------------------------------
# Two regimes. DENSE: indexer budget >= every context and top-k == all experts, so the model
# has no discrete choice left (QSA == causal attention, routing == soft mixture) and FreeToken
# must track HF to bf16 noise at EVERY position (~0.06-0.15 on logits of std ~5.5).
# SPARSE: budget 8 tokens (2 blocks) and top-2 of 8 experts on contexts up to 78 tokens. The
# tiny random model is then full of near-ties, and bf16 noise flips discrete choices -- the HF
# bf16 and fp32 references disagree with EACH OTHER by >1.4 at several positions -- so there
# the bar is "within noise of the bf16 or the fp32 reference" at nearly every position (the
# selection itself is checked exactly, on identical inputs, by the QSA unit tests above).
_DENSE = dict(indexer_budget=512, num_experts_per_tok=8)
_TOL = 0.3


def _run_engine(tmp_path, max_extend: int, prompts, n_new: int, force_sparse: bool = False,
                **overrides):
    ckpt = tmp_path / "ckpt"
    hf = save_tiny_checkpoint(str(ckpt), **overrides)
    ppath = tmp_path / "prompts.json"
    ppath.write_text(json.dumps(prompts))
    out = tmp_path / "out.pt"
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen4_exp_engine_runner.py")
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.path.join(root, "python") + os.pathsep + env.get("PYTHONPATH", "")
    env["FREETOKEN_QWEN4_QSA_FORCE_SPARSE"] = "1" if force_sparse else "0"
    proc = subprocess.run(
        [sys.executable, runner, str(ckpt), str(out), str(max_extend), str(n_new), str(ppath)],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-8000:]
    return hf, torch.load(out, weights_only=False)


def _prompts():
    g = torch.Generator().manual_seed(3)
    prompts = []
    for n in (7, 40, 23, 70):
        ids = torch.randint(6, 512, (n,), generator=g).tolist()
        if n == 40:
            ids[10] = EOS  # n-gram eos boundary inside the prompt
        prompts.append(ids)
    return prompts


def _compare(hf, res, n_new: int):
    """Per checked position: (|ft - hf_bf16|max, |ft - hf_fp32|max, argmax == fp32 argmax)."""
    assert res["pool"] == "QSAKVCache"
    assert res["cache_type"] == "naive" and res["graph_bs"] == []
    assert all(len(o) >= n_new - 1 for o in res["outputs"])
    hf16 = hf.cuda()
    hf32 = copy.deepcopy(hf16).float()
    rows = []
    for uid, (p, o) in enumerate(zip(res["prompts"], res["outputs"])):
        toks = p + o
        with torch.no_grad():
            ids = torch.tensor([toks], device="cuda")
            r16 = hf16(input_ids=ids).logits[0].float().cpu()
            r32 = hf32(input_ids=ids).logits[0].float().cpu()
        for pos in range(len(p) - 1, len(toks)):
            got = res["logits"].get((uid, pos))
            if got is None:
                continue
            rows.append(((got - r16[pos]).abs().max().item(), (got - r32[pos]).abs().max().item(),
                         got.argmax().item() == r32[pos].argmax().item()))
    # every prompt's prefill-last position and every decode position was checked
    assert len(rows) == sum(len(o) + 1 for o in res["outputs"])
    return rows


@needs_cuda
@pytest.mark.parametrize("force_sparse", [False, True], ids=["backend_path", "sparse_path"])
@pytest.mark.parametrize("max_extend", [8192, 16], ids=["one_shot_prefill", "chunked_prefill"])
def test_engine_dense_regime_matches_hf(tmp_path, max_extend, force_sparse):
    """``force_sparse`` routes the (dense-equivalent) QSA layers through the model's own
    sparse gather path -- paged K/V + raw-key slab rows, block pooling, selection of every
    block -- instead of the attention backend, pinning that path's engine integration
    (chunked continuation, decode) to the strict bar."""
    n_new = 8
    hf, res = _run_engine(tmp_path, max_extend, _prompts(), n_new, force_sparse, **_DENSE)
    rows = _compare(hf, res, n_new)
    d16 = [r[0] for r in rows]
    d32 = [r[1] for r in rows]
    print(f"[qwen4_exp dense e2e max_extend={max_extend} force_sparse={force_sparse}] "
          f"positions={len(rows)} "
          f"max|ft-hf_bf16|={max(d16):.4f} max|ft-hf_fp32|={max(d32):.4f} "
          f"mean={sum(d32) / len(d32):.4f} argmax==fp32 {sum(r[2] for r in rows)}/{len(rows)} "
          f"backend={res['attention_backend']}")
    assert max(d16) < _TOL and max(d32) < _TOL


@needs_cuda
@pytest.mark.parametrize("max_extend", [8192, 16], ids=["one_shot_prefill", "chunked_prefill"])
def test_engine_sparse_regime_matches_hf(tmp_path, max_extend):
    n_new = 8
    hf, res = _run_engine(tmp_path, max_extend, _prompts(), n_new)
    rows = _compare(hf, res, n_new)
    best = [min(r[0], r[1]) for r in rows]
    within = sum(d < _TOL for d in best)
    print(f"[qwen4_exp sparse e2e max_extend={max_extend}] positions={len(rows)} "
          f"within {_TOL} of a reference: {within}/{len(rows)} max={max(best):.4f} "
          f"mean={sum(best) / len(best):.4f} argmax==fp32 {sum(r[2] for r in rows)}/{len(rows)}")
    assert within >= len(rows) - 2
    assert sum(best) / len(best) < 0.2
    assert max(best) < 3.0  # a flip moves logits by ~1-2; a real bug is far larger
