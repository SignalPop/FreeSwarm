"""GLM-5.3-Flash (``glm5_next``) vs the transformers reference.

* config parsing against the real ``nvidia/GLM-5.3-Flash-NVFP4`` config.json (and the
  raw-dict path used when transformers predates glm5_next);
* checkpoint key coverage against the real safetensors index (no tensor data needed);
* end-to-end logits of a TINY random-weight model (KDA + MLA/DSA with a shared-indexer
  layer, dense-first MLP layers, MoE with a shared expert, hyper-connections) built with
  ``transformers.models.glm5_next`` and loaded into the FreeToken model through the real
  weight mapping. The FreeToken side runs through its real serving pieces -- DSAKVCache
  (paged latent + k-pool index slab, scrambled rows), LinearStatePool (KDA conv/recurrent
  state), the ``dsa`` backend's metadata/decode staging -- over batched ragged prefill,
  a chunked-prefill continuation, and batched decode steps. The reference is ONE
  full-sequence forward (the model is causal, so position ``p``'s logits there are the
  decode-step logits at ``p``).
"""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import pytest
import torch

L, D = "linear_attention", "deepseek_sparse_attention"
REAL_REPO = "nvidia/GLM-5.3-Flash-NVFP4"


# ---------------------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------------------
def _real_config_path():
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(REAL_REPO, "config.json")
    except Exception as exc:  # offline / no cache
        pytest.skip(f"real GLM-5.3 config.json unavailable: {exc}")


def _check_real_model_config(cfg):
    from freetoken.attention.base import AttnType
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    a = cfg.glm_dsa_args
    assert cfg.num_layers == 45 and cfg.hidden_size == 4096 and cfg.vocab_size == 154880
    assert a.dsa_layer_ids == tuple(range(3, 45, 4))
    assert a.linear_layer_ids == tuple(i for i in range(45) if i % 4 != 3)
    assert a.indexer_layer_ids == a.dsa_layer_ids  # all "full"
    assert (a.num_heads, a.q_lora_rank, a.kv_lora_rank) == (64, 1536, 512)
    assert (a.qk_nope_head_dim, a.qk_rope_head_dim, a.v_head_dim) == (256, 0, 256)
    assert (a.index_n_heads, a.index_head_dim, a.index_topk, a.index_kpool) == (32, 128, 2048, 4)
    assert a.index_kpool_always_select_tail
    assert (a.linear_num_heads, a.linear_head_dim, a.linear_conv_kernel_dim) == (64, 128, 4)
    assert a.linear_lower_bound == -5.0
    assert (a.hc_mult, a.hc_sinkhorn_iters, a.hc_eps) == (4, 20, 1e-6)
    assert a.swiglu_limit == 10.0 and a.max_position == 1048576
    assert cfg.first_k_dense_replace == 3 and cfg.num_moe_layers == 42
    assert (cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size) == (288, 8, 2048)
    assert cfg.n_shared_experts == 1 and cfg.routed_scaling_factor == 2.5
    assert cfg.norm_topk_prob and cfg.intermediate_size == 12288
    assert cfg.expert_quant == "nvfp4" and cfg.hidden_act == "silu_clamp"
    assert cfg.attn_sm_scale == pytest.approx(256**-0.5)
    assert not cfg.tie_word_embeddings and not cfg.linear_state_prefix_cache

    (spec,) = cfg.kv_cache_group_specs()  # the KDA group holds no paged KV
    assert spec.attn_type == AttnType.DSA and spec.mla
    assert spec.layer_ids == a.dsa_layer_ids and spec.head_dim == 512
    assert (spec.index_head_dim, spec.num_index_layers) == (3 * 128, 11)
    lin = cfg.linear_attention_group()
    assert isinstance(lin, LinearGatedDeltaGroupConfig)
    assert (lin.num_key_heads, lin.num_value_heads, lin.key_head_dim, lin.value_head_dim) == (64, 64, 128, 128)
    assert cfg.has_linear_attention


def test_parse_real_config():
    from freetoken.models.glm5_next import parse_config
    from freetoken.utils import cached_load_hf_config

    path = _real_config_path()
    cfg = parse_config(cached_load_hf_config(os.path.dirname(path)))
    _check_real_model_config(cfg)


def test_parse_real_config_raw_shim():
    """Older transformers (no glm5_next): the engine falls back to the raw config.json."""
    from freetoken.models.glm5_next import parse_config
    from freetoken.utils.hf import RawConfigShim

    with open(_real_config_path(), encoding="utf-8") as f:
        raw = json.load(f)
    _check_real_model_config(parse_config(RawConfigShim(raw)))


def test_engine_resolution():
    from freetoken.engine.engine import _required_attn_types, _resolve_auto_attention_backend
    from freetoken.kvcache import resolve_pool_class
    from freetoken.kvcache.dsa_pool import DSAKVCache
    from freetoken.models.glm5_next import parse_config
    from freetoken.models.register import get_model_spec
    from freetoken.utils.hf import RawConfigShim

    with open(_real_config_path(), encoding="utf-8") as f:
        cfg = parse_config(RawConfigShim(json.load(f)))
    assert resolve_pool_class(cfg) is DSAKVCache
    required = _required_attn_types(cfg)
    assert _resolve_auto_attention_backend(required, True) == "dsa"
    spec = get_model_spec("Glm5NextForConditionalGeneration")
    assert spec.module == "freetoken.models.glm5_next"


def test_aot_table_claims_the_architecture():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS

    entry = next(m for m in SUPPORTED_MODELS if m.architecture == "Glm5NextForConditionalGeneration")
    assert entry.name == REAL_REPO and entry.hidden_size == 4096
    assert entry.kv_groups == ()  # MLA latent + index slab via torch scatter
    assert (entry.top_k, entry.moe_intermediate_size) == (8, 2048)
    assert entry.expert_formats == ("nvfp4",)  # silu_clamp -> Triton NVFP4 only


def test_engine_forces_naive_cache_for_kda():
    from freetoken.engine.engine import _adjust_config
    from freetoken.models.glm5_next import parse_config
    from freetoken.utils.hf import RawConfigShim

    with open(_real_config_path(), encoding="utf-8") as f:
        model_config = parse_config(RawConfigShim(json.load(f)))

    class _Cfg(SimpleNamespace):
        pass

    config = _Cfg(
        model_config=model_config, cache_type="radix", cuda_graph_max_bs=None,
        cuda_graph_bs=None, max_running_req=4, attention_backend="auto", page_size=1,
        moe_backend="offload", moe_cache_rate=None, moe_cpu_layers=None, dtype=torch.bfloat16,
        num_token_override=None, num_page_override=None, max_seq_len_override=None,
        moe_cache_size=0, moe_cache_auto=False, moe_cpu_threads=0, moe_hybrid_max_fetch=-1,
        moe_prefill_overlap=True, moe_prefill_hit_d2d=False, expert_load="auto",
        nvfp4_backend="triton",
    )
    try:
        _adjust_config(config)
    except Exception:
        pass  # later, unrelated knobs of the duck-typed config may not resolve here
    assert config.cache_type == "naive"


def test_auto_parsers_are_glm(monkeypatch):
    """--tool-call-parser / --reasoning-parser auto map GLM-5.3 like the other GLMs."""
    from freetoken.server.args import parse_args

    with open(_real_config_path(), encoding="utf-8") as f:
        raw = json.load(f)

    class _C:
        def to_dict(self):
            return raw

    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", lambda _p: _C())
    args, _ = parse_args(["--model", "/models/anon"])
    assert (args.tool_call_parser, args.reasoning_parser) == ("glm47", "glm")


def test_glm53_tool_call_format_parses():
    """The GLM-5.3 chat template emits ``<tool_call>{name}<arg_key>..`` with NO newline
    after the name (template: ``'<tool_call>' + tc.name`` then the arg pairs)."""
    from freetoken.server.function_call_parser import FunctionCallParser

    tools = [{"type": "function", "function": {
        "name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}}}}}]
    parser = FunctionCallParser(tools, tool_call_parser="glm47")
    text = (
        "Let me check.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris"
        "</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
    )
    res = parser.parse_non_stream(text)
    assert res.normal_text.strip() == "Let me check."
    assert len(res.calls) == 1 and res.calls[0].name == "get_weather"
    assert json.loads(res.calls[0].parameters) == {"city": "Paris", "days": 3}


def test_checkpoint_key_coverage_against_real_index(monkeypatch):
    """Every tensor the loader requests exists in the real checkpoint, and it yields
    exactly the model's (offload-MoE) state-dict keys."""
    import dataclasses

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.glm5_next import parse_config
    from freetoken.models.glm5_next import weight as w
    from freetoken.models.glm5_next.model import Glm5NextForCausalLM
    from freetoken.utils.hf import RawConfigShim

    try:
        from huggingface_hub import hf_hub_download

        index = hf_hub_download(REAL_REPO, "model.safetensors.index.json")
    except Exception as exc:
        pytest.skip(f"real index unavailable: {exc}")
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    with open(_real_config_path(), encoding="utf-8") as f:
        cfg = parse_config(RawConfigShim(json.load(f)))
    with open(index, encoding="utf-8") as f:
        names = set(json.load(f)["weight_map"])

    requested = []

    def get(name):
        assert name in names, name
        requested.append(name)
        return torch.empty(0)

    monkeypatch.setattr(w, "_dequant_nvfp4", lambda *a: torch.empty(0))
    keys = [k for k, _ in w.iter_glm5_next_weights(names.__contains__, get, cfg, include_moe_experts=False)]
    assert len(keys) == len(set(keys))
    with torch.device("meta"):
        model = Glm5NextForCausalLM(dataclasses.replace(cfg, moe_backend="offload"))
    assert set(keys) == set(model.state_dict())
    # nothing from the vision tower or the MTP layer is touched
    assert not any(".visual." in n or ".layers.45." in n for n in requested)
    # the dense MLPs of layers 0-2 are the NVFP4 tensors dequantized at load
    assert "model.language_model.layers.0.mlp.gate_proj.weight_scale_2" in names


# ---------------------------------------------------------------------------------------
# tiny random model: FreeToken vs transformers
# ---------------------------------------------------------------------------------------
TINY_LAYERS = [L, D, D, L, L, D]  # layer 2 = shared-indexer DSA layer after full layer 1
TINY_INDEXER = ["full", "full", "shared", "full", "full", "full"]
TINY_MLP = ["dense", "dense", "sparse", "sparse", "sparse", "sparse"]


def _tiny_hf_config(index_topk: int = 8, top_k: int = 2, **overrides):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

    text = dict(
        vocab_size=128, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
        num_hidden_layers=len(TINY_LAYERS), num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=top_k,
        routed_scaling_factor=2.5, kv_lora_rank=32, q_lora_rank=48, qk_rope_head_dim=0,
        qk_nope_head_dim=16, v_head_dim=16, n_group=1, topk_group=1,
        index_topk=index_topk, index_head_dim=16, index_n_heads=4, index_kpool=4,
        index_kpool_always_select_tail=True, layer_types=TINY_LAYERS,
        indexer_types=TINY_INDEXER, mlp_layer_types=TINY_MLP, linear_head_dim=16,
        linear_num_heads=4, linear_conv_kernel_dim=4, linear_lower_bound=-5.0, hc_mult=4,
        hc_sinkhorn_iters=20, max_position_embeddings=4096, rms_norm_eps=1e-5,
        swiglu_limit=1.0, pad_token_id=None, eos_token_id=None, tie_word_embeddings=False,
    )
    text.update(overrides)
    return Glm5NextConfig(text_config=text, architectures=["Glm5NextForConditionalGeneration"])


def _tiny_hf_model(hf_config, device, seed=0, gain=1.5):
    """Reference text model + lm_head (fp32), with every parameter randomized so the
    clamps, gates, decay, router bias, k-pool gate and hyper-connections are all active."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    text = hf_config.text_config
    text._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = Glm5NextTextModel(text).to(device=device, dtype=torch.float32).eval()
    lm_head = torch.nn.Linear(text.hidden_size, text.vocab_size, bias=False).to(device)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + [("lm_head.weight", lm_head.weight)]:
            if name.endswith(("layernorm.weight", "norm.weight")) and "k_norm" not in name:
                val = 1.0 + 0.2 * torch.randn(p.shape, generator=g)
            elif name.endswith("A_log"):
                val = 0.5 * torch.randn(p.shape, generator=g)
            elif name.endswith("hc.scale"):
                val = 0.5 + torch.rand(p.shape, generator=g)
            elif p.dim() >= 2:
                fan_in = p.shape[-1]
                val = torch.randn(p.shape, generator=g) * (gain / math.sqrt(fan_in))
            else:
                val = 0.3 * torch.randn(p.shape, generator=g)
            p.copy_(val.to(p.device, p.dtype))
        for layer in model.layers:
            gate = getattr(layer.mlp, "gate", None)
            if gate is not None:
                gate.e_score_correction_bias.copy_(0.1 * torch.randn(gate.num_experts, generator=g))
    return model, lm_head


def _hf_logits(model, lm_head, ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        x = torch.tensor([ids], device=lm_head.weight.device)
        h = model(input_ids=x).last_hidden_state
        return lm_head(h)[0].float()


def _hf_internal_state(model, lm_head) -> dict:
    sd = {k: v for k, v in model.state_dict().items()}
    sd["lm_head.weight"] = lm_head.weight
    return sd


def _to_checkpoint_layout(sd: dict, num_experts: int) -> dict:
    """transformers-internal names -> the released checkpoint's names/layout (the
    inverse of the glm5_next conversion mapping), under model.language_model."""
    out = {}
    for k, v in sd.items():
        if k == "lm_head.weight":
            out[k] = v
            continue
        k2 = k.replace("self_attn.forget_gate.", "self_attn.")
        for site in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                k2 = k2.replace(f"{site}_hc.{part}", f"hc_{site}_{part}")
        p = "model.language_model."
        if k2.endswith("self_attn.conv1d.weight"):
            base = k2[: -len("conv1d.weight")]
            for name, part in zip(("q", "k", "v"), v.chunk(3, dim=0)):
                out[p + base + f"{name}_conv1d.weight"] = part
        elif k2.endswith("mlp.experts.gate_up_proj"):
            base = k2[: -len("gate_up_proj")]
            for e in range(num_experts):
                gate, up = v[e].chunk(2, dim=0)
                out[p + base + f"{e}.gate_proj.weight"] = gate
                out[p + base + f"{e}.up_proj.weight"] = up
        elif k2.endswith("mlp.experts.down_proj"):
            base = k2[: -len("down_proj")]
            for e in range(num_experts):
                out[p + base + f"{e}.down_proj.weight"] = v[e]
        else:
            out[p + k2] = v
    return out


class _Harness:
    """FreeToken model + the real pools/backend, driven with hand-built batches."""

    def __init__(self, hf_config, state: dict, dtype: torch.dtype, *, max_reqs=4, max_len=256):
        from freetoken.attention.dsa import DSAAttnBackend
        import freetoken.core as core
        from freetoken.core import Context, set_global_ctx
        from freetoken.distributed import set_tp_info, try_get_tp_info
        from freetoken.kvcache.dsa_pool import DSAKVCache
        from freetoken.kvcache.linear_state_pool import LinearStatePool
        from freetoken.models.glm5_next import parse_config
        from freetoken.models.glm5_next.model import Glm5NextForCausalLM
        from freetoken.models.glm5_next.weight import iter_glm5_next_weights
        from freetoken.utils import torch_dtype

        if try_get_tp_info() is None:
            set_tp_info(rank=0, size=1)
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.cfg = parse_config(hf_config)
        args = self.cfg.glm_dsa_args
        with torch.device("meta"), torch_dtype(dtype):
            model = Glm5NextForCausalLM(self.cfg)
        expected = model.state_dict()
        loaded = {}
        for k, v in iter_glm5_next_weights(
            state.__contains__, state.__getitem__, self.cfg, include_moe_experts=True
        ):
            assert k in expected, k
            loaded[k] = v.detach().to(self.device, expected[k].dtype).contiguous()
        model.load_state_dict(loaded)
        self.model = model

        core._GLOBAL_CTX = None  # test-only: fresh ctx per harness
        ctx = Context(page_size=1)
        self.num_rows = max_reqs * max_len
        ctx.page_table = torch.zeros(max_reqs + 1, max_len, dtype=torch.int32, device=self.device)
        spec = self.cfg.kv_cache_group_specs()[0]
        ctx.kv_cache = DSAKVCache(
            latent_dim=spec.head_dim, num_layers=self.cfg.num_layers, num_pages=self.num_rows + 1,
            page_size=1, dtype=dtype, device=self.device, index_head_dim=spec.index_head_dim,
            num_index_layers=spec.num_index_layers, layer_ids=spec.layer_ids, index_dtype=dtype,
        )
        ctx.linear_state_pool = LinearStatePool(
            group=self.cfg.linear_attention_group(), num_slots=max_reqs + 1, dtype=dtype,
            device=self.device, tp_size=1,
        )
        set_global_ctx(ctx)
        ctx.attn_backend = DSAAttnBackend(self.cfg)
        self.ctx = ctx
        # scrambled physical rows per request (exercises the page-table indirection)
        gen = torch.Generator().manual_seed(1234)
        perm = torch.randperm(self.num_rows, generator=gen).to(torch.int32)
        for t in range(max_reqs):
            ctx.page_table[t] = perm[t * max_len : (t + 1) * max_len].to(self.device)
        assert args.latent_dim == spec.head_dim

    def forward(self, reqs_spec, phase: str) -> list[torch.Tensor]:
        """``reqs_spec``: list of (table_idx, ids, cached_len, device_len). Returns the
        per-request logits [extend_len, V] (fp32)."""
        from freetoken.core import Batch, Req, SamplingParams
        import torch.nn.functional as F

        reqs = []
        for table_idx, ids, cached, dev_len in reqs_spec:
            req = Req(
                input_ids=torch.tensor(ids[:dev_len], dtype=torch.int32),
                table_idx=table_idx, cached_len=cached, output_len=4, uid=table_idx,
                sampling_params=SamplingParams(), cache_handle=None,
            )
            reqs.append(req)
        batch = Batch(reqs=reqs, phase=phase)
        batch.padded_reqs = reqs
        ids_all, pos_all, loc_all = [], [], []
        for r in reqs:
            ids_all.append(r.input_ids[r.cached_len : r.device_len])
            pos = torch.arange(r.cached_len, r.device_len, dtype=torch.int64)
            pos_all.append(pos)
            loc_all.append(self.ctx.page_table[r.table_idx, pos.to(self.device)])
        batch.input_ids = torch.cat(ids_all).to(self.device)
        batch.positions = torch.cat(pos_all).to(torch.int32).to(self.device)
        batch.out_loc = torch.cat(loc_all)
        if phase == "decode":
            tids = torch.tensor([r.table_idx for r in reqs], device=self.device)
            batch.active_table_idx = tids
            batch.linear_table_idx = tids.to(torch.int32)
        with torch.no_grad(), self.ctx.forward_batch(batch):
            self.ctx.attn_backend.prepare_metadata(batch)
            h = self.model.model.forward(batch.input_ids)
            logits = F.linear(h, self.model.lm_head.weight).float()
        out, off = [], 0
        for r in reqs:
            out.append(logits[off : off + r.extend_len])
            off += r.extend_len
        return out


def _torch_routed_forward(self, hidden_states, topk_weights, topk_ids):
    """fp32 reference for the resident expert GEMMs (the fused Triton MoE is bf16/fp16)."""
    out = torch.zeros_like(hidden_states, dtype=torch.float32)
    limit = self.swiglu_limit
    for t in range(hidden_states.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            gu = self.gate_up_proj[e].float() @ hidden_states[t].float()
            gate, up = gu.chunk(2)
            act = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
            out[t] += topk_weights[t, j] * (self.down_proj[e].float() @ act)
    return out.to(hidden_states.dtype)


def _run_scenario(h: _Harness, seqs: dict[int, list[int]], prompt_lens: dict[int, int],
                  chunk_split: dict[int, int], n_decode: int) -> dict[int, list[torch.Tensor]]:
    """Ragged batched prefill (with an optional first chunk per request), then batched
    teacher-forced decode steps. Returns per-request logits rows in position order."""
    got = {t: [] for t in seqs}
    # first chunk for the requests that are split
    first = [(t, seqs[t], 0, chunk_split[t]) for t in seqs if t in chunk_split]
    if first:
        for (t, *_), lg in zip(first, h.forward(first, "prefill")):
            got[t].append(lg)
    rest = [
        (t, seqs[t], chunk_split.get(t, 0), prompt_lens[t]) for t in seqs
    ]
    for (t, *_), lg in zip(rest, h.forward(rest, "prefill")):
        got[t].append(lg)
    for step in range(n_decode):
        spec = [(t, seqs[t], prompt_lens[t] + step, prompt_lens[t] + step + 1) for t in seqs]
        for (t, *_), lg in zip(spec, h.forward(spec, "decode")):
            got[t].append(lg)
    return {t: torch.cat(v) for t, v in got.items()}


pytestmark_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _plain_embedding(monkeypatch):
    """The tiny hidden size has no prebuilt ``indexing`` kernel (its JIT needs a matching
    nvcc); the embedding lookup is not what these tests exercise."""
    from freetoken.layers.embedding import VocabParallelEmbedding

    monkeypatch.setattr(
        VocabParallelEmbedding, "forward",
        lambda self, x: torch.nn.functional.embedding(x.long(), self.weight),
    )


def _scenario_inputs(vocab: int):
    g = torch.Generator().manual_seed(99)
    n_decode = 4
    prompt_lens = {1: 37, 2: 9, 3: 22}
    seqs = {
        t: torch.randint(0, vocab, (n + n_decode,), generator=g).tolist()
        for t, n in prompt_lens.items()
    }
    chunk_split = {3: 13}  # request 3 prefills 13 tokens, then continues with 9 more
    return seqs, prompt_lens, chunk_split, n_decode


@pytestmark_cuda
@pytest.mark.parametrize("layout", ["hf_internal", "checkpoint"])
def test_tiny_model_matches_reference_fp32(monkeypatch, layout):
    """fp32 end to end (routed-expert GEMMs via a torch reference; everything else is the
    serving code): prefill + chunked prefill + decode logits match to ~fp32 precision,
    in both the sparse (k-pool top-k) and dense DSA regimes."""
    from freetoken.layers.moe import MoELayer

    monkeypatch.setattr(MoELayer, "routed_forward", _torch_routed_forward)
    hf_config = _tiny_hf_config(index_topk=8)
    ref, lm_head = _tiny_hf_model(hf_config, "cuda")
    state = _hf_internal_state(ref, lm_head)
    if layout == "checkpoint":
        state = _to_checkpoint_layout(state, hf_config.text_config.n_routed_experts)
    h = _Harness(hf_config, state, torch.float32)
    seqs, prompt_lens, chunk_split, n_decode = _scenario_inputs(hf_config.text_config.vocab_size)
    got = _run_scenario(h, seqs, prompt_lens, chunk_split, n_decode)
    worst = 0.0
    for t, ids in seqs.items():
        want = _hf_logits(ref, lm_head, ids[: prompt_lens[t] + n_decode])
        # the decode step at position p consumed ids[p]; rows line up position-wise
        diff = (got[t] - want).abs().max().item()
        scale = want.abs().max().item()
        worst = max(worst, diff / scale)
        print(f"[fp32 {layout}] req {t}: max|diff|={diff:.3e} (max|logit|={scale:.2f})")
    assert worst < 1e-4, worst


@pytestmark_cuda
def test_tiny_model_bf16_serving_path():
    """bf16 serving path (real Triton fused MoE with the silu_clamp activation) vs the
    fp32 reference, judged against transformers' OWN bf16 error on the same weights.
    Dense DSA regime (index_topk >= every length) so bf16 noise cannot flip a top-k.

    A random tiny model is chaotic in bf16 (router flips etc.: transformers' own bf16 run
    reaches cos ~0.75 at its worst position), so the check is statistical: our error must
    be at the reference's own bf16 noise level, not below a fixed threshold."""
    # every expert active (no bf16 routing flips) + a gentler init: a smoother model makes
    # the noise-level comparison sharper
    hf_config = _tiny_hf_config(index_topk=64, top_k=8)
    ref, lm_head = _tiny_hf_model(hf_config, "cuda", gain=1.0)
    state = _hf_internal_state(ref, lm_head)
    h = _Harness(hf_config, state, torch.bfloat16)
    seqs, prompt_lens, chunk_split, n_decode = _scenario_inputs(hf_config.text_config.vocab_size)
    got = _run_scenario(h, seqs, prompt_lens, chunk_split, n_decode)

    import copy

    ref_bf16 = copy.deepcopy(ref).to(torch.bfloat16)
    lm_bf16 = copy.deepcopy(lm_head).to(torch.bfloat16)
    ours_err, hf_err, ours_cos, hf_cos = [], [], [], []
    cos_fn = torch.nn.functional.cosine_similarity
    for t, ids in seqs.items():
        full = ids[: prompt_lens[t] + n_decode]
        want = _hf_logits(ref, lm_head, full)
        hf_bf16 = _hf_logits(ref_bf16, lm_bf16, full)
        ours_err.append((got[t] - want).abs().mean(-1))
        hf_err.append((hf_bf16 - want).abs().mean(-1))
        ours_cos.append(cos_fn(got[t], want, dim=-1))
        hf_cos.append(cos_fn(hf_bf16, want, dim=-1))
    ours_err, hf_err = torch.cat(ours_err).mean().item(), torch.cat(hf_err).mean().item()
    ours_cos, hf_cos = torch.cat(ours_cos).mean().item(), torch.cat(hf_cos).mean().item()
    print(f"[bf16] mean|diff| ours={ours_err:.4f} hf_bf16={hf_err:.4f}; "
          f"mean cos ours={ours_cos:.4f} hf_bf16={hf_cos:.4f}")
    assert ours_err < 1.5 * hf_err + 1e-2, (ours_err, hf_err)
    assert ours_cos > hf_cos - 0.005, (ours_cos, hf_cos)


_ENGINE_SMOKE = r'''
import json, os, sys
import torch
sys.path.insert(0, {tests_dir!r})
import test_glm5_next as T
from freetoken.layers.embedding import VocabParallelEmbedding
# tiny hidden size: no prebuilt `indexing` kernel (see the _plain_embedding fixture)
VocabParallelEmbedding.forward = lambda self, x: torch.nn.functional.embedding(x.long(), self.weight)
from freetoken.core import SamplingParams
from freetoken.llm import LLM

ckpt, graphs = sys.argv[1], sys.argv[2]
prompts = json.load(open(os.path.join(ckpt, "prompts.json")))
kw = dict(max_running_req=4, moe_backend="fused", max_seq_len_override=256,
          memory_ratio=0.5, num_page_override=2048)
if graphs == "off":
    kw["cuda_graph_max_bs"] = 0
llm = LLM(ckpt, dtype=torch.bfloat16, **kw)
assert llm.engine.config.attention_backend == "dsa"
res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
print("RESULT " + json.dumps({{"tokens": [r["token_ids"] for r in res],
                               "cache_type": llm.engine.config.cache_type}}))
'''


@pytestmark_cuda
@pytest.mark.slow
def test_engine_serves_tiny_checkpoint(tmp_path):
    """Real Engine + Scheduler on a tiny on-disk checkpoint in the RELEASED layout (split
    q/k/v convs, per-expert tensors, hc_* names, model.language_model. prefix): weight
    loading, DSAKVCache + LinearStatePool sizing, the dsa backend, forced naive cache,
    batched ragged prefill and decode, CUDA-graph capture. Checks graphs on == off and
    that greedy decoding tracks the fp32 reference (same bf16-rounded weights)."""
    import subprocess
    import sys

    import safetensors.torch

    try:
        from huggingface_hub import hf_hub_download

        tok = hf_hub_download(REAL_REPO, "tokenizer.json")
        tok_cfg = hf_hub_download(REAL_REPO, "tokenizer_config.json")
    except Exception as exc:
        pytest.skip(f"GLM-5.3 tokenizer unavailable: {exc}")
    import shutil

    hf_config = _tiny_hf_config(index_topk=8, top_k=8)
    ref, lm_head = _tiny_hf_model(hf_config, "cpu", gain=1.0)
    with torch.no_grad():  # the served checkpoint is bf16: make the reference identical
        for p in list(ref.parameters()) + list(lm_head.parameters()):
            p.copy_(p.to(torch.bfloat16).float())
    state = _to_checkpoint_layout(_hf_internal_state(ref, lm_head), hf_config.text_config.n_routed_experts)
    safetensors.torch.save_file(
        {k: v.detach().to(torch.bfloat16).contiguous() for k, v in state.items()},
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text(json.dumps(hf_config.to_dict()))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [127]}))
    shutil.copy(tok, tmp_path / "tokenizer.json")
    shutil.copy(tok_cfg, tmp_path / "tokenizer_config.json")
    g = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, 127, (n,), generator=g).tolist() for n in (37, 9, 22)]
    (tmp_path / "prompts.json").write_text(json.dumps(prompts))
    script = tmp_path / "smoke.py"
    script.write_text(_ENGINE_SMOKE.format(tests_dir=os.path.dirname(os.path.abspath(__file__))))

    results = {}
    for graphs in ("on", "off"):
        proc = subprocess.run(
            [sys.executable, str(script), str(tmp_path), graphs],
            capture_output=True, text=True, timeout=900, env=dict(os.environ),
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        assert line is not None, proc.stdout[-3000:] + proc.stderr[-6000:]
        results[graphs] = json.loads(line[len("RESULT "):])
    assert results["on"]["cache_type"] == "naive"
    assert results["on"]["tokens"] == results["off"]["tokens"]  # CUDA graphs == eager

    engine = results["on"]["tokens"]
    n_new = min(len(o) for o in engine)
    assert n_new >= 7

    # The same greedy loop through the in-process eager harness (validated against the
    # reference to fp32 precision above): the scheduler/engine path must reproduce it.
    h = _Harness(hf_config, {k: v.to(torch.bfloat16) for k, v in _hf_internal_state(ref, lm_head).items()},
                 torch.bfloat16)
    seqs = {i + 1: list(p) for i, p in enumerate(prompts)}
    eager = {t: [] for t in seqs}
    for step in range(n_new):
        if step == 0:
            spec = [(t, seqs[t], 0, len(seqs[t])) for t in seqs]
        else:
            spec = [(t, seqs[t], len(seqs[t]) - 1, len(seqs[t])) for t in seqs]
        for (t, *_), lg in zip(spec, h.forward(spec, "prefill" if step == 0 else "decode")):
            nxt = int(lg[-1].argmax())
            eager[t].append(nxt)
            seqs[t].append(nxt)
    eager_tokens = [eager[t] for t in sorted(eager)]
    print("[engine] tokens", json.dumps(engine), "eager", json.dumps(eager_tokens))
    assert [o[0] for o in engine] == [o[0] for o in eager_tokens]  # prefill logits
    same = sum(int(a == b) for e, o in zip(engine, eager_tokens) for a, b in zip(e[:n_new], o))
    assert same >= 0.9 * n_new * len(prompts), (engine, eager_tokens)

    agree = total = 0
    for p, out in zip(prompts, engine):
        want = _hf_logits(ref, lm_head, p + out)
        am = want.argmax(-1)[len(p) - 1 : len(p) - 1 + len(out)].tolist()
        agree += sum(int(a == b) for a, b in zip(am, out))
        total += len(out)
    print(f"[engine] teacher-forced greedy agreement with the fp32 reference: {agree}/{total}")
    # bf16 serving of a random tiny model flips near-tie argmaxes (the reference's own
    # bf16 run does too); this is a sanity floor, the exactness check is the eager match.
    assert agree >= 0.5 * total, (agree, total)


@pytestmark_cuda
def test_silu_clamp_activation_kernel():
    from freetoken.layers import silu_clamp_and_mul

    torch.manual_seed(0)
    x = torch.randn(37, 2 * 96, device="cuda", dtype=torch.bfloat16) * 4
    out = silu_clamp_and_mul(x, limit=2.5)
    gate, up = x.float().chunk(2, dim=-1)
    ref = torch.nn.functional.silu(gate.clamp(max=2.5)) * up.clamp(-2.5, 2.5)
    assert (out.float() - ref).abs().max().item() < 2e-2


@pytestmark_cuda
def test_kda_chunk_matches_recurrence():
    """The chunked prefill algorithm == token-by-token recurrence (incl. an initial state
    and a non-multiple-of-chunk length)."""
    from freetoken.models.glm5_next.kda import kda_chunk_prefill, kda_recurrent_step

    torch.manual_seed(0)
    T, H, K = 150, 3, 16
    q, k, v = (torch.randn(T, H, K, device="cuda") for _ in range(3))
    g = -5 * torch.sigmoid(torch.randn(T, H, K, device="cuda"))
    beta = torch.sigmoid(torch.randn(T, H, device="cuda"))
    s0 = torch.randn(H, K, K, device="cuda") * 0.1
    out, s_final = kda_chunk_prefill(q, k, v, g, beta, s0, chunk_size=64)
    s = s0.unsqueeze(0)
    ref = []
    for t in range(T):
        o, s = kda_recurrent_step(q[t : t + 1], k[t : t + 1], v[t : t + 1], g[t : t + 1], beta[t : t + 1], s)
        ref.append(o)
    ref = torch.cat(ref)
    assert (out - ref).abs().max().item() < 1e-4
    assert (s_final - s[0]).abs().max().item() < 1e-4
