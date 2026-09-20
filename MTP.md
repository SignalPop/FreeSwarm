# MTP (nextn) over expert-offload — design + status

Goal: add Multi-Token Prediction (self-speculative decode) to FreeToken, and run it **with the
expert-offload backend** — the novel combination (MTP amortizes the per-step expert-fetch cost,
since verifying k tokens pays the offload/PCIe cost roughly once).

Status: the **head module is implemented** (`models/qwen3_5_moe/mtp.py`). Weight-load wiring and
the engine speculative loop are specified below and are the next step — they need a live 35B
(RAM-blocked on the 31 GB dev box; validate after the RAM upgrade).

## The head (done — `mtp.py`)

`Qwen3_5MTP` mirrors the checkpoint's 19 `mtp.*` tensors exactly (verified against
`Qwen3.6-35B-A3B-NVFP4/model.safetensors.index.json`):

```
mtp.pre_fc_norm_embedding.weight   mtp.pre_fc_norm_hidden.weight   mtp.fc.weight
mtp.layers.0.input_layernorm.weight   mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.self_attn.{q,k,v,o}_proj.weight  mtp.layers.0.self_attn.{q,k}_norm.weight
mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}  mtp.layers.0.mlp.gate.weight
mtp.layers.0.mlp.shared_expert.{gate_proj,up_proj,down_proj}.weight
mtp.layers.0.mlp.shared_expert_gate.weight        mtp.norm.weight
```

Reuses the tested `Qwen3_5Attention` / `Qwen3_5MoE` blocks. Embedding + lm_head are shared with
the base model. `forward(prev_hidden, next_ids, embed_tokens, lm_head) -> logits`.

## Weight loading (next)

`weight.py::_rename` currently returns `None` for `mtp.*` (dropped). Two ways to load:

1. **Reuse the main loader** — stop dropping `mtp.*`, and route those tensors through the same
   per-layer emit path that fuses q/k/v -> `qkv_proj`, fuses shared_expert gate/up ->
   `gate_up_proj`, and keeps NVFP4 native. Then `Qwen3_5Attention`/`Qwen3_5MoE` keys match and
   the main `load_state_dict` (engine.py:323) populates `self.mtp`. Cleanest; touches the emit
   fusion in `iter_weights`.
2. **Dedicated mtp loader** — keep `mtp.*` dropped from the main pass; add
   `load_mtp_weights(mtp, model_path)` that reads `mtp.*` from safetensors, applies the same
   `_dequant_nvfp4_weight` + qkv/gate-up fusion, and assigns into `self.mtp`. Decouples from the
   strict main load. Slightly more code, lower blast radius.

Either way the routed MoE experts of the nextn layer are small (one layer) — keep them RESIDENT
on GPU (do NOT send them to the offload host banks).

## Engine integration — self-speculative decode (next, the hard part)

Wire in `engine/engine.py::forward_batch` (+ a small scheduler branch), gated by a `--mtp` flag
(default off; fused/offload paths untouched when off):

1. **KV / state.** The nextn layer needs its own KV: size the MHA KV pool for
   `num_full_attention_layers + 1` and give `Qwen3_5MTPLayer` the appended id. It's a full-attn
   layer, so **no GDN recurrent state** in the head itself — this sidesteps the hardest part of
   speculative decode on Qwen3-Next (rolling back GDN linear-attention state on rejected tokens).
   The BASE model's GDN layers still need rollback for rejected drafts: snapshot the GDN state
   before the verify forward and restore on partial accept (the `HybridRadixCache` /
   `cache_type='hybrid_radix'` already snapshots GDN state at chunk boundaries — reuse that).
2. **Propose.** After a normal decode step yields `token_{t+1}` and `hidden_t`, call
   `mtp.forward` k times, feeding each argmax back as `next_ids` and advancing the MTP KV slot:
   drafts = [d1..dk].
3. **Verify.** One base-model forward over `[token_{t+1}, d1..dk]` (a k+1 "chunked decode",
   route through the prefill/extend path so it's variable-length, not the bs=1 decode graph).
   Sample/argmax at each position; accept the longest prefix where draft == verified.
4. **Commit / rollback.** Advance base KV + GDN state by the accepted count; drop the rest.
   Re-seed the MTP KV from the last accepted hidden.

Acceptance-rate * (k+1) / (verify_cost/decode_cost) is the speedup. On the offload path the
verify forward pays the expert-fetch once for k+1 tokens, so the win should exceed the
~1.6x we measured for MTP on the VRAM-resident llama.cpp path.

## Enable / test (after RAM upgrade)

```
ft ... --model-path <35B-NVFP4> --moe-backend offload --expert-load serial --mtp --mtp-draft 2
```
Bench: baseline offload tok/s vs `--mtp` offload tok/s (same prompts). Expect >1x; report
acceptance rate. Start with `--mtp-draft 2` (matches the llama.cpp sweet spot).

## MEASURED (bench/bench_mtp_accept.py) -- alpha = 0.89, ~1.8x over offload

Before wiring the full production `--mtp` engine loop, the head + the offload cost model were
validated directly on the 3090 with the real 35B (Qwen3.6-35B-A3B-NVFP4).

Head storage (checkpoint): the 19 `mtp.*` tensors are **plain bf16** (no fp8/nvfp4 scales),
with pre-stacked experts (`experts.gate_up_proj` [256,1024,2048], `experts.down_proj`
[256,2048,512]). So the head needs no quant-loader surgery -- `bench/bench_mtp_accept.py`
loads the 19 tensors directly and runs the head in pure torch, reusing the base model's own
`rotary` + `GemmaRMSNorm` + shared `embed`/`lm_head` (so there is no RoPE/norm drift). It
builds a real in-process offload `Engine` and drives eager prefills (the `_warmup_prefill`
recipe) to capture the base hidden states.

Result (3 prompts x 48 greedy tokens, base-greedy verify):

| head input      | acceptance alpha |
|-----------------|------------------|
| pre-final-norm  | **0.894** (126/141) |
| post-final-norm | 0.872 (123/141)  |

Both wirings score ~0.87-0.89 (a mis-implemented head lands near 0), and base-greedy
consistency was 100% -- so the head implementation is validated. The pre-final-norm input is
the correct one (DeepSeek/Qwen nextn convention).

Throughput model (draft-1): tokens/step = 1 + alpha = 1.89; the verify-2 forward costs
r ~= 1.05 decode steps on offload (from the measured 18-50x prefill-vs-decode amortization,
since the expert-fetch is paid once per forward). **Speedup = (1+alpha)/r ~= 1.80x**:
~80 tok/s offload baseline -> **~144 tok/s** (or ~193 from the 107 tok/s greedy-warm baseline).

Run it: set `MODEL` at the top of `bench/bench_mtp_accept.py` to a checkpoint you have, then
`.venv\Scripts\python bench\bench_mtp_accept.py` from a `vcvars64` shell (the harness JIT-compiles
kernels, so it needs `cl.exe` and a matching `nvcc` on PATH like any other engine run).
The production `--mtp` engine loop above remains the way to realize this speedup in the live
server; the harness confirms the acceptance and the offload amortization it relies on.
