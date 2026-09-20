"""In-process MTP (nextn) acceptance-rate measurement for FreeToken on the offload 35B.

Builds a real offload Engine (same flags as serve_offload.bat), loads the checkpoint's 19
bf16 ``mtp.*`` head tensors directly, and measures the self-speculative acceptance rate alpha:

  at each generated position i the head drafts token i+2 from (base_hidden_i, embed(S[i+1]));
  accept iff draft == S[i+2] (the base model's own greedy token i+2).

alpha is backend-independent (offload only changes latency, not logits), so it plugs straight
into the measured offload cost model: MTP tok/s ~= baseline x (1 + accepted_per_step) / r,
with r~=1 verified separately (verify(k+1) ~= decode(1) on offload).

The head runs in pure torch, reusing the base model's own rotary + GemmaRMSNorm semantics so
there is no RoPE/norm convention drift. The base-vs-head hidden ambiguity (pre- vs post-final
-norm input to the head) is resolved empirically: alpha is computed BOTH ways; the correct one
is the high number (a mismatched wiring lands near chance).
"""
from __future__ import annotations
import json, os, sys, time
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Point this at a checkpoint you have before running: the harness loads it in-process.
MODEL = r"models\Qwen3.6-35B-A3B"

# ---- build the offload engine in-process (mirror serve_offload.bat flags) ---------------
from freetoken.server.args import parse_args
from freetoken.engine import Engine
from freetoken.core import Batch, Req
from freetoken.layers import GemmaRMSNorm
from freetoken.utils.hf import load_tokenizer
from freetoken.kernel.triton.nvfp4_linear import nvfp4_dense_linear_t, Nvfp4LMHead

ARGV = [
    "--model-path", MODEL,
    "--moe-backend", "offload", "--expert-load", "serial",
    "--attention-backend", "triton",
    "--num-pages", "8192", "--max-seq-len-override", "8192",
    "--host", "127.0.0.1", "--port", "2600",  # distributed init uses port+1 (2601); fresh ports
]
server_args, _ = parse_args(ARGV, False, prog="bench")
print("[bench] building offload engine (pins ~19GB experts, ~1-2 min)...", flush=True)
t0 = time.time()
engine = Engine(server_args)
dev = engine.device
print(f"[bench] engine ready in {time.time()-t0:.0f}s on {dev}", flush=True)

model = engine.model            # Qwen3_5MoEForCausalLM
base = model.model              # Qwen3_5Model
embed = base.embed_tokens
lm_head = model.lm_head
cfg = base.norm  # placeholder; real config numbers hardcoded from config.json below

# ---- config numbers (from the checkpoint's config.json) ----------------------------------
H = 2048; NQ = 16; NKV = 2; HD = 256; EPS = 1e-6
NEXP = 256; TOPK = 8; MOE_I = 512
tok = load_tokenizer(MODEL)

# reuse a base full-attention layer's rotary (positional only, no weights)
rotary = None
for layer in base.layers.op_list:
    if hasattr(layer, "self_attn"):
        rotary = layer.self_attn.rotary; break
assert rotary is not None, "no full-attention layer found for rotary reuse"

# ---- load the 19 bf16 mtp.* tensors directly -------------------------------------------
from safetensors import safe_open
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
def load(name):
    with safe_open(os.path.join(MODEL, idx[name]), framework="pt", device=str(dev)) as f:
        return f.get_tensor(name)
W = {k: load(k) for k in idx if k.startswith("mtp.")}
print(f"[bench] loaded {len(W)} mtp tensors", flush=True)

def lmhead_all(hidden):
    """All-position logits, bypassing Nvfp4LMHead.forward's batch-context/last-token slice.
    The head reads get_global_ctx().batch only to keep the last token per sequence at prefill;
    we want every position and run outside a forward_batch, so call the W4A16 GEMM directly."""
    if isinstance(lm_head, Nvfp4LMHead):
        return nvfp4_dense_linear_t(hidden, lm_head.weight, lm_head.weight_scale, lm_head.weight_global)
    return lm_head.forward(hidden)

def gnorm(dim, wname):
    m = GemmaRMSNorm(dim, eps=EPS)
    m.weight.data = (W[wname].float() + 1.0).to(W[wname].dtype).to(dev)  # (1+w) Gemma
    return m
n_emb = gnorm(H, "mtp.pre_fc_norm_embedding.weight")
n_hid = gnorm(H, "mtp.pre_fc_norm_hidden.weight")
n_in  = gnorm(H, "mtp.layers.0.input_layernorm.weight")
n_post= gnorm(H, "mtp.layers.0.post_attention_layernorm.weight")
n_out = gnorm(H, "mtp.norm.weight")
qn    = gnorm(HD, "mtp.layers.0.self_attn.q_norm.weight")
kn    = gnorm(HD, "mtp.layers.0.self_attn.k_norm.weight")

fc_w   = W["mtp.fc.weight"]                                   # [H, 2H]
qw     = W["mtp.layers.0.self_attn.q_proj.weight"]           # [NQ*HD*2, H]
kw     = W["mtp.layers.0.self_attn.k_proj.weight"]           # [NKV*HD, H]
vw     = W["mtp.layers.0.self_attn.v_proj.weight"]           # [NKV*HD, H]
ow     = W["mtp.layers.0.self_attn.o_proj.weight"]           # [H, NQ*HD]
gate_w = W["mtp.layers.0.mlp.gate.weight"]                   # [NEXP, H]
e_gu   = W["mtp.layers.0.mlp.experts.gate_up_proj"]          # [NEXP, 2*MOE_I, H]
e_dn   = W["mtp.layers.0.mlp.experts.down_proj"]             # [NEXP, H, MOE_I]
sh_g   = W["mtp.layers.0.mlp.shared_expert.gate_proj.weight"]# [MOE_I, H]
sh_u   = W["mtp.layers.0.mlp.shared_expert.up_proj.weight"]  # [MOE_I, H]
sh_d   = W["mtp.layers.0.mlp.shared_expert.down_proj.weight"]# [H, MOE_I]
sh_gate= W["mtp.layers.0.mlp.shared_expert_gate.weight"]     # [1, H]

def head_attn(x):
    """causal gated GQA attention over the whole [L,H] sequence, reusing base rotary."""
    L = x.shape[0]
    qg = (x @ qw.T).view(L, NQ, HD * 2)
    q = qg[..., :HD].contiguous()                     # [L,NQ,HD]
    gate = qg[..., HD:].reshape(L, NQ * HD)           # [L,NQ*HD]
    k = (x @ kw.T).view(L, NKV, HD).contiguous()
    v = (x @ vw.T).view(L, NKV, HD).contiguous()
    q = qn.forward(q); k = kn.forward(k)              # per-head Gemma RMSNorm
    positions = torch.arange(L, dtype=torch.int32, device=dev)
    qf, kf = rotary.forward(positions, q.reshape(L, NQ * HD), k.reshape(L, NKV * HD))
    q = qf.view(L, NQ, HD); k = kf.view(L, NKV, HD)
    # GQA expand kv 2->16 (consecutive q heads share a kv head)
    k = k.repeat_interleave(NQ // NKV, dim=1)
    v = v.repeat_interleave(NQ // NKV, dim=1)
    # SDPA expects [.., heads, seq, dim]
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), is_causal=True)   # [1,NQ,L,HD]
    o = o.squeeze(0).transpose(0, 1).reshape(L, NQ * HD)
    o = o * torch.sigmoid(gate)
    return o @ ow.T

def head_moe(x):
    L = x.shape[0]
    logits = x @ gate_w.T                              # [L,NEXP]
    probs = torch.softmax(logits.float(), dim=-1)
    tv, ti = torch.topk(probs, TOPK, dim=-1)           # [L,TOPK]
    tv = (tv / tv.sum(-1, keepdim=True)).to(x.dtype)
    out = torch.zeros(L, H, dtype=x.dtype, device=dev)
    # gather selected experts per token (L*TOPK small); loop over the TOPK slots
    for s in range(TOPK):
        e = ti[:, s]                                   # [L] expert id per token
        gu = torch.einsum("lh,lih->li", x, e_gu[e])    # [L,2*MOE_I]
        a, b = gu[:, :MOE_I], gu[:, MOE_I:]
        act = F.silu(a) * b
        contrib = torch.einsum("li,lhi->lh", act, e_dn[e])  # [L,H]
        out += tv[:, s : s + 1] * contrib
    # shared expert (gated)
    sa = F.silu(x @ sh_g.T) * (x @ sh_u.T)
    shared = sa @ sh_d.T
    shared = shared * torch.sigmoid(x @ sh_gate.T)
    return out + shared

def head_forward(prev_hidden, next_ids):
    e = n_emb.forward(embed.forward(next_ids))
    h = n_hid.forward(prev_hidden)
    x = torch.cat([e, h], dim=-1) @ fc_w.T
    x = x + head_attn(n_in.forward(x))
    x = x + head_moe(n_post.forward(x))
    x = n_out.forward(x)
    return lmhead_all(x)

# ---- capture pre-final-norm residual from the base model --------------------------------
_captured = {}
_orig_far = base.norm.forward_add_residual
def _cap_far(x, residual):
    normed, summ = _orig_far(x, residual)
    _captured["pre"] = summ
    return normed, summ
base.norm.forward_add_residual = _cap_far

def base_prefill(ids_list):
    """prefill ids (list[int]) from scratch; return (post_norm_hidden[L,H], pre_norm_hidden[L,H])."""
    L = len(ids_list)
    row = engine.page_table[engine.dummy_req.table_idx]
    slot = int(row[0].item())
    row[:L] = torch.arange(L, dtype=torch.int32, device=dev)
    req = Req(input_ids=torch.zeros(L, dtype=torch.int32),
              table_idx=engine.dummy_req.table_idx, cached_len=0, output_len=1,
              uid=-1, sampling_params=None, cache_handle=None)
    req.linear_slot_idx = engine.dummy_req.linear_slot_idx
    b = Batch(reqs=[req], phase="prefill"); b.padded_reqs = b.reqs
    ids = torch.tensor(ids_list, dtype=torch.int32, device=dev)
    b.input_ids = ids
    b.positions = torch.arange(L, dtype=torch.int32, device=dev)
    b.out_loc = row[:L]
    engine.attn_backend.prepare_metadata(b)
    with engine.ctx.forward_batch(b):
        post = base.forward(ids)
    pre = _captured.get("pre")
    row.fill_(slot)
    # NB: no offload-cache reset -- it's a read-through LRU (perf only, not correctness);
    # keeping experts warm across the ~145 re-prefills avoids a cold fetch every step.
    return post.detach(), (pre.detach() if pre is not None else None)

@torch.inference_mode()
def greedy(prompt_ids, gen):
    ids = list(prompt_ids)
    for _ in range(gen):
        post, _ = base_prefill(ids)
        nxt = int(torch.argmax(lmhead_all(post[-1:]), dim=-1).item())
        ids.append(nxt)
    return ids

@torch.inference_mode()
def measure(prompt_text, gen=48):
    pids = tok(prompt_text)["input_ids"] if callable(tok) else tok.encode(prompt_text)
    plen = len(pids)
    S = greedy(pids, gen)
    L = len(S)
    post, pre = base_prefill(S)                       # hidden[i] predicts S[i+1]
    base_pred = torch.argmax(lmhead_all(post), dim=-1)  # [L]
    # sanity: base_pred[i]==S[i+1] over generated region
    gen_ok = sum(int(base_pred[i].item()) == S[i + 1] for i in range(plen - 1, L - 1))
    results = {}
    next_ids = torch.tensor(S, dtype=torch.int32, device=dev)
    for label, hid in (("post_norm", post), ("pre_norm", pre)):
        if hid is None:
            continue
        # draft[i] from hidden[i] and token S[i+1]  -> predicts S[i+2]
        # feed next_ids = S shifted: at pos i we pass token S[i+1]
        shift = torch.roll(next_ids, -1)              # shift[i] = S[i+1]
        logits = head_forward(hid, shift)             # [L,V]
        draft = torch.argmax(logits, dim=-1)          # draft[i] predicts S[i+2]
        acc = tot = 0
        for i in range(plen - 1, L - 2):              # generated region, need S[i+2]
            tot += 1
            if int(draft[i].item()) == S[i + 2]:
                acc += 1
        results[label] = (acc, tot)
    return plen, L, gen_ok, results

PROMPTS = [
    "The history of the printing press begins in",
    "Here is a simple Python function that sorts a list:",
    "In economics, inflation refers to",
]
print("[bench] measuring MTP acceptance...\n", flush=True)
agg = {}
for p in PROMPTS:
    plen, L, gen_ok, res = measure(p, gen=48)
    line = f"prompt[{plen}t]->{L}t  base-greedy-consistency {gen_ok}/{L-1-plen} | "
    for label, (acc, tot) in res.items():
        rate = acc / tot if tot else 0
        agg.setdefault(label, [0, 0]); agg[label][0] += acc; agg[label][1] += tot
        line += f"{label} alpha={rate:.3f} ({acc}/{tot})  "
    print(line, flush=True)
print("\n=== AGGREGATE ===", flush=True)
for label, (acc, tot) in agg.items():
    print(f"  {label}: alpha = {acc/tot:.4f}  ({acc}/{tot})", flush=True)
best = max(agg.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))
a = best[1][0] / best[1][1]
print(f"\n[bench] best-wiring alpha = {a:.4f} ({best[0]})", flush=True)
print(f"[bench] projected MTP-over-offload speedup (draft-1, r~=1.05): {(1+a)/1.05:.2f}x", flush=True)
print(f"[bench] => ~{80*(1+a)/1.05:.0f} tok/s from an 80 tok/s baseline", flush=True)
