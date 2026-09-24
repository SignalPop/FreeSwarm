"""Subprocess entry for tests/models/test_qwen4_exp.py: serve a tiny Qwen4-Exp checkpoint
through the real FreeToken stack (``LLM`` -> scheduler -> engine: config parse, weight
loader, QSA KV pool, GDN state pool, attention/MoE backends, chunked prefill, eager decode),
greedy-generate, and dump every sampled row's logits keyed by (request, position).

Runs in its own process because ``Engine`` requires CUDA to be uninitialized and binds a
process group; the caller compares the dump against the HF reference.

usage: python qwen4_exp_engine_runner.py <ckpt_dir> <out.pt> <max_extend_tokens> <n_new> <prompts.json>
"""

from __future__ import annotations

import json
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen4_exp_tiny import cuda_home_for_jit  # noqa: E402

cuda_home_for_jit()

import torch  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    ckpt, out, max_ext, n_new, prompts_path = sys.argv[1:6]
    with open(prompts_path) as f:
        prompts = json.load(f)

    from freetoken.core import SamplingParams
    from freetoken.engine.config import EngineConfig
    from freetoken.llm import LLM

    port = _free_port()  # the default fixed :2333 collides with any concurrent engine
    EngineConfig.distributed_addr = property(lambda self: f"tcp://127.0.0.1:{port}")

    llm = LLM(
        ckpt,
        dtype=torch.bfloat16,
        moe_backend="fused",
        max_running_req=4,
        max_extend_tokens=int(max_ext),
        max_seq_len_override=512,
        num_page_override=4096,
        attention_backend=os.environ.get("QWEN4_TEST_ATTN", "auto"),
    )
    cur: dict = {}
    logits: dict = {}
    orig_sample = llm.engine.sampler.sample
    orig_fb = llm.engine.forward_batch

    def sample(batch_logits, args):
        for row, (uid, dl) in enumerate(cur["rows"]):
            logits[(uid, dl - 1)] = batch_logits[row].float().cpu().clone()
        return orig_sample(batch_logits, args)

    def forward_batch(batch, args):
        cur["rows"] = [(r.uid, r.device_len) for r in batch.reqs]
        return orig_fb(batch, args)

    llm.engine.sampler.sample = sample
    llm.engine.forward_batch = forward_batch
    res = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=int(n_new), ignore_eos=True)
    )
    torch.save(
        {
            "prompts": prompts,
            "outputs": [r["token_ids"] for r in res],
            "logits": logits,
            "cache_type": llm.engine.config.cache_type,
            "graph_bs": list(getattr(llm.engine.graph_runner, "graph_bs_list", [])),
            "pool": type(llm.engine.kv_cache).__name__,
            "attention_backend": llm.engine.config.attention_backend,
        },
        out,
    )
    llm.shutdown()


if __name__ == "__main__":
    main()
