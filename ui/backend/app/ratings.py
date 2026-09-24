"""Published quality scores for LLMs: the single table the console and the swarm read.

* ``swe`` -- SWE-bench Verified, % of real GitHub issues resolved (coding ability).
* ``aa``  -- Artificial Analysis Intelligence Index v4.3.2 (artificialanalysis.ai), a composite
  of reasoning, knowledge, math and coding evals.

A score belongs to the base model; quantized or re-hosted copies (NVFP4, FP8, a Groq or
OpenRouter endpoint, ``model@computer`` on the network) are matched to it by name. ``None``
means no published number was found -- nothing here is estimated. Each value carries its
source. Where a developer publishes only SWE-bench Pro or DeepSWE, ``swe`` stays ``None``:
those are different benchmarks on a different scale.

The swarm uses these to pick models (see ``swarm_policy``): SWE decides which models search
alongside the local ones, AA decides who is asked for new ideas when the search is stuck.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from fastapi import APIRouter

router = APIRouter(tags=["ratings"])

AA_VERSION = "v4.3.2"
COLLECTED = "2026-09-23"
_AA = "https://artificialanalysis.ai/models/"
_HF = "https://huggingface.co/"
_OAI_CARD = "https://arxiv.org/html/2508.10925v1"  # gpt-oss model card, Table 3


@dataclass(frozen=True)
class Rating:
    key: str
    label: str
    match: str  # regex, matched against the lower-cased model name
    swe: float | None
    swe_source: str | None
    aa: float | None
    aa_source: str | None


def _r(key: str, label: str, match: str, swe: float | None, swe_source: str | None,
       aa: float | None, aa_slug: str | None) -> Rating:
    return Rating(key, label, match, swe, swe_source, aa, _AA + aa_slug if aa_slug else None)


_OR = "https://openrouter.ai/api/v1/models"  # OpenRouter republishes AA's index per model

# Ordered most-specific first: the first pattern that matches wins.
RATINGS: list[Rating] = [
    # ---- Hosted models (Groq / OpenRouter). Frontier labs mostly publish DeepSWE / SWE-bench
    # Pro now, so many have no SWE-bench Verified number. AA values here are the top reasoning
    # setting as OpenRouter reports it; for OpenRouter models not listed, the live value is used.
    Rating("claude-opus-5.5", "Claude Opus 5.5", r"claude-opus-5[.-]5", None, None, 57.6, _OR),
    Rating("claude-fable-5.1", "Claude Fable 5.1", r"claude-fable-5[.-]1", None, None, 53.4, _OR),
    Rating("claude-opus-5", "Claude Opus 5", r"claude-opus-5(?![.-]\d)", None, None, 50.8, _OR),
    Rating("claude-sonnet-5", "Claude Sonnet 5", r"claude-sonnet-5(?![.-]\d)", None, None, 38.2, _OR),
    Rating("gpt-5.6-sol", "GPT-5.6 Sol", r"gpt-5\.6-sol", None, None, 47.0, _OR),
    Rating("gpt-5.6-terra", "GPT-5.6 Terra", r"gpt-5\.6-terra", None, None, 42.1, _OR),
    Rating("gpt-5.6-luna", "GPT-5.6 Luna", r"gpt-5\.6-luna", None, None, 37.3, _OR),
    Rating("grok-4.7", "Grok 4.7", r"grok-4\.7", None, None, 46.4, _OR),
    Rating("glm-5.3", "GLM-5.3", r"glm-5\.3(?!-flash)", None, None, 44.8, _OR),
    Rating("kimi-k3", "Kimi K3", r"kimi-k3", None, None, 43.6, _OR),
    Rating("kimi-k2.6", "Kimi K2.6", r"kimi-k2\.6", 80.2, _HF + "moonshotai/Kimi-K2.6", 27.0, _OR),
    Rating("kimi-k2-0905", "Kimi K2 0905", r"kimi-k2-0905", 69.2, _HF + "moonshotai/Kimi-K2-Instruct-0905", None, None),
    Rating("gemini-3.8-flash", "Gemini 3.8 Flash", r"gemini-3\.8-flash", None, None, 40.9, _OR),
    Rating("gemini-3.1-pro", "Gemini 3.1 Pro Preview", r"gemini-3\.1-pro", 80.6,
           "https://deepmind.google/models/model-cards/gemini-3-1-pro/", 29.7, _OR),
    Rating("qwen3.8-2.4t", "Qwen3.8 2.4T-A95B", r"qwen3\.8-2\.4t", None, None, 39.9, _OR),
    Rating("deepseek-v4.1-flash", "DeepSeek V4.1 Flash", r"deepseek-v4\.1-flash", None, None, 39.5, _OR),
    Rating("deepseek-v4-pro-0813", "DeepSeek V4 Pro 0813", r"deepseek-v4-pro-0813", None, None, 36.0, _OR),
    Rating("deepseek-v4-pro", "DeepSeek V4 Pro (max)", r"deepseek-v4-pro", 80.6, _HF + "deepseek-ai/DeepSeek-V4-Pro", 30.4, _OR),
    Rating("qwen3.5-397b", "Qwen3.5-397B-A17B", r"qwen3\.5-397b", 76.4, _HF + "Qwen/Qwen3.5-397B-A17B", 18.4, _OR),
    Rating("qwen3-coder-next", "Qwen3-Coder-Next", r"qwen3-coder-next", 70.6, _HF + "Qwen/Qwen3-Coder-Next", 9.2, _OR),
    Rating("minimax-m2.7", "MiniMax M2.7", r"minimax-m2\.7", None, None, 22.8, _OR),
    # ---- Models this install runs locally (and their hosted copies, matched by name)
    _r("minimax-m3", "MiniMax-M3", r"minimax-m3", 80.5, _HF + "MiniMaxAI/MiniMax-M3", 29, "minimax-m3"),
    _r("minimax-m2.5", "MiniMax-M2.5", r"minimax-m2\.5", 80.2, _HF + "MiniMaxAI/MiniMax-M2.5", 23, "minimax-m2-5"),
    _r("qwen3.6-27b", "Qwen3.6 27B (reasoning)", r"qwen3\.6-27b", 77.2, _HF + "Qwen/Qwen3.6-27B", 21, "qwen3-6-27b"),
    _r("muse-glimmer-30b", "Muse Glimmer 30B (high)", r"muse-glimmer", 76.0, _HF + "meta-models/Muse-Glimmer-30B", 17, "muse-glimmer"),
    _r("glm-4.7", "GLM-4.7", r"glm-4\.7", 73.8, _HF + "zai-org/GLM-4.7", 22, "glm-4-7"),
    _r("qwen3.6-35b-a3b", "Qwen3.6 35B-A3B (reasoning)", r"qwen3\.6-35b-a3b", 73.4, _HF + "Qwen/Qwen3.6-35B-A3B", 18, "qwen3-6-35b-a3b"),
    _r("gpt-oss-120b", "gpt-oss-120b (high reasoning)", r"gpt-oss-120b", 62.4, _OAI_CARD, 12, "gpt-oss-120b"),
    _r("gpt-oss-20b", "gpt-oss-20b (high reasoning)", r"gpt-oss-20b", 60.7, _OAI_CARD, 9, "gpt-oss-20b"),
    _r("glm-5.3-flash", "GLM-5.3-Flash", r"glm-5\.3-flash", None, None, 42, "glm-5-3-flash"),
    _r("qwen3.8-flash-next", "Qwen3.8-Flash-Next", r"qwen3\.8-flash-next", None, None, 40, "qwen3-8-flash-next"),
    _r("deepseek-v4-flash", "DeepSeek V4 Flash 0731 (max effort)", r"deepseek-v4-flash", None, None, 34, "deepseek-v4-flash"),
    # AA lists two Qwen3.8 27B entries: 34 at xhigh effort (model page), 20 on the leaderboard summary.
    _r("qwen3.8-27b", "Qwen3.8 27B (xhigh effort)", r"qwen3\.8-27b", None, None, 34, "qwen3-8-27b"),
    _r("glm-5.2", "GLM-5.2 (max)", r"glm-5\.2", None, None, 34, "glm-5-2"),
    _r("gemma-4-31b", "Gemma 4 31B (reasoning; AA marks this estimated)", r"gemma-4-31b", None, None, 19, "gemma-4-31b"),
    _r("gemma-4-26b-a4b", "Gemma 4 26B-A4B (reasoning)", r"gemma-4-26b-a4b", None, None, 17, "gemma-4-26b-a4b"),
    _r("gemma-4-12b", "Gemma 4 12B", r"gemma-4-12b", None, None, 14, "gemma-4-12b"),
    _r("qwen3-30b-a3b", "Qwen3 30B-A3B (reasoning, Apr 2025)", r"qwen3-30b-a3b", None, None, 8, "qwen3-30b-a3b-instruct-reasoning"),
]

_COMPILED = [(re.compile(r.match), r) for r in RATINGS]


def rating_for(*names: str | None) -> Rating | None:
    hay = " ".join(n for n in names if n).lower()
    return next((r for rx, r in _COMPILED if rx.search(hay)), None)


def scores(*names: str | None) -> dict:
    """{swe, aa, label} for a model (Nones when unrated) -- what lists attach per row."""
    r = rating_for(*names)
    return {"swe": r.swe, "aa": r.aa, "rating_label": r.label} if r else {"swe": None, "aa": None, "rating_label": None}


@router.get("/ratings")
async def list_ratings() -> dict:
    return {"aa_version": AA_VERSION, "collected": COLLECTED, "ratings": [asdict(r) for r in RATINGS]}
