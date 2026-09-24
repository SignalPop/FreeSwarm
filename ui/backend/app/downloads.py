"""Download the models FreeToken uses -- or any Hugging Face model -- when they are missing.

**Known models** (MANIFEST) are the checkpoints this install runs, each pinned to the exact
revision in use. Pinning keeps two federated computers on byte-identical weights, and makes
"the model" mean the same thing tomorrow as today. Each lands where the catalog already looks
for it: the repo's ``models/`` folder or the Hugging Face cache.

**Only what the engine loads is fetched.** The gpt-oss repos, for example, carry the same
weights three times (HF safetensors, ``original/``, ``metal/``): 182 GiB for a 61 GiB model.
Known models list their files by rule; a custom repo is downloaded root-level only, and the
plan is shown before anything starts.

**Code is never fetched silently.** The engine executes Python shipped in a model folder
(DeepSeek's ``encoding/encoding_dsv4.py`` at tokenizer load; ``auto_map`` config classes via
``trust_remote_code``). So:

* pickle-format weights (``.bin``, ``.pt``, ``.pth``, ``.pkl``, ``.ckpt``) are never downloaded
  -- loading them can execute code; safetensors cannot;
* ``.py`` files come only for a known model at its pinned revision (listed in the plan), or for
  a custom repo when the operator explicitly opts in, with every file named first.

Every download is checked against free disk space first, runs in its own process (cancel
kills it; partial files stay and the next run resumes), and is verified against the SHA-256
Hugging Face publishes for each weight file before it is reported done.

A Hugging Face token (for gated models) is stored in ui/backend/auth/hf_token (permissions
restricted), passed to the worker in its environment, and never returned to the browser.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .config import settings

router = APIRouter(prefix="/downloads", tags=["downloads"])

WORKER = Path(__file__).resolve().parent.parent / "model_fetch.py"
TOKEN_PATH = Path(__file__).resolve().parent.parent / "auth" / "hf_token"
MODELS_DIR = settings.model_roots[0]
HUB_DIR = settings.model_roots[1] if len(settings.model_roots) > 1 else Path.home() / ".cache" / "huggingface" / "hub"

# Formats that can execute code when loaded (pickle; Keras HDF5 with lambda layers). Never fetched.
PICKLE = ("*.bin", "*.pt", "*.pth", "*.pkl", "*.pickle", "*.ckpt", "*.h5", "*.keras")
# Other frameworks' copies of the same weights: harmless, but the engine loads safetensors only.
OTHER_FORMATS = ("*.msgpack", "*.onnx", "*.onnx_data", "*.ot", "*.gguf", "*.mlmodel", "*.tflite", "*.mlpackage/*")
# Documentation and optional extras: fetched with the model, but a checkpoint missing only
# these still counts as installed (it loads fine without them).
EXTRAS = ("readme*", "license*", "usage_policy*", "notice*", "*.md", "*.png", "*.jpg", "*.jpeg", "*.gif",
          "generation_config.json", "*test*.py", "inference/*")


def _is_extra(name: str) -> bool:
    low = name.lower()
    return any(fnmatch.fnmatch(low, p) or fnmatch.fnmatch(low.rsplit("/", 1)[-1], p) for p in EXTRAS)
DISK_MARGIN = 1.02
DISK_FLOOR = 2 << 30

# The checkpoints this install runs, pinned to the revisions in use (verified against the
# local copies' .cache/huggingface/trees records and HF cache snapshot names).
MANIFEST = [
    {"key": "qwen3.6-35b-a3b", "name": "Qwen3.6-35B-A3B", "repo": "Qwen/Qwen3.6-35B-A3B",
     "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0", "dest": "models", "local": "Qwen3.6-35B-A3B",
     "role": "LLM -- mid tier (MoE, offloaded)", "include": ["*"], "exclude": []},
    {"key": "gpt-oss-20b", "name": "openai/gpt-oss-20b", "repo": "openai/gpt-oss-20b",
     "revision": "6cee5e81ee83917806bbde320786a8fb61efebee", "dest": "hf-cache",
     "role": "LLM -- small tier; fits the RTX 3080 with offload", "include": ["*"], "exclude": ["original/*", "metal/*"]},
    {"key": "gpt-oss-120b", "name": "gpt-oss-120b", "repo": "openai/gpt-oss-120b",
     "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a", "dest": "models", "local": "gpt-oss-120b",
     "role": "LLM -- hard tier (MoE, offloaded)", "include": ["*"], "exclude": ["original/*", "metal/*"]},
    {"key": "deepseek-v4-flash-0731", "name": "DeepSeek-V4-Flash-0731", "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
     "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062", "dest": "models", "local": "DeepSeek-V4-Flash-0731",
     "role": "LLM -- largest (needs the Ada box's GPU)", "include": ["*"], "exclude": [],
     "code_ok": True, "code_note": "the engine runs encoding/encoding_dsv4.py (DeepSeek's chat encoder) from this revision"},
    {"key": "gemma-4-26b-a4b-it", "name": "gemma-4-26B-A4B-it", "repo": "google/gemma-4-26B-A4B-it",
     "revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52", "dest": "models", "local": "gemma-4-26B-A4B-it",
     "role": "LLM -- Gemma-4 MoE, 26B total / 4B active (served text-only)", "include": ["*"], "exclude": [".eval_results/*"]},
    {"key": "muse-glimmer-30b-nvfp4", "name": "Muse-Glimmer-30B-NVFP4", "repo": "RedHatAI/Muse-Glimmer-30B-NVFP4",
     "revision": "e83ead547c81973aaa09c0072c0dbc916e0d8190", "dest": "models", "local": "Muse-Glimmer-30B-NVFP4",
     "role": "LLM -- Muse-Glimmer 30B dense, 4-bit weights (served text-only)", "include": ["*"], "exclude": []},
    # The MiniMax configs are custom classes (auto_map): the engine loads them with
    # trust_remote_code, so the configuration .py is fetched. It imports only transformers.
    # The modeling/processor code is never used (FreeToken has its own) and is left out.
    {"key": "minimax-m2.5-nvfp4", "name": "MiniMax-M2.5-NVFP4", "repo": "nvidia/MiniMax-M2.5-NVFP4",
     "revision": "b6220d658389629b9d507d4b2bb314f41fea7898", "dest": "models", "local": "MiniMax-M2.5-NVFP4",
     "role": "LLM -- MiniMax-M2.5 MoE, 4-bit experts (offloaded)", "include": ["*"], "exclude": ["modeling_*.py"],
     "code_ok": True, "code_note": "the engine reads the model config through configuration_minimax_m2.py (config class only)"},
    {"key": "minimax-m3-nvfp4", "name": "MiniMax-M3-NVFP4", "repo": "nvidia/MiniMax-M3-NVFP4",
     "revision": "901464083161bf8612a29ff7ad29914cd4ab4a85", "dest": "models", "local": "MiniMax-M3-NVFP4",
     "role": "LLM -- MiniMax-M3 MoE, very large (offloaded; served text-only)", "include": ["*"],
     "exclude": ["image_processor.py", "processing_minimax.py", "video_processor.py"],
     "code_ok": True, "code_note": "the engine reads the model config through configuration_minimax_m3_vl.py (config class only)"},
    {"key": "chronos-bolt-base", "name": "amazon/chronos-bolt-base", "repo": "amazon/chronos-bolt-base",
     "revision": "5d9f166d69f47aef3401367a7b842e78fe97b121", "dest": "hf-cache",
     "role": "time series -- zero-shot probabilistic forecaster", "include": ["*"], "exclude": []},
    {"key": "granite-patchtst", "name": "ibm-granite/granite-timeseries-patchtst", "repo": "ibm-granite/granite-timeseries-patchtst",
     "revision": "7fe295d8bc8fbac8041b60ab351882634165517f", "dest": "hf-cache",
     "role": "time series -- PatchTST (7 channels)", "include": ["*"], "exclude": []},
    {"key": "chronos-2", "name": "amazon/chronos-2", "repo": "amazon/chronos-2",
     "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c", "dest": "hf-cache",
     "role": "time series -- Chronos-2: multivariate + covariates (GEX/Greeks as inputs), 120M",
     "include": ["*"], "exclude": []},
    {"key": "kronos-base", "name": "NeoQuasar/Kronos-base", "repo": "NeoQuasar/Kronos-base",
     "revision": "2b554741eca47781b64468546e77fef3e85130e6", "dest": "hf-cache",
     "role": "time series -- Kronos: financial candlestick (OHLCV) foundation model, 102M; needs the tokenizer below",
     "include": ["*"], "exclude": []},
    {"key": "kronos-tokenizer-base", "name": "NeoQuasar/Kronos-Tokenizer-base", "repo": "NeoQuasar/Kronos-Tokenizer-base",
     "revision": "0e0117387f39004a9016484a186a908917e22426", "dest": "hf-cache",
     "role": "time series -- tokenizer used by Kronos-base and Kronos-small", "include": ["*"], "exclude": []},
    {"key": "kronos-small", "name": "NeoQuasar/Kronos-small", "repo": "NeoQuasar/Kronos-small",
     "revision": "901c26c1332695a2a8f243eb2f37243a37bea320", "dest": "hf-cache",
     "role": "time series -- Kronos-small, 25M (faster; same tokenizer)", "include": ["*"], "exclude": []},
]

_jobs: dict[str, dict] = {}
_repo_cache: dict[tuple[str, str], tuple[float, list[dict], str]] = {}


# =======================================================================================
# Token
# =======================================================================================
def _token() -> str | None:
    try:
        t = TOKEN_PATH.read_text(encoding="utf-8").strip()
        return t or None
    except OSError:
        return os.environ.get("HF_TOKEN") or None


def _set_token(token: str) -> None:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not token:
        TOKEN_PATH.unlink(missing_ok=True)
        return
    TOKEN_PATH.write_text(token.strip(), encoding="utf-8")
    try:
        from .auth import _restrict_permissions  # noqa: PLC0415

        _restrict_permissions(TOKEN_PATH)
    except Exception:  # noqa: BLE001
        pass


# =======================================================================================
# Planning: which files, how big, what code
# =======================================================================================
def _repo_files(repo: str, revision: str | None) -> tuple[list[dict], str]:
    """[{name, size, sha256}] of every file in the repo revision, and the resolved commit."""
    key = (repo, revision or "main")
    hit = _repo_cache.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1], hit[2]
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

    try:
        info = HfApi(token=_token()).model_info(repo, revision=revision, files_metadata=True)
    except GatedRepoError:
        raise HTTPException(status_code=403, detail=(
            f"{repo} is gated: accept its licence on huggingface.co and set a Hugging Face token here")) from None
    except RepositoryNotFoundError:
        raise HTTPException(status_code=404, detail=f"no model repo {repo!r} on Hugging Face (or it is private)") from None
    except RevisionNotFoundError:
        raise HTTPException(status_code=404, detail=f"{repo} has no revision {revision!r}") from None
    except Exception as exc:  # noqa: BLE001 -- offline, DNS, proxy...
        raise HTTPException(status_code=502, detail=f"could not reach Hugging Face: {exc}") from None
    files = [{"name": s.rfilename, "size": s.size or 0, "sha256": (s.lfs.sha256 if s.lfs else None)}
             for s in info.siblings or []]
    _repo_cache[key] = (time.time(), files, info.sha)
    return files, info.sha


def _select(files: list[dict], include: list[str], exclude: list[str], allow_code: bool,
            root_only: bool) -> tuple[list[dict], list[dict], list[str]]:
    """(chosen files, python files among them, names excluded as unsafe)."""
    chosen, unsafe = [], []
    for f in files:
        n = f["name"]
        if n in (".gitattributes",) or n.endswith(".md") and "/" in n:
            continue
        if root_only and "/" in n:
            continue
        if not any(fnmatch.fnmatch(n, p) for p in include) or any(fnmatch.fnmatch(n, p) for p in exclude):
            continue
        if any(fnmatch.fnmatch(n.lower(), p) for p in OTHER_FORMATS):
            continue
        if any(fnmatch.fnmatch(n.lower(), p) for p in PICKLE):
            unsafe.append(n)
            continue
        if n.endswith(".py") and not allow_code:
            unsafe.append(n)
            continue
        chosen.append(f)
    return chosen, [f for f in chosen if f["name"].endswith(".py")], unsafe


def _target(entry: dict, commit: str) -> Path:
    """Where the chosen files end up (what the catalog scans)."""
    if entry["dest"] == "models":
        return MODELS_DIR / entry["local"]
    return HUB_DIR / f"models--{entry['repo'].replace('/', '--')}" / "snapshots" / commit


def _present(target: Path, files: list[dict]) -> tuple[int, int]:
    """(bytes of wanted files fully present, count complete)."""
    have = done = 0
    for f in files:
        p = target / f["name"]
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size == f["size"]:
            have += size
            done += 1
    return have, done


def _partial_bytes(entry: dict) -> int:
    """Bytes already written for in-flight files (resume state)."""
    if entry["dest"] == "models":
        d = MODELS_DIR / entry["local"] / ".cache" / "huggingface" / "download"
    else:
        d = HUB_DIR / f"models--{entry['repo'].replace('/', '--')}" / "blobs"
    total = 0
    try:
        for p in d.rglob("*.incomplete"):
            total += p.stat().st_size
    except OSError:
        pass
    return total


def plan(entry: dict) -> dict:
    files, commit = _repo_files(entry["repo"], entry.get("revision"))
    chosen, code, unsafe = _select(files, entry.get("include") or ["*"], entry.get("exclude") or [],
                                   bool(entry.get("code_ok")), bool(entry.get("root_only")))
    target = _target(entry, commit)
    total = sum(f["size"] for f in chosen)
    have, done = _present(target, chosen)
    complete_names = {f["name"] for f in chosen if _present(target, [f])[1] == 1}
    required = [f for f in chosen if not _is_extra(f["name"])]
    _, req_done = _present(target, required)
    missing_extras = [f["name"] for f in chosen if _is_extra(f["name"]) and not (target / f["name"]).is_file()]
    drive = target.anchor or str(target)
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    return {
        "repo": entry["repo"], "revision": commit, "dest": entry["dest"], "target": str(target),
        "files": len(chosen), "bytes": total, "present_bytes": have, "present_files": done,
        "partial_bytes": _partial_bytes(entry), "needed_bytes": max(0, total - have),
        "free_bytes": free, "drive": drive,
        "fits": free >= (total - have) * DISK_MARGIN + DISK_FLOOR,
        "code_files": [f["name"] for f in code], "code_note": entry.get("code_note"),
        "excluded_unsafe": unsafe[:40], "excluded_unsafe_count": len(unsafe),
        "skipped_bytes": sum(f["size"] for f in files) - total,
        "installed": req_done == len(required) and len(required) > 0,
        "complete": done == len(chosen),
        "missing_extras": missing_extras[:20],
        "_chosen": chosen,
        "_complete_names": complete_names,
    }


# =======================================================================================
# Jobs
# =======================================================================================
def _run(job: dict, entry: dict, p: dict) -> None:
    job["phase"] = "starting"
    payload = {"repo": entry["repo"], "revision": p["revision"], "files": [f["name"] for f in p["_chosen"]],
               "dest": entry["dest"], "local_dir": str(MODELS_DIR / entry["local"]) if entry["dest"] == "models" else None,
               "cache_dir": str(HUB_DIR), "verify": True,
               # Only files not already complete when the run started: resuming a 150 GB model
               # must not re-hash the shards that were verified the first time.
               "checksums": [{"name": f["name"], "sha256": f["sha256"]} for f in p["_chosen"]
                             if f.get("sha256") and f["name"] not in p["_complete_names"]]}
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    tok = _token()
    if tok:
        env["HF_TOKEN"] = tok
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen([sys.executable, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env, creationflags=flags)
    job["pid"] = proc.pid
    job["_proc"] = proc
    proc.stdin.write(json.dumps(payload))
    proc.stdin.close()
    tail: list[str] = []
    for line in proc.stdout:
        line = line.strip()
        try:
            msg = json.loads(line)
        except ValueError:
            if line:
                tail.append(line[:300])
                del tail[:-20]
            continue
        job["phase"] = msg.get("phase", job["phase"])
        if msg.get("bytes") is not None:
            job["_session_bytes"] = int(msg["bytes"])
        if msg.get("phase") == "verifying":
            job["verify_done"], job["verify_files"] = msg.get("done", 0), msg.get("files", 0)
        if msg.get("phase") == "error":
            job["error"] = msg.get("error")
    proc.wait()
    if job.get("cancelled"):
        job["phase"] = "cancelled"
    elif proc.returncode != 0 and job["phase"] != "error":
        job["phase"] = "error"
        job["error"] = job.get("error") or ("\n".join(tail[-5:]) or f"worker exited with {proc.returncode}")
    job["finished_at"] = time.time()
    job.pop("_proc", None)


def _progress(job: dict) -> dict:
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    if job["phase"] in ("downloading", "starting"):
        entry = job["_entry"]
        have, done = _present(Path(job["target"]), job["_files"])
        # The worker's own byte count (this run) on top of what was already complete at the
        # start; the disk scan is the fallback before the first report arrives.
        if "_session_bytes" in job:
            now_bytes = min(job["bytes"], job["_base_bytes"] + job["_session_bytes"])
        else:
            now_bytes = have + _partial_bytes(entry)
        t = time.time()
        hist = job.setdefault("_hist", [])
        hist.append((t, now_bytes))
        hist[:] = [h for h in hist if t - h[0] < 20]
        rate = (hist[-1][1] - hist[0][1]) / max(1e-6, hist[-1][0] - hist[0][0]) if len(hist) > 1 else 0.0
        remaining = max(0, job["bytes"] - now_bytes)
        out.update(done_bytes=now_bytes, done_files=done, rate_bytes_s=rate,
                   eta_s=(remaining / rate) if rate > 256 * 1024 else None)
    return out


def _entry_for(key: str) -> dict:
    e = next((m for m in MANIFEST if m["key"] == key), None)
    if e is None:
        raise HTTPException(status_code=404, detail=f"no known model {key!r}")
    return e


def start(entry: dict) -> dict:
    running = [j for j in _jobs.values() if j["repo"] == entry["repo"] and j["phase"] in ("starting", "downloading", "verifying")]
    if running:
        raise HTTPException(status_code=409, detail=f"{entry['repo']} is already downloading")
    p = plan(entry)
    # Installed = everything the engine loads is present. Missing READMEs or licences are no
    # reason to start a download (a test "download" of an installed model did exactly that).
    if p["installed"]:
        raise HTTPException(status_code=409, detail=f"{entry['name']} is already downloaded")
    if not p["fits"]:
        raise HTTPException(status_code=507, detail=(
            f"not enough space on {p['drive']}: needs {p['needed_bytes'] / 2**30:.1f} GiB (+2%), "
            f"{p['free_bytes'] / 2**30:.1f} GiB free"))
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "key": entry.get("key"), "name": entry["name"], "repo": entry["repo"], "revision": p["revision"],
           "target": p["target"], "bytes": p["bytes"], "files": p["files"], "phase": "queued",
           "started_at": time.time(), "error": None, "_entry": entry, "_files": p["_chosen"],
           "_base_bytes": p["present_bytes"]}
    _jobs[jid] = job
    threading.Thread(target=_run, args=(job, entry, p), name=f"download-{jid}", daemon=True).start()
    return _progress(job)


# =======================================================================================
# Routes (loopback console, behind the operator's auth)
# =======================================================================================
def _known_status() -> list[dict]:
    out = []
    for e in MANIFEST:
        row = {k: e[k] for k in ("key", "name", "repo", "revision", "dest", "role")}
        job = next((j for j in reversed(list(_jobs.values())) if j["repo"] == e["repo"]), None)
        try:
            p = plan(e)
            row.update({k: v for k, v in p.items() if not k.startswith("_")})
            row["state"] = "installed" if p["installed"] else ("partial" if p["present_bytes"] or p["partial_bytes"] else "missing")
        except HTTPException as exc:
            # Hugging Face unreachable (offline, proxy): still say what is on disk.
            from .catalog import list_models

            on_disk = any(m["id"] in (e["name"], e["repo"], e.get("local")) for m in list_models())
            row.update(state="installed" if on_disk else "unknown", installed=on_disk, offline=True, error=exc.detail)
        if job and job["phase"] not in ("done",):
            row["job"] = _progress(job)
        out.append(row)
    return out


@router.get("")
async def overview() -> dict:
    return {"known": await asyncio.to_thread(_known_status),
            "jobs": [_progress(j) for j in _jobs.values()],
            "token_set": _token() is not None,
            "models_dir": str(MODELS_DIR), "hub_dir": str(HUB_DIR)}


@router.post("/known/{key}")
async def download_known(key: str) -> dict:
    return await asyncio.to_thread(start, _entry_for(key))


class CustomReq(BaseModel):
    repo: str = Field(..., pattern=r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$", max_length=120)
    revision: str | None = Field(None, max_length=80)
    dest: str = Field("hf-cache", pattern="^(hf-cache|models)$")
    include_code: bool = False
    subfolders: bool = False


def _custom_entry(req: CustomReq) -> dict:
    return {"name": req.repo, "repo": req.repo, "revision": req.revision, "dest": req.dest,
            "local": req.repo.split("/")[-1], "include": ["*"], "exclude": [], "code_ok": req.include_code,
            "root_only": not req.subfolders}


@router.post("/plan")
async def plan_custom(req: CustomReq) -> dict:
    """What a custom download would fetch -- shown before anything starts."""
    p = await asyncio.to_thread(plan, _custom_entry(req))
    return {k: v for k, v in p.items() if not k.startswith("_")}


@router.post("/custom")
async def download_custom(req: CustomReq) -> dict:
    return await asyncio.to_thread(start, _custom_entry(req))


@router.post("/jobs/{jid}/cancel")
async def cancel(jid: str) -> dict:
    job = _jobs.get(jid)
    if job is None:
        raise HTTPException(status_code=404, detail="no such download")
    proc = job.get("_proc")
    job["cancelled"] = True
    if proc is not None and proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"] if sys.platform == "win32"
                       else ["kill", "-9", str(proc.pid)], capture_output=True, check=False)
    return _progress(job)


class TokenReq(BaseModel):
    token: str = Field("", max_length=400)


_token_user: dict[str, str | None] = {}  # token -> Hugging Face username it belongs to


def _whoami(token: str) -> str | None:
    """The account a token belongs to; None when Hugging Face cannot be reached.

    Raises HTTPException(401) when Hugging Face rejects the token.
    """
    if token in _token_user:
        return _token_user[token]
    from huggingface_hub import HfApi

    try:
        info = HfApi().whoami(token=token)
    except Exception as exc:  # noqa: BLE001 -- rejected (HTTP error; its class varies by hub version), or offline
        if getattr(getattr(exc, "response", None), "status_code", None) in (401, 403):
            raise HTTPException(status_code=401, detail=(
                "Hugging Face rejected this token. It may be mistyped, revoked or expired -- "
                "create a new one at huggingface.co/settings/tokens")) from None
        return None
    _token_user[token] = info.get("name")
    return _token_user[token]


def _token_status() -> dict:
    t = _token()
    if not t:
        return {"token_set": False, "token_user": None}
    try:
        return {"token_set": True, "token_user": _whoami(t)}
    except HTTPException:
        return {"token_set": True, "token_user": None, "token_invalid": True}


@router.get("/token")
async def token_status() -> dict:
    """Whether a token is stored and whose account it is (the token itself is never returned)."""
    return await asyncio.to_thread(_token_status)


@router.put("/token")
async def set_token(req: TokenReq) -> dict:
    """Check the token with Hugging Face, then store it (or clear it, with an empty value).

    Write-only: never echoed. Returns the account name so the page can confirm who it is.
    """
    t = req.token.strip()
    if t and "@" in t:
        raise HTTPException(status_code=400, detail=(
            "That looks like an email address. Hugging Face does not let apps sign in with your email "
            "and password -- paste an access token instead (it starts with hf_)"))
    if t and not t.startswith("hf_"):
        raise HTTPException(status_code=400, detail=(
            "That is not an access token. Hugging Face does not let apps sign in with your password -- "
            "create a token at huggingface.co/settings/tokens and paste it here (it starts with hf_)"))
    user = await asyncio.to_thread(_whoami, t) if t else None
    await asyncio.to_thread(_set_token, t)
    _repo_cache.clear()
    return {"token_set": bool(t), "token_user": user, "verified": user is not None}
