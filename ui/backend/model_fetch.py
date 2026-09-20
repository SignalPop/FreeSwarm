"""Download worker: fetch an exact file list of one Hugging Face repo revision, then verify it.

Run by app/downloads.py as a separate process -- one per download -- so a download can be
cancelled by killing it (partial files stay and the next run resumes them) and so hashing
150 GB of weights never touches the control plane's event loop.

Input: one JSON object on stdin. Output: JSON lines on stdout ({"phase": ...}).
The token, when needed, arrives in the HF_TOKEN environment variable -- never on the command
line, where any process listing would show it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def say(**kw) -> None:
    print(json.dumps(kw), flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


class _Bytes:
    """Stands in for huggingface_hub's per-file tqdm bars: counts bytes, reports twice a
    second. The Xet backend stages data in its own cache rather than in .incomplete files,
    so watching the disk shows nothing until a file is finished -- the bars are the only
    live signal."""

    total_done = 0
    last = 0.0

    def __init__(self, *args, **kwargs) -> None:
        self.n = 0
        self.total = kwargs.get("total")

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self._report(force=True)

    def update(self, n=1) -> None:
        n = int(n or 0)
        self.n += n
        _Bytes.total_done += n
        self._report()

    @staticmethod
    def _report(force: bool = False) -> None:
        import time

        now = time.time()
        if force or now - _Bytes.last > 0.5:
            _Bytes.last = now
            say(phase="downloading", bytes=_Bytes.total_done)

    def __getattr__(self, _name):  # set_description, refresh, close, reset, ... -> no-op
        return lambda *a, **k: None


def main() -> int:
    job = json.loads(sys.stdin.read())
    import huggingface_hub.file_download as _fd
    from huggingface_hub import snapshot_download

    # Plain-HTTP downloads report through these bars...
    _fd._get_progress_bar_context = lambda **kw: _Bytes(**kw)
    # ...Xet downloads through their own reporter: count the bytes written to disk.
    try:
        from huggingface_hub.utils import _xet_progress_reporting as _xp

        _orig = _xp.XetDownloadProgressReporter.update_progress

        def _update(self, group_report, _item_reports=None):
            _Bytes.total_done += max(0, group_report.total_bytes_completed - self._prev_bytes_completed)
            _Bytes._report()
            return _orig(self, group_report, _item_reports)

        _xp.XetDownloadProgressReporter.update_progress = _update
    except (ImportError, AttributeError):
        pass  # an older huggingface_hub without Xet: the disk scan still shows finished files

    say(phase="downloading", files=len(job["files"]))
    kwargs = dict(repo_id=job["repo"], revision=job["revision"], allow_patterns=job["files"],
                  max_workers=int(job.get("workers") or 8))
    if job["dest"] == "models":
        kwargs["local_dir"] = job["local_dir"]
    else:
        kwargs["cache_dir"] = job["cache_dir"]
    try:
        path = snapshot_download(**kwargs)
    except Exception as exc:  # noqa: BLE001 -- report it; the parent decides what to show
        say(phase="error", error=f"{type(exc).__name__}: {exc}"[:2000])
        return 1

    if job.get("verify"):
        base = Path(path)
        want = {f["name"]: f["sha256"] for f in job.get("checksums", []) if f.get("sha256")}
        say(phase="verifying", files=len(want))
        bad = []
        for i, (name, digest) in enumerate(sorted(want.items()), 1):
            got = sha256(base / name)
            if got != digest:
                bad.append(name)
            say(phase="verifying", done=i, files=len(want), file=name)
        if bad:
            say(phase="error", error=f"checksum mismatch in {len(bad)} file(s): {', '.join(bad[:5])} -- "
                                     "delete them and download again")
            return 2
    say(phase="done", path=str(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
