"""Vectors and search for the research library.

Two retrievers, fused:

* **Dense** -- a small local sentence-embedding model (default ``BAAI/bge-small-en-v1.5``,
  384 dims) run on the CPU. On the CPU on purpose: the GPUs belong to the engines and
  forecasters, and a few hundred chunks per document embed in seconds. Set
  ``FREESWARM_EMBED_MODEL`` / ``FREESWARM_EMBED_DEVICE`` to change either. The model is
  fetched from Hugging Face on first use; when it cannot be loaded (offline, no torch) the
  library still works on the lexical retriever alone, and says so.
* **Lexical** -- BM25 over the chunk text. It is what finds exact identifiers
  (``wall_pos``, ``PinTrend_ProbPin``, ``C6_event_or_wallpin_x_gex``) that a dense model
  blurs, and it needs nothing installed.

The two rankings are merged by reciprocal-rank fusion, so neither's score scale matters.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import Counter

import numpy as np

logger = logging.getLogger("freetoken.research")

EMBED_MODEL = os.environ.get("FREESWARM_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DEVICE = os.environ.get("FREESWARM_EMBED_DEVICE", "cpu")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge's retrieval prompt
RRF_K = 60

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokens(text: str) -> list[str]:
    """Lower-case words; identifiers are kept whole AND split on '_' so `wall_pos` matches
    both `wall_pos` and `wall position`."""
    out = []
    for t in _TOKEN_RE.findall((text or "").lower()):
        if len(t) < 2 and not t.isdigit():
            continue
        out.append(t)
        if "_" in t:
            out += [p for p in t.split("_") if len(p) > 1]
    return out


# =======================================================================================
# Dense embeddings
# =======================================================================================
class Embedder:
    """Lazily loaded; thread-safe; `available` is False (with `error`) when it cannot load."""

    def __init__(self, name: str = EMBED_MODEL, device: str = EMBED_DEVICE) -> None:
        self.name = name
        self.device = device
        self.error: str | None = None
        self._model = None
        self._tok = None
        self._lock = threading.Lock()
        self._tried = False

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def _load(self) -> None:
        if self._tried:
            return
        with self._lock:
            if self._tried:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(self.name)
                self._model = AutoModel.from_pretrained(self.name).to(self.device).eval()
                torch.set_grad_enabled(False)
                logger.info("research embeddings: %s on %s", self.name, self.device)
            except Exception as exc:  # noqa: BLE001 -- lexical search still works
                self.error = f"{type(exc).__name__}: {exc}"[:500]
                self._model = None
                logger.warning("research embeddings unavailable (%s); lexical search only", self.error)
            self._tried = True

    def embed(self, texts: list[str], query: bool = False, batch: int = 32) -> np.ndarray:
        """Unit-length float32 rows, one per text."""
        self._load()
        if self._model is None:
            raise RuntimeError(self.error or "embedding model not loaded")
        import torch

        out = []
        with self._lock, torch.no_grad():
            for i in range(0, len(texts), batch):
                part = [(QUERY_PREFIX + t) if query else t for t in texts[i:i + batch]]
                x = self._tok(part, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                h = self._model(**x).last_hidden_state[:, 0]         # CLS pooling, as bge is trained
                out.append(torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


_embedder: Embedder | None = None


def embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def set_embedder(e: Embedder | None) -> None:
    """Tests swap in a fake; None resets to the configured model."""
    global _embedder
    _embedder = e


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray | None:
    return np.frombuffer(b, np.float32) if b else None


# =======================================================================================
# Lexical: BM25
# =======================================================================================
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for d in self.tf:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf), np.float32)
        if not self.tf:
            return s
        norm = self.k1 * (1 - self.b + self.b * self.len / max(self.avg, 1e-9))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = np.array([d.get(t, 0) for d in self.tf], np.float32)
            s += idf * f * (self.k1 + 1) / (f + norm)
        return s


# =======================================================================================
# Fused search over an in-memory corpus
# =======================================================================================
class Corpus:
    """The searchable rows of one scope (a project plus the shared documents)."""

    def __init__(self, rows: list[dict]) -> None:
        # rows: {"key", "text", "vector" (np.ndarray | None), ...anything the caller wants back}
        self.rows = rows
        self.bm25 = BM25([tokens(r["text"]) for r in rows])
        vecs = [r.get("vector") for r in rows]
        dim = next((len(v) for v in vecs if v is not None), 0)
        self.has_dense = dim > 0
        self.matrix = np.stack([v if v is not None and len(v) == dim else np.zeros(dim, np.float32) for v in vecs]) \
            if self.has_dense else None

    def search(self, query: str, qvec: np.ndarray | None, k: int = 8, keep=None) -> list[dict]:
        """Top k rows by reciprocal-rank fusion of BM25 and cosine; `keep(row)` filters first.
        Each hit carries `score` (the fused score), `cosine` and `bm25`."""
        if not self.rows:
            return []
        mask = np.array([bool(keep(r)) if keep else True for r in self.rows])
        if not mask.any():
            return []
        lex = self.bm25.scores(tokens(query))
        fused = np.zeros(len(self.rows), np.float64)
        rankings = [lex]
        cos = None
        if qvec is not None and self.has_dense and self.matrix is not None and self.matrix.shape[1] == len(qvec):
            cos = self.matrix @ qvec.astype(np.float32)
            rankings.append(cos)
        for sc in rankings:
            live = np.where(mask & (sc > 0) if sc is lex else mask)[0]
            order = live[np.argsort(-sc[live], kind="stable")]
            fused[order] += 1.0 / (RRF_K + np.arange(1, len(order) + 1))
        idx = [i for i in np.argsort(-fused, kind="stable") if mask[i] and fused[i] > 0][:k]
        return [{**self.rows[i], "score": round(float(fused[i]), 5),
                 "cosine": round(float(cos[i]), 4) if cos is not None else None,
                 "bm25": round(float(lex[i]), 3)} for i in idx]


def similarity(a: np.ndarray | None, b: np.ndarray | None, ta: str, tb: str) -> float:
    """How related two texts are, in [0, 1]: cosine of their vectors when both exist, else the
    token overlap (Jaccard) -- a weaker but always-available stand-in."""
    if a is not None and b is not None and len(a) == len(b) and len(a):
        return float(np.clip(np.dot(a, b), 0, 1))
    sa, sb = set(tokens(ta)), set(tokens(tb))
    return len(sa & sb) / max(1, len(sa | sb))
