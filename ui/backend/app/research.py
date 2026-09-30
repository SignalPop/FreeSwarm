"""The research library: documents the operator drops in, turned into ideas the swarm tests.

A strategy report (PDF or HTML) holds three things the swarm can use: **findings** (what
worked, on what data, with which caveats), **evidence** (tables, charts) and **method**
(signal and backtest code). The library keeps all three and puts them to work:

1. **Store.** The original file is kept under ``research/<doc id>/``; its figures beside it.
2. **Parse** (research_parse): typed chunks -- prose per section, tables as markdown,
   figures with captions, code listings with their file names.
3. **Vectorize** (research_index): every chunk is embedded with a small local model and
   indexed for BM25; search fuses the two. A figure is found through its caption and, when
   a vision model is chosen in the settings, through a description of what it shows.
4. **Extract ideas.** A model reads the document and lists its testable trading ideas --
   rules, horizon, fields, the reported evidence, the caveats, the experiment that would
   falsify it, and which code blocks implement it. Ideas are embedded too.
5. **Feed the idea stream.** Ideas relevant to a running objective are written into its
   ``ideas`` table (escalation.py) with trigger ``research``, where every agent reads them in
   its brief: a couple when the document arrives, then one at a time on a slow drip, most
   relevant first, never the same one twice. The operator can also send one by hand.
6. **Make the code runnable.** Every Python listing ships into each sandbox run as
   ``/work/.ft/research/<package>/<module>.py`` (``from research.<package> import <module>``),
   and agents search the library and read any chunk with the research_search / research_get
   tools. The idea models asked for new directions see the relevant findings and code in
   their prompt (escalation._prompt).

Findings are the documents' authors' claims on their data, not facts about this objective's
data: every place they reach an agent says so, and an idea from a document is tested like
any other idea.

Storage: research.sqlite3 beside the control plane (own connection and lock: an ingest
writes hundreds of vectors and must not hold the objectives lock while it does).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import projects
from . import research_index as RI
from . import research_parse as RP

logger = logging.getLogger("freetoken.research")
router = APIRouter(tags=["research"])

BACKEND = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND / "research.sqlite3"
ROOT = BACKEND / "research"

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
IDEA_MAX_TOKENS = 8000
IDEA_WINDOW_CHARS = 30_000          # document digest per extraction call
IDEA_MAX_WINDOWS = 4
MAX_IDEAS_PER_DOC = 16
FIGURE_DESCRIBE_MAX = 30
FIGURE_MAX_TOKENS = 600
TICK_S = 300
CODE_SHIP_MAX_CHARS = 400_000       # per document, into each sandbox run

DEFAULT_SETTINGS = {
    # Model that extracts ideas; "" = the project's first idea rung (its best free model).
    "idea_model": "",
    # Vision model that describes figures; "" = figures are indexed by their captions only.
    "figure_model": "",
    # Feed relevant ideas into running objectives' idea streams.
    "auto_push": True,
    "push_on_ingest": 2,            # per objective, when a document finishes
    "push_gap_minutes": 120,        # then at most one more per objective this often
    # Least relevance to push automatically: cosine of the idea and the objective (dense),
    # or their token overlap when no embedding model loaded (a weaker, lower scale).
    "min_relevance": 0.5,
    "min_overlap": 0.04,
}

CompleteFn = Callable[[str, list[dict], int, str], Awaitable[str]]
LoadedFn = Callable[[], list[dict]]
_complete: CompleteFn | None = None
_loaded: LoadedFn | None = None
_queue: asyncio.Queue | None = None
_active: dict[str, str] = {}        # doc id -> stage, while processing

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_version = 0                        # bumped on every write the search index depends on
_corpora: dict[str, tuple[int, RI.Corpus]] = {}
_obj_vecs: dict[str, tuple[str, Any]] = {}


# =======================================================================================
# Storage
# =======================================================================================
def db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL DEFAULT '', title TEXT NOT NULL DEFAULT '',
                filename TEXT NOT NULL, kind TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                pages INTEGER, package TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'queued',
                detail TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '', outline TEXT NOT NULL DEFAULT '[]',
                author TEXT NOT NULL DEFAULT 'operator', idea_model TEXT NOT NULL DEFAULT '',
                embed_model TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL, seq INTEGER NOT NULL,
                kind TEXT NOT NULL, section TEXT NOT NULL DEFAULT '', page INTEGER, label TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL, embed_text TEXT NOT NULL, image TEXT NOT NULL DEFAULT '',
                meta TEXT NOT NULL DEFAULT '{}', vector BLOB
            );
            CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, seq);
            CREATE TABLE IF NOT EXISTS doc_ideas (
                id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL, seq INTEGER NOT NULL,
                title TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', rules TEXT NOT NULL DEFAULT '',
                horizon TEXT NOT NULL DEFAULT '', fields TEXT NOT NULL DEFAULT '[]', evidence TEXT NOT NULL DEFAULT '',
                caveats TEXT NOT NULL DEFAULT '', falsify TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '[]',
                code_refs TEXT NOT NULL DEFAULT '[]', model TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL,
                dismissed INTEGER NOT NULL DEFAULT 0, vector BLOB
            );
            CREATE INDEX IF NOT EXISTS ideas_doc ON doc_ideas(doc_id);
            CREATE TABLE IF NOT EXISTS pushes (
                objective_id TEXT NOT NULL, doc_idea_id INTEGER NOT NULL, idea_row_id INTEGER, ts REAL NOT NULL,
                relevance REAL, by TEXT NOT NULL DEFAULT 'auto', PRIMARY KEY (objective_id, doc_idea_id)
            );
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        _conn.commit()
    return _conn


def _bump() -> None:
    global _version
    _version += 1


def settings() -> dict:
    with _lock:
        r = db().execute("SELECT value FROM settings WHERE key='settings'").fetchone()
    saved = json.loads(r["value"]) if r else {}
    return {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}


def save_settings(changes: dict) -> dict:
    cur = settings()
    for k, v in changes.items():
        if k in DEFAULT_SETTINGS and v is not None:
            cur[k] = type(DEFAULT_SETTINGS[k])(v)
    with _lock:
        db().execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('settings', ?)", (json.dumps(cur),))
        db().commit()
    return cur


def _doc_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["outline"] = json.loads(d.get("outline") or "[]")
    return d


def _idea_row(r: sqlite3.Row) -> dict:
    d = {k: r[k] for k in r.keys() if k != "vector"}
    for k in ("fields", "tags", "code_refs"):
        d[k] = json.loads(d.get(k) or "[]")
    return d


def get_doc(doc_id: str) -> dict:
    with _lock:
        r = db().execute("SELECT * FROM docs WHERE id=?", (doc_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail=f"no research document {doc_id!r}")
    return _doc_row(r)


def scope_ids(project_id: str | None) -> list[str]:
    """Documents a project sees: its own and the shared ones (project_id '')."""
    with _lock:
        if project_id:
            rows = db().execute("SELECT id FROM docs WHERE project_id IN (?, '') ORDER BY created_at",
                                (project_id,)).fetchall()
        else:
            rows = db().execute("SELECT id FROM docs ORDER BY created_at").fetchall()
    return [r["id"] for r in rows]


def doc_dir(doc_id: str) -> Path:
    return ROOT / doc_id


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    s = re.sub(r"^[0-9_]+", "", s)[:40].strip("_")
    return s or fallback


# =======================================================================================
# Ingest
# =======================================================================================
def create_doc(data: bytes, filename: str, project_id: str, author: str = "operator") -> tuple[dict, bool]:
    """Store an uploaded file and queue it. Returns (doc, created) -- the same file uploaded
    to the same scope again returns the existing document instead of a duplicate."""
    kind = RP.kind_of(filename, data)
    if kind is None:
        raise HTTPException(status_code=415, detail=f"{filename}: only PDF, HTML, Markdown and text documents")
    sha = hashlib.sha256(data).hexdigest()
    with _lock:
        dup = db().execute("SELECT * FROM docs WHERE sha256=? AND project_id=?", (sha, project_id)).fetchone()
        if dup is not None:
            return _doc_row(dup), False
        doc_id = "d_" + secrets.token_hex(5)
        now = time.time()
        ext = {"pdf": ".pdf", "html": ".html", "md": ".md", "txt": ".txt"}[kind]
        d = doc_dir(doc_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"source{ext}").write_bytes(data)
        db().execute(
            "INSERT INTO docs (id, project_id, title, filename, kind, sha256, size_bytes, status, author, created_at, "
            "updated_at) VALUES (?,?,?,?,?,?,?,'queued',?,?,?)",
            (doc_id, project_id, Path(filename).stem, Path(filename).name, kind, sha, len(data), author, now, now))
        db().commit()
    return get_doc(doc_id), True


def _source(doc: dict) -> Path:
    return next(doc_dir(doc["id"]).glob("source.*"))


def _set(doc_id: str, **fields: Any) -> None:
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock:
        db().execute(f"UPDATE docs SET {cols} WHERE id=?", (*fields.values(), doc_id))
        db().commit()


def _unique_package(title: str, doc_id: str) -> str:
    base = _slug(title, "doc")
    with _lock:
        taken = {r[0] for r in db().execute("SELECT package FROM docs WHERE id<>?", (doc_id,)).fetchall()}
    name, n = base, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    return name


def parse_and_store(doc_id: str) -> int:
    """Parse the stored source into chunks (replacing any earlier ones). Returns the count."""
    doc = get_doc(doc_id)
    parsed = RP.parse(_source(doc).read_bytes(), doc["filename"])
    chunks = RP.chunk(parsed)
    fig_dir = doc_dir(doc_id) / "figures"
    fig_dir.mkdir(exist_ok=True)
    for old in fig_dir.iterdir():
        old.unlink()
    used: set[str] = set()
    rows = []
    for seq, b in enumerate(chunks):
        image = ""
        if b.kind == "figure" and b.image:
            image = f"figures/{seq:04d}.{b.image_ext or 'png'}"
            (doc_dir(doc_id) / image).write_bytes(b.image)
        meta = dict(b.meta)
        if b.kind == "code" and meta.get("language") == "python":
            stem = _slug(b.label.rsplit(".", 1)[0], "") if b.label else ""
            mod = stem if stem and stem not in used else f"block_{seq}"
            used.add(mod)
            meta["module"] = mod
        rows.append((doc_id, seq, b.kind, b.section, b.page, b.label, b.text, RP.embed_text(b, parsed.title),
                     image, json.dumps(meta)))
    title = parsed.title if parsed.title and parsed.title != doc["filename"] else doc["title"]
    with _lock:
        db().execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        db().executemany("INSERT INTO chunks (doc_id, seq, kind, section, page, label, text, embed_text, image, meta) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        db().commit()
    _set(doc_id, title=title[:300], pages=parsed.pages, outline=json.dumps(parsed.outline[:300]),
         package=_unique_package(title, doc_id))
    _bump()
    return len(rows)


def embed_doc(doc_id: str) -> str:
    """Embed the document's chunks and ideas. Returns the model used, or '' (lexical only)."""
    emb = RI.embedder()
    if not emb.available:
        _set(doc_id, embed_model="")
        return ""
    with _lock:
        chunks = db().execute("SELECT id, embed_text FROM chunks WHERE doc_id=? ORDER BY seq", (doc_id,)).fetchall()
        ideas = db().execute("SELECT * FROM doc_ideas WHERE doc_id=?", (doc_id,)).fetchall()
    if chunks:
        vecs = emb.embed([c["embed_text"] for c in chunks])
        with _lock:
            db().executemany("UPDATE chunks SET vector=? WHERE id=?",
                             [(RI.to_blob(v), c["id"]) for c, v in zip(chunks, vecs)])
            db().commit()
    if ideas:
        _embed_ideas([_idea_row(r) for r in ideas])
    _set(doc_id, embed_model=emb.name)
    _bump()
    return emb.name


def _embed_ideas(ideas: list[dict]) -> None:
    emb = RI.embedder()
    if not ideas or not emb.available:
        return
    vecs = emb.embed([idea_text(i) for i in ideas])
    with _lock:
        db().executemany("UPDATE doc_ideas SET vector=? WHERE id=?", [(RI.to_blob(v), i["id"]) for i, v in zip(ideas, vecs)])
        db().commit()
    _bump()


async def describe_figures(doc_id: str, model: str) -> int:
    """Ask a vision model what each figure shows, so a chart is found by its content."""
    import base64

    with _lock:
        figs = db().execute("SELECT id, image, label, section, text, meta FROM chunks WHERE doc_id=? AND kind='figure' "
                            "AND image<>'' ORDER BY seq LIMIT ?", (doc_id, FIGURE_DESCRIBE_MAX)).fetchall()
    done = 0
    title = get_doc(doc_id)["title"]
    for f in figs:
        path = doc_dir(doc_id) / f["image"]
        ext = path.suffix.lstrip(".").replace("jpg", "jpeg") or "png"
        url = f"data:image/{ext};base64," + base64.b64encode(path.read_bytes()).decode()
        prompt = (f"This figure is from the research document \"{title}\", section \"{f['section']}\""
                  + (f", captioned \"{f['label']}\"" if f["label"] else "") + ". Describe what it shows for a "
                  "quantitative trader in at most 120 words: chart type, axes and units, the series, and the "
                  "main quantitative takeaway (levels, trends, which bars or deciles stand out). No preamble.")
        try:
            text = await _complete(model, [{"role": "user", "content": [
                {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}],
                FIGURE_MAX_TOKENS, f"research:{doc_id}")
        except Exception as exc:  # noqa: BLE001 -- a figure without a description is still indexed by caption
            logger.warning("figure %s of %s not described by %s: %s", f["id"], doc_id, model, exc)
            meta = json.loads(f["meta"] or "{}") | {"describe_error": str(getattr(exc, "detail", exc))[:300]}
            with _lock:
                db().execute("UPDATE chunks SET meta=? WHERE id=?", (json.dumps(meta), f["id"]))
                db().commit()
            if done == 0 and f is figs[0]:
                break      # the model cannot see images at all: do not repeat the failure 30 times
            continue
        text = (text or "").strip()
        if not text:
            continue
        meta = json.loads(f["meta"] or "{}") | {"described_by": model}
        body = "\n".join(x for x in (f["label"], text) if x)
        with _lock:
            db().execute("UPDATE chunks SET text=?, embed_text=?, meta=? WHERE id=?",
                         (body, f"{title} | {f['section']}\nfigure {body}", json.dumps(meta), f["id"]))
            db().commit()
        done += 1
    _bump()
    return done


# =======================================================================================
# Idea extraction
# =======================================================================================
IDEA_PROMPT = """You are a senior quantitative trading researcher. Below is (part of) a research document. \
Extract the distinct, TESTABLE trading ideas in it: strategies, signals, entry filters, regime conditions, \
position-sizing and risk rules -- including ones the document rejected or only mentions in passing when they \
are still worth testing elsewhere (say so in caveats). Skip generic statements and pure data-quality notes.

Return ONLY a JSON array (no prose, no code fence). Each item:
{"title": "short name",
 "summary": "1-2 sentences: what to trade, when, and why it should work",
 "rules": "precise entry / exit / filter / sizing rules with the parameters the document states",
 "horizon": "holding period and bar size / timing",
 "fields": ["data fields or signals it needs, named as in the document"],
 "evidence": "the numbers the document reports for it: Sharpe, drawdown, hit rate, sample, significance",
 "caveats": "limits the document admits or you see: sample size, multiple testing, look-ahead, data issues",
 "falsify": "the first experiment that would show it does NOT work",
 "code": ["the [code N] markers of the listings that implement it"],
 "tags": ["2-5 short tags"]}
At most 8 items. An empty array is fine when the text holds no trading idea."""


def digest(doc_id: str) -> list[str]:
    """The document as the idea model reads it, split into windows: section headings, prose,
    tables, figure captions, and each code listing as `[code <chunk id>] <file>` + its head."""
    doc = get_doc(doc_id)
    with _lock:
        chunks = [dict(r) for r in db().execute(
            "SELECT id, kind, section, page, label, text FROM chunks WHERE doc_id=? ORDER BY seq", (doc_id,)).fetchall()]
    parts: list[str] = []
    last_section = None
    for c in chunks:
        if c["section"] != last_section:
            parts.append(f"\n## {c['section'] or doc['title']}")
            last_section = c["section"]
        if c["kind"] == "text":
            parts.append(c["text"])
        elif c["kind"] == "table":
            rows = c["text"].splitlines()
            parts.append(f"[table{' ' + c['label'] if c['label'] else ''}]\n" + "\n".join(rows[:40])
                         + (f"\n... {len(rows) - 40} more rows" if len(rows) > 40 else ""))
        elif c["kind"] == "figure":
            parts.append(f"[figure] {c['text'][:600]}")
        elif c["kind"] == "code":
            lines = c["text"].splitlines()
            parts.append(f"[code {c['id']}] {c['label'] or 'listing'} ({len(lines)} lines):\n" + "\n".join(lines[:40])
                         + ("\n..." if len(lines) > 40 else ""))
    windows, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > IDEA_WINDOW_CHARS:
            windows.append(cur)
            cur = ""
        cur += p[:IDEA_WINDOW_CHARS] + "\n"
    if cur.strip():
        windows.append(cur)
    return windows[:IDEA_MAX_WINDOWS]


def parse_ideas(text: str) -> list[dict]:
    """The JSON array out of a model's answer -- tolerant of fences, prose around it, or an
    object wrapping the list."""
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    for cand in (t, t[t.find("["): t.rfind("]") + 1] if "[" in t else "", t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        if not cand:
            continue
        try:
            v = json.loads(cand)
        except ValueError:
            continue
        if isinstance(v, dict):
            v = next((x for x in v.values() if isinstance(x, list)), [v] if v.get("title") else [])
        if isinstance(v, list):
            return [i for i in v if isinstance(i, dict) and str(i.get("title") or "").strip()]
    return []


def _as_text(v: Any) -> str:
    if isinstance(v, list):
        return "; ".join(str(x) for x in v if str(x).strip())
    if isinstance(v, dict):
        return "; ".join(f"{k}: {x}" for k, x in v.items())
    return str(v or "").strip()


def _as_list(v: Any) -> list[str]:
    if isinstance(v, str):
        v = re.split(r"[,;\n]", v)
    return [str(x).strip() for x in (v or []) if str(x).strip()][:30]


def idea_text(i: dict) -> str:
    """One idea as agents and the index read it."""
    lines = [i["title"], i.get("summary") or ""]
    for key, label in (("rules", "Rules"), ("horizon", "Horizon"), ("evidence", "Evidence reported"),
                       ("caveats", "Caveats"), ("falsify", "Falsify first")):
        if i.get(key):
            lines.append(f"{label}: {i[key]}")
    if i.get("fields"):
        lines.append("Fields: " + ", ".join(i["fields"]))
    return "\n".join(x for x in lines if x)


def _similar_title(a: str, b: str) -> bool:
    ta, tb = set(RI.tokens(a)), set(RI.tokens(b))
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.6


def store_ideas(doc_id: str, found: list[dict], model: str) -> list[dict]:
    """Replace the document's ideas with `found` (deduplicated by title)."""
    with _lock:
        code_ids = {r[0] for r in db().execute("SELECT id FROM chunks WHERE doc_id=? AND kind='code'", (doc_id,))}
        # Ideas already sent survive a re-extraction; a new one with the same title would be a twin.
        sent = [r[0] for r in db().execute(
            "SELECT title FROM doc_ideas WHERE doc_id=? AND id IN (SELECT doc_idea_id FROM pushes)", (doc_id,))]
    kept: list[dict] = []
    for raw in found:
        title = _as_text(raw.get("title"))[:200]
        if not title or any(_similar_title(title, t) for t in [*sent, *(k["title"] for k in kept)]):
            continue
        refs = sorted({int(n) for n in re.findall(r"\d+", json.dumps(raw.get("code") or [])) if int(n) in code_ids})
        kept.append({"title": title, "summary": _as_text(raw.get("summary"))[:1500],
                     "rules": _as_text(raw.get("rules"))[:2500], "horizon": _as_text(raw.get("horizon"))[:300],
                     "fields": _as_list(raw.get("fields")), "evidence": _as_text(raw.get("evidence"))[:1500],
                     "caveats": _as_text(raw.get("caveats"))[:1500], "falsify": _as_text(raw.get("falsify"))[:1000],
                     "tags": _as_list(raw.get("tags"))[:8], "code_refs": refs})
        if len(kept) >= MAX_IDEAS_PER_DOC:
            break
    now = time.time()
    with _lock:
        # Ideas already pushed keep their rows (the pushes point at them); only unpushed go.
        db().execute("DELETE FROM doc_ideas WHERE doc_id=? AND id NOT IN (SELECT doc_idea_id FROM pushes)", (doc_id,))
        for seq, i in enumerate(kept):
            cur = db().execute(
                "INSERT INTO doc_ideas (doc_id, seq, title, summary, rules, horizon, fields, evidence, caveats, falsify, "
                "tags, code_refs, model, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, seq, i["title"], i["summary"], i["rules"], i["horizon"], json.dumps(i["fields"]), i["evidence"],
                 i["caveats"], i["falsify"], json.dumps(i["tags"]), json.dumps(i["code_refs"]), model, now))
            i["id"] = cur.lastrowid
        db().commit()
    _embed_ideas(kept)
    _bump()
    return kept


def pick_model(project_id: str, requested: str = "") -> str | None:
    """Who extracts ideas: the requested model, the settings' choice, else the project's first
    idea rung (swarm_policy: its best free model), else the best-rated model loaded."""
    from . import swarm_policy

    loaded = _loaded() if _loaded else []
    names = {m.get("model") for m in loaded}
    for want in (requested, settings()["idea_model"]):
        if want and want in names:
            return want
    project = projects.get(project_id) if project_id else None
    if project:
        ladder = swarm_policy.plan(project, loaded).get("ladder") or []
        if ladder:
            return ladder[0]["model"]
    local = [m for m in loaded if m.get("model") and not m.get("external") and m.get("ready", True)]
    if local:
        return max(local, key=lambda m: m.get("aa") or 0)["model"]
    return None


async def extract_ideas(doc_id: str, model: str | None = None) -> list[dict]:
    if _complete is None:
        raise HTTPException(status_code=503, detail="the research worker is not running")
    doc = get_doc(doc_id)
    model = pick_model(doc["project_id"], model or "")
    if not model:
        raise HTTPException(status_code=409, detail="no model to read the document: load one, or pick one in the "
                                                    "research settings")
    windows = await asyncio.to_thread(digest, doc_id)
    found: list[dict] = []
    for n, w in enumerate(windows, 1):
        part = f" (part {n} of {len(windows)})" if len(windows) > 1 else ""
        text = await _complete(model, [{"role": "user", "content": f"{IDEA_PROMPT}\n\nDOCUMENT: {doc['title']}{part}\n{w}"}],
                               IDEA_MAX_TOKENS, f"research:{doc_id}")
        got = parse_ideas(text)
        if not got and text and n == 1 and len(windows) == 1:
            logger.info("%s: %s found no ideas in %s", doc_id, model, doc["title"])
        found += got
    ideas = await asyncio.to_thread(store_ideas, doc_id, found, model)
    _set(doc_id, idea_model=model)
    return ideas


async def process(doc_id: str, stage: str = "all", model: str | None = None) -> None:
    """The ingest pipeline for one document: parse, describe figures, embed, extract ideas,
    push. `stage` re-runs a tail of it: 'embed' (from embedding) or 'ideas' (ideas + push)."""
    cfg = settings()
    try:
        if stage == "all":
            _active[doc_id] = "parsing"
            _set(doc_id, status="parsing", error="", detail="")
            n = await asyncio.to_thread(parse_and_store, doc_id)
            if cfg["figure_model"] and _complete is not None:
                _active[doc_id] = "figures"
                _set(doc_id, status="figures", detail=f"{n} chunks; describing figures with {cfg['figure_model']}")
                await describe_figures(doc_id, cfg["figure_model"])
        if stage in ("all", "embed"):
            _active[doc_id] = "embedding"
            _set(doc_id, status="embedding")
            used = await asyncio.to_thread(embed_doc, doc_id)
            if not used:
                _set(doc_id, detail=f"lexical search only: {RI.embedder().error or 'no embedding model'}")
        if stage in ("all", "ideas"):
            _active[doc_id] = "ideas"
            _set(doc_id, status="ideas")
            try:
                ideas = await extract_ideas(doc_id, model)
                _set(doc_id, detail=f"{len(ideas)} ideas")
            except HTTPException as exc:
                _set(doc_id, detail=f"ideas not extracted: {exc.detail}")
            except Exception as exc:  # noqa: BLE001 -- the document is still searchable
                logger.exception("idea extraction for %s failed", doc_id)
                _set(doc_id, detail=f"ideas not extracted: {type(exc).__name__}: {exc}"[:500])
            if cfg["auto_push"]:
                await push_for_doc(doc_id, cfg["push_on_ingest"])
        _set(doc_id, status="ready")
    except Exception as exc:  # noqa: BLE001
        logger.exception("research document %s failed", doc_id)
        _set(doc_id, status="error", error=f"{type(exc).__name__}: {exc}"[:1000])
    finally:
        _active.pop(doc_id, None)
        _bump()


def enqueue(doc_id: str, stage: str = "all", model: str | None = None) -> None:
    if _queue is None:
        raise HTTPException(status_code=503, detail="the research worker is not running")
    _queue.put_nowait((doc_id, stage, model))


# =======================================================================================
# Search
# =======================================================================================
def corpus(project_id: str | None) -> RI.Corpus:
    key = project_id or "*"
    hit = _corpora.get(key)
    if hit and hit[0] == _version:
        return hit[1]
    ids = scope_ids(project_id)
    rows: list[dict] = []
    if ids:
        marks = ",".join("?" * len(ids))
        with _lock:
            docs = {r["id"]: dict(r) for r in db().execute(f"SELECT id, title, package FROM docs WHERE id IN ({marks})", ids)}
            for r in db().execute(f"SELECT id, doc_id, kind, section, page, label, text, embed_text, image, meta, vector "
                                  f"FROM chunks WHERE doc_id IN ({marks})", ids):
                rows.append({"key": f"c{r['id']}", "chunk_id": r["id"], "doc_id": r["doc_id"], "kind": r["kind"],
                             "section": r["section"], "page": r["page"], "label": r["label"],
                             "body": r["text"], "text": r["embed_text"], "image": r["image"],
                             "meta": json.loads(r["meta"] or "{}"), "vector": RI.from_blob(r["vector"]),
                             "doc_title": docs[r["doc_id"]]["title"], "package": docs[r["doc_id"]]["package"]})
            for r in db().execute(f"SELECT * FROM doc_ideas WHERE doc_id IN ({marks}) AND dismissed=0", ids):
                i = _idea_row(r)
                rows.append({"key": f"i{r['id']}", "idea_id": r["id"], "doc_id": r["doc_id"], "kind": "idea",
                             "section": "", "page": None, "label": i["title"], "body": idea_text(i),
                             "text": idea_text(i), "image": "", "meta": {"code_refs": i["code_refs"]},
                             "vector": RI.from_blob(r["vector"]), "doc_title": docs[r["doc_id"]]["title"],
                             "package": docs[r["doc_id"]]["package"]})
    c = RI.Corpus(rows)
    _corpora[key] = (_version, c)
    return c


def _query_vec(text: str) -> Any:
    emb = RI.embedder()
    if not emb.available:
        return None
    return emb.embed([text], query=True)[0]


def search(project_id: str | None, query: str, k: int = 8, kinds: list[str] | None = None,
           doc_id: str | None = None, snippet: int = 700) -> list[dict]:
    c = corpus(project_id)
    keep = None
    if kinds or doc_id:
        ks = set(kinds or [])
        keep = lambda r: (not ks or r["kind"] in ks) and (not doc_id or r["doc_id"] == doc_id)  # noqa: E731
    hits = c.search(query, _query_vec(query), k=max(1, min(k, 40)), keep=keep)
    return [_hit(h, snippet) for h in hits]


def _hit(h: dict, snippet: int) -> dict:
    out = {"doc_id": h["doc_id"], "doc": h["doc_title"], "kind": h["kind"], "section": h["section"],
           "page": h["page"], "label": h["label"], "score": h["score"], "cosine": h.get("cosine"),
           "text": h["body"][:snippet] + ("…" if len(h["body"]) > snippet else "")}
    if h.get("chunk_id"):
        out["chunk_id"] = h["chunk_id"]
    if h.get("idea_id"):
        out["idea_id"] = h["idea_id"]
    if h["kind"] == "figure" and h["image"]:
        out["image_url"] = f"/api/research/chunks/{h['chunk_id']}/image"
    if h["kind"] == "code" and h["meta"].get("module"):
        out["import"] = f"from research.{h['package']} import {h['meta']['module']}"
    return out


# =======================================================================================
# Ideas -> the objectives' idea stream
# =======================================================================================
def _objective_text(obj: dict) -> str:
    m = obj.get("metric") or {}
    task = (m.get("task_info") or {}) if isinstance(m, dict) else {}
    return " ".join(str(x) for x in (obj.get("title"), obj.get("description"), m.get("kind") if isinstance(m, dict) else "",
                                     obj.get("dataset"), task.get("title"), task.get("description")) if x)


def _objective_vec(obj: dict) -> Any:
    text = _objective_text(obj)
    hit = _obj_vecs.get(obj["id"])
    if hit and hit[0] == text:
        return hit[1]
    v = _query_vec(text)
    _obj_vecs[obj["id"]] = (text, v)
    return v


def relevant_ideas(obj: dict, limit: int = 20, include_pushed: bool = True) -> list[dict]:
    """The research ideas in the objective's scope, most relevant first, each with its
    relevance and whether (and when) it was sent to this objective."""
    ids = scope_ids(obj["project_id"])
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    with _lock:
        rows = db().execute(f"SELECT i.*, d.title AS doc_title, d.package AS package FROM doc_ideas i "
                            f"JOIN docs d ON d.id=i.doc_id WHERE i.doc_id IN ({marks}) AND i.dismissed=0", ids).fetchall()
        pushed = {r["doc_idea_id"]: dict(r) for r in db().execute(
            "SELECT * FROM pushes WHERE objective_id=?", (obj["id"],)).fetchall()}
    if not rows:
        return []
    ov = _objective_vec(obj)
    otext = _objective_text(obj)
    out = []
    for r in rows:
        i = _idea_row(r) | {"doc_title": r["doc_title"], "package": r["package"]}
        i["relevance"] = round(RI.similarity(RI.from_blob(r["vector"]), ov, idea_text(i), otext), 4)
        i["dense"] = ov is not None and r["vector"] is not None
        i["pushed"] = pushed.get(r["id"])
        if include_pushed or not i["pushed"]:
            out.append(i)
    out.sort(key=lambda i: -i["relevance"])
    return out[:limit]


def _code_lines(idea: dict) -> list[str]:
    if not idea.get("code_refs"):
        return []
    marks = ",".join("?" * len(idea["code_refs"]))
    with _lock:
        rows = db().execute(f"SELECT c.id, c.label, c.meta, d.package FROM chunks c JOIN docs d ON d.id=c.doc_id "
                            f"WHERE c.id IN ({marks})", idea["code_refs"]).fetchall()
    out = []
    for r in rows:
        mod = json.loads(r["meta"] or "{}").get("module")
        how = f"`from research.{r['package']} import {mod}`" if mod else "read it with research_get"
        out.append(f"Code: research_get(chunk={r['id']}) -> {r['label'] or 'listing'}; in a script {how}")
    return out


def push_text(idea: dict) -> str:
    """The idea as it enters an objective's idea stream."""
    return "\n".join([
        f"RESEARCH IDEA from \"{idea['doc_title']}\" (research library {idea['doc_id']}). The evidence is the "
        "document's own, on its data -- not verified here. Adapt it to this objective's data and scoring, "
        "and run the falsifying experiment first.",
        idea_text(idea), *_code_lines(idea)])


def push(obj: dict, idea: dict, by: str = "auto") -> dict:
    """Write one research idea into an objective's idea stream (escalation's ideas table)."""
    from . import escalation
    from . import objectives as obj_mod

    escalation._ensure_table()
    with obj_mod._lock:
        n = obj_mod.db().execute("SELECT count(*) FROM candidates WHERE objective_id=?", (obj["id"],)).fetchone()[0]
        cur = obj_mod.db().execute(
            "INSERT INTO ideas (objective_id, ts, model, rung, text, candidates_at, trigger) VALUES (?,?,?,?,?,?,?)",
            (obj["id"], time.time(), f"research: {idea['doc_title'][:80]}", 0, push_text(idea)[:8000], n, "research"))
        obj_mod.db().commit()
    with _lock:
        db().execute("INSERT OR REPLACE INTO pushes (objective_id, doc_idea_id, idea_row_id, ts, relevance, by) "
                     "VALUES (?,?,?,?,?,?)", (obj["id"], idea["id"], cur.lastrowid, time.time(), idea.get("relevance"), by))
        db().commit()
    logger.info("research idea %s (%s) sent to objective %s by %s", idea["id"], idea["title"], obj["id"], by)
    return {"idea_row_id": cur.lastrowid, "objective_id": obj["id"], "doc_idea_id": idea["id"]}


def _eligible(idea: dict, cfg: dict) -> bool:
    return idea["relevance"] >= (cfg["min_relevance"] if idea["dense"] else cfg["min_overlap"])


def push_next(obj: dict, n: int = 1, doc_id: str | None = None) -> list[dict]:
    """Push up to n unpushed, relevant-enough ideas (optionally from one document)."""
    cfg = settings()
    out = []
    for i in relevant_ideas(obj, limit=200, include_pushed=False):
        if len(out) >= n:
            break
        if doc_id and i["doc_id"] != doc_id:
            continue
        if _eligible(i, cfg):
            out.append(push(obj, i))
    return out


async def _running(project_id: str | None = None) -> list[dict]:
    from . import objectives as obj_mod

    plist = [projects.get(project_id)] if project_id else await asyncio.to_thread(projects.list_projects)
    out = []
    for p in plist:
        if not p or not p.get("swarm_enabled", True):
            continue
        for o in (await obj_mod.list_objectives(p["id"], "running")).get("objectives", []):
            out.append(obj_mod.get_objective(o["id"]))
    return out


async def push_for_doc(doc_id: str, n: int) -> int:
    doc = get_doc(doc_id)
    pushed = 0
    for obj in await _running(doc["project_id"] or None):
        try:
            pushed += len(await asyncio.to_thread(push_next, obj, n, doc_id))
        except Exception:  # noqa: BLE001 -- one objective must not stop the others
            logger.exception("pushing %s ideas to %s failed", doc_id, obj["id"])
    return pushed


def _last_push(oid: str) -> float:
    with _lock:
        return db().execute("SELECT max(ts) FROM pushes WHERE objective_id=? AND by='auto'", (oid,)).fetchone()[0] or 0


async def drip() -> int:
    """One more idea per running objective, at most every push_gap_minutes."""
    cfg = settings()
    if not cfg["auto_push"] or not scope_ids(None):
        return 0
    n = 0
    for obj in await _running():
        if time.time() - _last_push(obj["id"]) < cfg["push_gap_minutes"] * 60:
            continue
        try:
            n += len(await asyncio.to_thread(push_next, obj, 1))
        except Exception:  # noqa: BLE001
            logger.exception("research drip to %s failed", obj["id"])
    return n


# =======================================================================================
# What agents and idea models get
# =======================================================================================
def code_files(project_id: str) -> dict[str, str]:
    """{path: source} of every Python listing in scope, for a sandbox run:
    .ft/research/<package>/<module>.py, importable as `from research.<package> import <module>`."""
    ids = scope_ids(project_id)
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    with _lock:
        rows = db().execute(f"SELECT c.doc_id, c.text, c.meta, d.package, d.title FROM chunks c JOIN docs d ON d.id=c.doc_id "
                            f"WHERE c.kind='code' AND c.doc_id IN ({marks}) AND d.status='ready' ORDER BY c.doc_id, c.seq",
                            ids).fetchall()
    files: dict[str, str] = {}
    shipped: dict[str, int] = {}
    for r in rows:
        mod = json.loads(r["meta"] or "{}").get("module")
        if not mod or not r["package"] or shipped.get(r["doc_id"], 0) + len(r["text"]) > CODE_SHIP_MAX_CHARS:
            continue
        shipped[r["doc_id"]] = shipped.get(r["doc_id"], 0) + len(r["text"])
        files.setdefault(".ft/research/__init__.py", '"""Code listings from the research library (read-only)."""\n')
        files.setdefault(f".ft/research/{r['package']}/__init__.py", f'"""Code from: {r["title"]}"""\n')
        files[f".ft/research/{r['package']}/{mod}.py"] = (
            f"# From the research document \"{r['title']}\" ({r['doc_id']}), extracted from its text:\n"
            f"# check it before relying on it (a PDF can lose line breaks).\n{r['text']}\n")
    return files


def brief(project_id: str) -> dict | None:
    """The research library in an agent's brief: which documents exist and how to use them."""
    ids = scope_ids(project_id)
    if not ids:
        return None
    marks = ",".join("?" * len(ids))
    with _lock:
        docs = db().execute(f"SELECT d.id, d.title, d.package, d.status, "
                            f"(SELECT count(*) FROM doc_ideas i WHERE i.doc_id=d.id AND i.dismissed=0) AS ideas, "
                            f"(SELECT count(*) FROM chunks c WHERE c.doc_id=d.id AND c.kind='code') AS code "
                            f"FROM docs d WHERE d.id IN ({marks}) ORDER BY d.created_at DESC LIMIT 15", ids).fetchall()
    return {"documents": [dict(d) for d in docs], "total": len(ids)}


def prompt_lines(obj: dict, query: str = "", k_ideas: int = 4, k_chunks: int = 4) -> list[str]:
    """Research findings for an idea model's prompt (escalation): the ideas most relevant to
    the objective, then the passages most relevant to it and to what the team is doing."""
    try:
        ideas = [i for i in relevant_ideas(obj, limit=k_ideas * 3) if _eligible(i, settings())][:k_ideas]
        hits = search(obj["project_id"], f"{_objective_text(obj)} {query}".strip(), k=k_chunks,
                      kinds=["text", "table", "code", "figure"], snippet=500)
    except Exception:  # noqa: BLE001 -- ideas are still asked for without the library
        logger.exception("research context for %s failed", obj.get("id"))
        return []
    if not ideas and not hits:
        return []
    lines = ["", "FROM THE RESEARCH LIBRARY -- documents the operator added. Their numbers are the authors' own, on "
             "their data; treat each as a hypothesis to adapt and test here, not a result:"]
    for i in ideas:
        tried = f" (already sent to this objective {time.strftime('%Y-%m-%d', time.localtime(i['pushed']['ts']))})" \
            if i.get("pushed") else ""
        lines.append(f"- IDEA \"{i['title']}\" from \"{i['doc_title']}\"{tried}: {i['summary'][:400]} Rules: "
                     f"{i['rules'][:400]} Evidence: {i['evidence'][:250]} Caveats: {i['caveats'][:200]}")
        lines += [f"  {c}" for c in _code_lines(i)[:2]]
    for h in hits:
        where = f"p.{h['page']}, " if h.get("page") else ""
        lines.append(f"- {h['kind'].upper()} from \"{h['doc']}\" ({where}{h['section'][:80]}): "
                     + re.sub(r"\s+", " ", h["text"])[:450])
    return lines


# =======================================================================================
# Background worker
# =======================================================================================
async def run(complete: CompleteFn, loaded: LoadedFn) -> None:
    """Process queued documents one at a time (embedding is CPU-bound) and drip ideas."""
    global _complete, _loaded, _queue
    _complete, _loaded = complete, loaded
    _queue = asyncio.Queue()
    ROOT.mkdir(parents=True, exist_ok=True)
    # A document caught mid-pipeline by a restart starts over.
    with _lock:
        stale = [r[0] for r in db().execute("SELECT id FROM docs WHERE status NOT IN ('ready','error')")]
    for d in stale:
        _queue.put_nowait((d, "all", None))

    async def worker() -> None:
        while True:
            doc_id, stage, model = await _queue.get()
            try:
                get_doc(doc_id)
                await process(doc_id, stage, model)
            except HTTPException:
                pass            # deleted while queued
            except Exception:  # noqa: BLE001
                logger.exception("research worker failed on %s", doc_id)

    task = asyncio.create_task(worker())
    try:
        while True:
            await asyncio.sleep(TICK_S)
            try:
                await drip()
            except Exception:  # noqa: BLE001
                logger.exception("research drip failed")
    finally:
        task.cancel()


# =======================================================================================
# API
# =======================================================================================
def _counts(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    out: dict[str, dict] = {i: {"chunks": {}, "ideas": 0, "pushed": 0} for i in ids}
    with _lock:
        for r in db().execute(f"SELECT doc_id, kind, count(*) FROM chunks WHERE doc_id IN ({marks}) GROUP BY doc_id, kind", ids):
            out[r[0]]["chunks"][r[1]] = r[2]
        for r in db().execute(f"SELECT doc_id, count(*), sum(dismissed=0) FROM doc_ideas WHERE doc_id IN ({marks}) "
                              f"GROUP BY doc_id", ids):
            out[r[0]]["ideas"] = r[2] or 0
        for r in db().execute(f"SELECT i.doc_id, count(*) FROM pushes p JOIN doc_ideas i ON i.id=p.doc_idea_id "
                              f"WHERE i.doc_id IN ({marks}) GROUP BY i.doc_id", ids):
            out[r[0]]["pushed"] = r[1]
    return out


@router.get("/research/status")
async def status() -> dict:
    emb = RI.embedder()
    return {"running": _queue is not None, "queued": _queue.qsize() if _queue else 0, "active": dict(_active),
            "embedding": {"model": emb.name, "device": emb.device, "loaded": emb._model is not None,
                          "error": emb.error},
            "settings": settings()}


@router.get("/research/docs")
async def list_docs(project_id: str | None = None) -> dict:
    ids = await asyncio.to_thread(scope_ids, project_id)
    with _lock:
        rows = [_doc_row(r) for r in db().execute(
            f"SELECT * FROM docs WHERE id IN ({','.join('?' * len(ids))}) ORDER BY created_at DESC", ids)] if ids else []
    counts = await asyncio.to_thread(_counts, ids)
    return {"docs": [{**{k: v for k, v in d.items() if k != "outline"}, **counts.get(d["id"], {}),
                      "stage": _active.get(d["id"])} for d in rows]}


@router.post("/research/docs")
async def upload(request: Request) -> dict:
    """Multipart: one or more `file` parts, `project_id` ('' or absent = shared by all projects)."""
    form = await request.form()
    project_id = str(form.get("project_id") or "")
    if project_id and projects.get(project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    out = []
    for f in form.getlist("file"):
        if not hasattr(f, "read"):
            continue
        data = await f.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{f.filename}: larger than {MAX_UPLOAD_BYTES // 2**20} MB")
        doc, created = await asyncio.to_thread(create_doc, data, str(f.filename or "document"), project_id,
                                               str(form.get("author") or "operator"))
        if created:
            enqueue(doc["id"])
        out.append({**doc, "created": created})
    if not out:
        raise HTTPException(status_code=400, detail="no file in the upload")
    return {"docs": out}


@router.get("/research/docs/{doc_id}")
async def read_doc(doc_id: str, compact: bool = False, objective_id: str | None = None) -> dict:
    doc = get_doc(doc_id)
    with _lock:
        chunks = [dict(r) for r in db().execute(
            "SELECT id, seq, kind, section, page, label, text, image, meta FROM chunks WHERE doc_id=? ORDER BY seq",
            (doc_id,))]
        ideas = [_idea_row(r) for r in db().execute("SELECT * FROM doc_ideas WHERE doc_id=? ORDER BY seq", (doc_id,))]
        pushes = [dict(r) for r in db().execute(
            "SELECT p.* FROM pushes p JOIN doc_ideas i ON i.id=p.doc_idea_id WHERE i.doc_id=?", (doc_id,))]
    for c in chunks:
        c["meta"] = json.loads(c["meta"] or "{}")
        if c["image"]:
            c["image_url"] = f"/api/research/chunks/{c['id']}/image"
        if c["kind"] == "code" and c["meta"].get("module"):
            c["import"] = f"from research.{doc['package']} import {c['meta']['module']}"
    for i in ideas:
        i["pushes"] = [p for p in pushes if p["doc_idea_id"] == i["id"]]
    if compact:
        # For an agent: the map of the document, not its full text (research_get a chunk for that).
        return {"id": doc["id"], "title": doc["title"], "pages": doc["pages"],
                "outline": [o["title"] for o in doc["outline"][:60]],
                "ideas": [{"id": i["id"], "text": idea_text(i), "code_chunks": i["code_refs"]} for i in ideas if not i["dismissed"]],
                "code": [{"chunk": c["id"], "label": c["label"], "language": c["meta"].get("language"),
                          "lines": c["meta"].get("lines"), "import": c.get("import")} for c in chunks if c["kind"] == "code"],
                "tables": [{"chunk": c["id"], "label": c["label"], "section": c["section"], "page": c["page"],
                            "head": "\n".join(c["text"].splitlines()[:4])} for c in chunks if c["kind"] == "table"][:40],
                "figures": [{"chunk": c["id"], "caption": c["text"][:300], "page": c["page"]}
                            for c in chunks if c["kind"] == "figure"][:40],
                "note": "research_get(chunk=<id>) returns any chunk in full; the numbers are the document's own"}
    return {**doc, "stage": _active.get(doc_id), "chunks": chunks, "ideas": ideas}


@router.get("/research/docs/{doc_id}/source")
async def doc_source(doc_id: str) -> FileResponse:
    doc = get_doc(doc_id)
    media = {"pdf": "application/pdf", "html": "text/html", "md": "text/markdown", "txt": "text/plain"}[doc["kind"]]
    # An uploaded HTML page is served from the console's origin: sandboxed (no scripts, an
    # opaque origin) so a report's JavaScript can never act with the console's permissions.
    headers = {"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"}
    return FileResponse(_source(doc), media_type=media, filename=doc["filename"], headers=headers,
                        content_disposition_type="inline")


@router.get("/research/chunks/{chunk_id}")
async def read_chunk(chunk_id: int) -> dict:
    with _lock:
        r = db().execute("SELECT c.*, d.title AS doc_title, d.package FROM chunks c JOIN docs d ON d.id=c.doc_id "
                         "WHERE c.id=?", (chunk_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail=f"no research chunk {chunk_id}")
    c = {k: r[k] for k in r.keys() if k not in ("vector", "embed_text")}
    c["meta"] = json.loads(c["meta"] or "{}")
    if c["kind"] == "code" and c["meta"].get("module"):
        c["import"] = f"from research.{c['package']} import {c['meta']['module']}"
    if c["image"]:
        c["image_url"] = f"/api/research/chunks/{chunk_id}/image"
    return c


@router.get("/research/chunks/{chunk_id}/image")
async def chunk_image(chunk_id: int) -> Response:
    with _lock:
        r = db().execute("SELECT doc_id, image FROM chunks WHERE id=?", (chunk_id,)).fetchone()
    if r is None or not r["image"]:
        raise HTTPException(status_code=404, detail="no image")
    path = doc_dir(r["doc_id"]) / r["image"]
    ext = path.suffix.lstrip(".").replace("jpg", "jpeg")
    return Response(path.read_bytes(), media_type=f"image/{ext}", headers={"Cache-Control": "max-age=86400"})


@router.post("/research/docs/{doc_id}/delete")
async def delete_doc(doc_id: str) -> dict:
    """Remove a document, its chunks, figures and ideas. Ideas already sent to objectives stay
    in their idea streams (they are the record of what agents were told)."""
    import shutil

    get_doc(doc_id)
    with _lock:
        db().execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        db().execute("DELETE FROM pushes WHERE doc_idea_id IN (SELECT id FROM doc_ideas WHERE doc_id=?)", (doc_id,))
        db().execute("DELETE FROM doc_ideas WHERE doc_id=?", (doc_id,))
        db().execute("DELETE FROM docs WHERE id=?", (doc_id,))
        db().commit()
    shutil.rmtree(doc_dir(doc_id), ignore_errors=True)
    _bump()
    return {"deleted": doc_id}


class Reprocess(BaseModel):
    stage: str = Field("all", pattern="^(all|embed|ideas)$")
    model: str | None = Field(None, max_length=300)


@router.post("/research/docs/{doc_id}/reprocess")
async def reprocess(doc_id: str, req: Reprocess) -> dict:
    get_doc(doc_id)
    if doc_id in _active:
        raise HTTPException(status_code=409, detail=f"already {_active[doc_id]}")
    enqueue(doc_id, req.stage, req.model)      # first: a refused enqueue must not leave it "queued"
    _set(doc_id, status="queued")
    return {"queued": doc_id, "stage": req.stage}


class Move(BaseModel):
    project_id: str = Field("", max_length=200)


@router.post("/research/docs/{doc_id}/scope")
async def move_doc(doc_id: str, req: Move) -> dict:
    """Share a document with every project ('') or give it to one."""
    get_doc(doc_id)
    if req.project_id and projects.get(req.project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    _set(doc_id, project_id=req.project_id)
    _bump()
    return get_doc(doc_id)


class SearchReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    project_id: str | None = Field(None, max_length=200)
    kinds: list[str] | None = None
    doc_id: str | None = None
    k: int = Field(8, ge=1, le=40)


@router.post("/research/search")
async def search_api(req: SearchReq) -> dict:
    hits = await asyncio.to_thread(search, req.project_id, req.query, req.k, req.kinds, req.doc_id)
    return {"hits": hits, "dense": RI.embedder()._model is not None}


class PushReq(BaseModel):
    objective_id: str = Field(..., max_length=200)


@router.post("/research/ideas/{idea_id}/push")
async def push_api(idea_id: int, req: PushReq) -> dict:
    """The operator sends one research idea to an objective now, whatever its relevance."""
    from . import objectives as obj_mod

    obj = obj_mod.get_objective(req.objective_id)
    idea = next((i for i in await asyncio.to_thread(relevant_ideas, obj, 10_000) if i["id"] == idea_id), None)
    if idea is None:
        raise HTTPException(status_code=404, detail="that idea is not in this objective's project")
    return await asyncio.to_thread(push, obj, idea, "operator")


class DismissReq(BaseModel):
    dismissed: bool = True


@router.post("/research/ideas/{idea_id}/dismiss")
async def dismiss(idea_id: int, req: DismissReq) -> dict:
    """A dismissed idea is never pushed and leaves search; it can be restored."""
    with _lock:
        n = db().execute("UPDATE doc_ideas SET dismissed=? WHERE id=?", (int(req.dismissed), idea_id)).rowcount
        db().commit()
    if not n:
        raise HTTPException(status_code=404, detail="no such idea")
    _bump()
    return {"id": idea_id, "dismissed": req.dismissed}


@router.get("/objectives/{oid}/research")
async def objective_research(oid: str, limit: int = 20) -> dict:
    """Research ideas for one objective, most relevant first, with what was already sent."""
    from . import objectives as obj_mod

    obj = obj_mod.get_objective(oid)
    ideas = await asyncio.to_thread(relevant_ideas, obj, max(1, min(limit, 100)))
    cfg = settings()
    return {"ideas": [{k: v for k, v in i.items()} | {"eligible": _eligible(i, cfg)} for i in ideas],
            "settings": cfg}


class SettingsReq(BaseModel):
    idea_model: str | None = Field(None, max_length=300)
    figure_model: str | None = Field(None, max_length=300)
    auto_push: bool | None = None
    push_on_ingest: int | None = Field(None, ge=0, le=10)
    push_gap_minutes: int | None = Field(None, ge=5, le=10_080)
    min_relevance: float | None = Field(None, ge=0, le=1)
    min_overlap: float | None = Field(None, ge=0, le=1)


@router.put("/research/settings")
async def put_settings(req: SettingsReq) -> dict:
    return save_settings(req.model_dump(exclude_none=True))
