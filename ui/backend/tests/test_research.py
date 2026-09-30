"""The research library: documents are parsed into text/table/figure/code chunks, searched,
mined for ideas, and those ideas reach the objectives' idea stream without disturbing the
stuck ladder -- and the documents' code ships into sandbox runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

import numpy as np
import pytest

from app import escalation, external
from app import objectives as O
from app import research as R
from app import research_index as RI
from app import research_parse as RP

from research_fixtures import CODE, make_html, make_pdf


# =======================================================================================
# Parsing
# =======================================================================================
def _chunks(data: bytes, name: str) -> tuple[RP.Parsed, list[RP.Block]]:
    p = RP.parse(data, name)
    return p, RP.chunk(p)


def test_pdf_keeps_headings_tables_figures_and_code():
    p, chunks = _chunks(make_pdf(), "odpd.pdf")
    assert p.kind == "pdf" and p.pages == 3
    assert p.title == "ODPD Overnight Dealer-Positioning Drift"
    assert [o["title"] for o in p.outline][1:] == ["1. Backtests", "2. Final backtest script", "3. Robustness"]

    table = next(c for c in chunks if c.kind == "table")
    # Identifiers survive (PyMuPDF's own extract() splits the underscores off).
    assert "| B0_always_long | 1.44 | 1.59 |" in table.text
    assert table.section.endswith("1. Backtests")

    fig = next(c for c in chunks if c.kind == "figure")
    assert fig.image and fig.image[:4] == b"\x89PNG"
    assert fig.label.startswith("Figure 1")

    code = [c for c in chunks if c.kind == "code"]
    assert len(code) == 1, "the listing continues across the page break"
    assert code[0].text == "\n".join(CODE), "indentation and the blank line are rebuilt"
    assert code[0].label == "odpd_strategy.py" and code[0].meta["language"] == "python"

    text = " ".join(c.text for c in chunks if c.kind == "text")
    assert "placebo p-value is 0.008" in text
    assert "page 1/3" not in text, "running footers are dropped"


def test_html_keeps_captions_images_and_labelled_code():
    p, chunks = _chunks(make_html(), "odpd.html")
    assert p.title == "ODPD research"
    table = next(c for c in chunks if c.kind == "table")
    assert table.label == "Overnight family" and "| C6_event_or_wallpin_x_gex | 3.019 | 4.756 |" in table.text
    fig = next(c for c in chunks if c.kind == "figure")
    assert fig.image[:4] == b"\x89PNG" and fig.label.startswith("Figure 2")
    code = next(c for c in chunks if c.kind == "code")
    assert code.text == "\n".join(CODE)
    assert code.label == "odpd_strategy.py" and code.meta["language"] == "python"
    assert not any("var x" in c.text for c in chunks), "scripts are not content"
    assert code.section == "ODPD — Overnight Dealer-Positioning Drift > 12. Final backtest script"


def test_markdown_fences_and_tables():
    md = b"# Study\n\nSee signals.py below.\n\n```python\ndef s(df):\n    return df\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    p, chunks = _chunks(md, "study.md")
    kinds = [c.kind for c in chunks]
    assert kinds == ["text", "code", "table"]
    assert chunks[1].label == "signals.py" and chunks[1].text == "def s(df):\n    return df"
    assert "| 1 | 2 |" in chunks[2].text


def test_unsupported_type_is_refused():
    with pytest.raises(ValueError):
        RP.parse(b"\x00\x01binary", "data.bin")


def test_prose_is_chunked_per_section():
    blocks = [RP.Block("text", "a " * 500, "S1"), RP.Block("text", "b " * 500, "S1"), RP.Block("text", "c", "S2")]
    out = RP.chunk(RP.Parsed("t", "md", blocks), limit=1400)
    assert [b.section for b in out] == ["S1", "S1", "S2"]


# =======================================================================================
# Ideas out of a model's answer
# =======================================================================================
@pytest.mark.parametrize("answer", [
    '[{"title": "Overnight drift"}]',
    '```json\n[{"title": "Overnight drift"}]\n```',
    'Here are the ideas:\n[{"title": "Overnight drift"}]\nHope this helps.',
    '{"ideas": [{"title": "Overnight drift"}]}',
])
def test_parse_ideas_tolerates_wrapping(answer):
    assert [i["title"] for i in R.parse_ideas(answer)] == ["Overnight drift"]


def test_parse_ideas_garbage_is_empty():
    assert R.parse_ideas("I could not find any.") == []


# =======================================================================================
# Search
# =======================================================================================
def test_bm25_finds_identifiers():
    c = RI.Corpus([{"key": "a", "text": "the wall_pos signal ranks spot in the band", "vector": None},
                   {"key": "b", "text": "overnight drift before macro releases", "vector": None}])
    assert [h["key"] for h in c.search("wall_pos", None, k=1)] == ["a"]
    assert [h["key"] for h in c.search("wall position", None, k=1)] == ["a"], "identifiers also match their parts"
    assert c.search("zzz", None) == []


class FakeEmbedder:
    """Deterministic bag-of-words vectors: related texts are close, like a real model."""
    name, device, error = "fake-embed", "cpu", None
    _model = object()
    available = True

    def embed(self, texts, query=False, batch=32):
        out = np.zeros((len(texts), 256), np.float32)
        for i, t in enumerate(texts):
            for tok in RI.tokens(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


# =======================================================================================
# The library end to end, into the idea stream
# =======================================================================================
@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(external, "CONFIG_PATH", tmp_path / "external.json")
    monkeypatch.setattr(R, "DB_PATH", tmp_path / "research.sqlite3")
    monkeypatch.setattr(R, "ROOT", tmp_path / "research")
    monkeypatch.setattr(R, "_conn", None)
    monkeypatch.setattr(R, "_corpora", {})
    monkeypatch.setattr(R, "_obj_vecs", {})
    monkeypatch.setattr(R.projects, "get", lambda pid: {"id": pid, "name": pid} if pid == "p1" else None)
    RI.set_embedder(FakeEmbedder())
    now = time.time()
    O.db().execute(
        "INSERT INTO objectives (id, project_id, title, description, metric, split_date, status, created_at, updated_at) "
        "VALUES ('o1', 'p1', 'Best SPY overnight strategy', 'Trade SPY overnight from dealer gamma and macro events', "
        "?, '2024-07-19', 'running', ?, ?)",
        (json.dumps({"kind": "sharpe", "higher_is_better": True, "price_column": "Close", "cost_bps": 2}), now, now))
    O.db().commit()
    escalation._ensure_table()
    yield
    RI.set_embedder(None)
    O.db().close()
    R.db().close()


IDEAS_JSON = json.dumps([
    {"title": "Overnight drift on macro nights and benign dealer gamma",
     "summary": "Hold SPY overnight before tier-1 macro releases, or when dealers are long gamma and spot is off the call wall.",
     "rules": "Long at 15:59 if event_next or (wall_rank < 2/3 and pin_rank < 2/3 and GEX > 0); exit 09:30",
     "horizon": "overnight", "fields": ["GEX", "CallWall_MaxStrike", "PinTrend_ProbPin"],
     "evidence": "Sharpe 3.02, Burke 4.76, 95 trades", "caveats": "16 months; DSR 0.93",
     "falsify": "Random night selection of the same size does as well", "code": ["[code CODEID] odpd_strategy.py"],
     "tags": ["overnight", "gex"]},
    {"title": "Bike rental seasonality", "summary": "Rent more bikes in summer.", "rules": "n/a"},
])


def _ingest(data=None, name="odpd.pdf"):
    doc, created = R.create_doc(data or make_pdf(), name, "p1")
    assert created
    R.parse_and_store(doc["id"])
    R.embed_doc(doc["id"])
    code_id = R.db().execute("SELECT id FROM chunks WHERE doc_id=? AND kind='code'", (doc["id"],)).fetchone()[0]
    R.store_ideas(doc["id"], R.parse_ideas(IDEAS_JSON.replace("CODEID", str(code_id))), "m")
    return R.get_doc(doc["id"]), code_id


def test_same_file_twice_is_one_document(lib):
    data = make_pdf()   # built once: every build stamps a new creation date
    d1, created1 = R.create_doc(data, "a.pdf", "p1")
    d2, created2 = R.create_doc(data, "b.pdf", "p1")
    assert created1 and not created2 and d1["id"] == d2["id"]


def test_search_code_and_tables(lib):
    doc, code_id = _ingest()
    assert doc["embed_model"] == "fake-embed" and doc["package"] == "odpd_overnight_dealer_positioning_drift"
    hits = R.search("p1", "causal_rank", k=3, kinds=["code"])
    assert hits[0]["chunk_id"] == code_id
    assert hits[0]["import"] == f"from research.{doc['package']} import odpd_strategy"
    table = R.search("p1", "C6_event_or_wallpin_x_gex Sharpe", k=1, kinds=["table"])[0]
    assert "4.76" in table["text"]
    assert R.search("other-project", "causal_rank") == [], "a project's documents stay in that project"


def test_code_ships_into_the_sandbox(lib):
    doc, _ = _ingest()
    R._set(doc["id"], status="ready")
    files = R.code_files("p1")
    src = files[f".ft/research/{doc['package']}/odpd_strategy.py"]
    assert "def causal_rank" in src and ".ft/research/__init__.py" in files
    ns: dict = {}
    exec(compile(src, "odpd_strategy.py", "exec"), ns)  # the rebuilt listing is valid Python
    assert ns["rule"](False, 0.1, 0.2, 1.0) is True


def test_ideas_reach_the_idea_stream_once_and_only_when_relevant(lib):
    doc, code_id = _ingest()
    obj = O.get_objective("o1")
    ranked = R.relevant_ideas(obj)
    assert ranked[0]["title"].startswith("Overnight drift")
    assert ranked[0]["relevance"] > ranked[1]["relevance"]
    R.save_settings({"min_relevance": (ranked[0]["relevance"] + ranked[1]["relevance"]) / 2})

    sent = R.push_next(obj, n=5)
    assert len(sent) == 1, "the irrelevant idea is not pushed"
    assert R.push_next(obj, n=5) == [], "an idea is pushed to an objective once"

    row = O.db().execute("SELECT * FROM ideas WHERE id=?", (sent[0]["idea_row_id"],)).fetchone()
    assert row["trigger"] == "research" and row["model"].startswith("research: ODPD")
    assert f"research_get(chunk={code_id})" in row["text"] and "from research." in row["text"]
    assert "not verified here" in row["text"]

    brief = escalation.ideas_for_context("o1")
    assert [i["id"] for i in brief if i["trigger"] == "research"] == [row["id"]]
    assert escalation.assess(obj)["ideas"] == [], "a research idea neither climbs nor resets the stuck ladder"
    assert row["id"] in [i["id"] for i in escalation.regular_ideas("o1")]


def test_dismissed_ideas_are_never_pushed(lib):
    _ingest()
    obj = O.get_objective("o1")
    R.save_settings({"min_relevance": 0.0})
    for i in R.relevant_ideas(obj):
        R.db().execute("UPDATE doc_ideas SET dismissed=1 WHERE id=?", (i["id"],))
    R.db().commit()
    R._bump()
    assert R.push_next(obj, n=5) == []
    assert R.search("p1", "overnight drift", kinds=["idea"]) == []


def test_idea_models_see_research_findings(lib):
    _ingest()
    obj = O.get_objective("o1")
    R.save_settings({"min_relevance": 0.0})
    prompt = escalation._prompt(obj, escalation.assess(obj), scheduled=True)
    assert "FROM THE RESEARCH LIBRARY" in prompt and "Overnight drift on macro nights" in prompt
    assert "research_get(chunk=" in prompt


def test_reextracting_keeps_ideas_already_sent(lib):
    doc, code_id = _ingest()
    obj = O.get_objective("o1")
    R.save_settings({"min_relevance": 0.0})
    sent = R.push_next(obj, n=1)
    first = R.db().execute("SELECT title FROM doc_ideas WHERE id=?", (sent[0]["doc_idea_id"],)).fetchone()[0]
    R.store_ideas(doc["id"], R.parse_ideas(json.dumps([{"title": "A new idea"}, {"title": first}])), "m2")
    titles = [r["title"] for r in R.db().execute("SELECT title FROM doc_ideas WHERE doc_id=?", (doc["id"],))]
    assert sorted(titles) == sorted([first, "A new idea"]), "the sent idea is kept, and not duplicated"
    assert R.db().execute("SELECT count(*) FROM doc_ideas WHERE id=?", (sent[0]["doc_idea_id"],)).fetchone()[0] == 1


def test_pipeline_end_to_end(lib, monkeypatch):
    calls = []

    async def complete(model, messages, max_tokens, purpose):
        calls.append((model, purpose, messages[0]["content"]))
        code_id = R.db().execute("SELECT id FROM chunks WHERE kind='code'").fetchone()[0]
        return IDEAS_JSON.replace("CODEID", str(code_id))

    monkeypatch.setattr(R, "_complete", complete)
    monkeypatch.setattr(R, "_loaded", lambda: [{"model": "local-llm", "ready": True, "aa": 40}])
    monkeypatch.setattr(R, "pick_model", lambda pid, req="": "local-llm")

    async def running(project_id=None):
        return [O.get_objective("o1")]

    monkeypatch.setattr(R, "_running", running)
    R.save_settings({"min_relevance": 0.0, "push_on_ingest": 1})
    doc, _ = R.create_doc(make_html(), "odpd.html", "p1")
    asyncio.run(R.process(doc["id"]))
    doc = R.get_doc(doc["id"])
    assert doc["status"] == "ready" and doc["idea_model"] == "local-llm", doc
    assert calls and calls[0][1] == f"research:{doc['id']}"
    assert "[code " in calls[0][2] and "odpd_strategy.py" in calls[0][2], "the model sees the code listings"
    assert O.db().execute("SELECT count(*) FROM ideas WHERE trigger='research'").fetchone()[0] == 1


def test_upload_api(lib):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(R.router, prefix="/api")
    queued = []
    pdf = make_pdf()
    try:
        R._queue = asyncio.Queue()
        with TestClient(app) as client:
            r = client.post("/api/research/docs", data={"project_id": "p1"},
                            files=[("file", ("odpd.pdf", pdf, "application/pdf"))])
            assert r.status_code == 200, r.text
            doc = r.json()["docs"][0]
            assert doc["created"] and doc["status"] == "queued"
            queued.append(R._queue.get_nowait())
            again = client.post("/api/research/docs", data={"project_id": "p1"},
                                files=[("file", ("copy.pdf", pdf, "application/pdf"))]).json()["docs"][0]
            assert again["id"] == doc["id"] and not again["created"]
            bad = client.post("/api/research/docs", data={"project_id": "p1"},
                              files=[("file", ("x.exe", b"MZ\x00", "application/octet-stream"))])
            assert bad.status_code == 415
            R.parse_and_store(doc["id"])
            listed = client.get("/api/research/docs", params={"project_id": "p1"}).json()["docs"]
            assert listed[0]["chunks"]["code"] == 1
            full = client.get(f"/api/research/docs/{doc['id']}").json()
            fig = next(c for c in full["chunks"] if c["kind"] == "figure")
            assert client.get(fig["image_url"]).content[:4] == b"\x89PNG"
            src = client.get(f"/api/research/docs/{doc['id']}/source")
            assert src.content == pdf and src.headers["content-security-policy"] == "sandbox"
            compact = client.get(f"/api/research/docs/{doc['id']}", params={"compact": True}).json()
            assert compact["code"][0]["import"].endswith("import odpd_strategy")
            assert client.post(f"/api/research/docs/{doc['id']}/delete").json() == {"deleted": doc["id"]}
            assert client.get(f"/api/research/docs/{doc['id']}").status_code == 404
    finally:
        R._queue = None
    assert queued[0][0] == doc["id"]


# =======================================================================================
# The runner: tools and brief
# =======================================================================================
@pytest.fixture
def runner():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "swarm_runner.py"
    spec = importlib.util.spec_from_file_location("swarm_runner_research_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(runner, monkeypatch, docs):
    calls = []

    def request(base, path, payload=None, **kw):
        calls.append((path, payload))
        if path.startswith("/api/research/docs?"):
            return {"docs": docs}
        if path == "/api/research/search":
            return {"hits": [{"doc_id": "d_1", "kind": "code", "chunk_id": 7, "score": 0.03, "cosine": 0.6,
                              "text": "def causal_rank", "import": "from research.odpd import odpd_strategy"}]}
        if path.startswith("/api/research/chunks/"):
            return {"id": 7, "kind": "code", "text": "def causal_rank(x): ...", "vector": "never", "label": "odpd_strategy.py"}
        return {}

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [])
    return runner.ProjectWorld({"id": "p1", "name": "P"}, "me", [], []), calls


def test_research_tools_only_with_documents(runner, monkeypatch):
    world, _ = _world(runner, monkeypatch, [])
    assert "research_search" not in {t["function"]["name"] for t in world.tools()}
    world, calls = _world(runner, monkeypatch, [{"id": "d_1", "title": "ODPD", "status": "ready", "ideas": 3,
                                                 "chunks": {"code": 2}},
                                                {"id": "d_2", "title": "Half-parsed", "status": "parsing"}])
    names = {t["function"]["name"] for t in world.tools()}
    assert {"research_search", "research_get"} <= names
    assert "Research library (1 documents: ODPD)" in world.briefing()
    hits = world.call("research_search", {"query": "rank", "kinds": ["code", "bogus"], "k": 99})
    assert hits == [{"doc_id": "d_1", "kind": "code", "chunk_id": 7, "text": "def causal_rank",
                     "import": "from research.odpd import odpd_strategy"}]
    assert calls[-1][1] == {"query": "rank", "project_id": "p1", "kinds": ["code"], "k": 20}
    chunk = world.call("research_get", {"chunk": 7})
    assert chunk["text"].startswith("def causal_rank") and "vector" not in chunk
    assert world.call("research_get", {}) == {"documents": [{"doc": "d_1", "title": "ODPD", "ideas": 3, "code": 2}]}


def test_brief_lists_the_library(runner):
    ctx = {"objective": {"title": "T", "description": "", "metric": {"kind": "sharpe"}}, "metric_label": "Sharpe",
           "mode": "explore", "research": {"documents": [
               {"id": "d_1", "title": "ODPD", "package": "odpd", "status": "ready", "ideas": 3, "code": 2}]}}
    p = runner.iteration_prompt(ctx)
    assert "RESEARCH LIBRARY" in p and 'd_1 "ODPD" (package odpd): 3 ideas, 2 code listings' in p
