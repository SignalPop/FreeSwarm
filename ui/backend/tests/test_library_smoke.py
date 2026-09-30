"""library_save's smoke test mounts the right data for the objective it was called with.

The bug the tests here pin down: in a TASK objective the smoke test used to run without
mounting the task server's rows, so ft.rows() / ft.rows_pl() / ft.task() -- the loaders the
brief tells agents to use -- raised "not scored by a task server" and no signal/regime module
in a task project could ever pass its own test. The fix mounts /task/rows.parquet (in-sample
cut, the same one scratch_python gives an experiment), and points the causality harness at it.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from app import library as L
from app import objectives as O
from app import research as RS
from app import task_objectives as T
from app import trade_book as TB


def _project_with_objective(tmp_path, monkeypatch, *, task: bool) -> dict:
    """A minimal project + objective wired to isolated on-disk state.

    Common fakes: `module_files` and `research.code_files` return no extra files (nothing to
    mount besides what the smoke test itself asks for), and `sandbox.execute` (the fenced-off
    call that would spawn a container) is monkey-patched by each test to just capture its args.
    """
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "o.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    L._ready = False
    (tmp_path / "data").mkdir()
    project = {"id": "p1", "data_dir": str(tmp_path / "data"),
               "task_server": "gex" if task else None}
    monkeypatch.setattr(L, "_project", lambda pid: project)
    monkeypatch.setattr(L, "module_files", lambda pid: {})
    monkeypatch.setattr(RS, "code_files", lambda pid: {})
    now = time.time()
    if task:
        metric = {"kind": "task", "task_server": "gex", "task": "t", "higher_is_better": True,
                  "task_info": {"version": "v1"}}
        O.db().execute(
            "INSERT INTO objectives (id, project_id, title, metric, dataset, time_column, split_date, "
            "created_at, updated_at) VALUES ('o1','p1','t',?, 'mcp_tasks_gex_t', 't', '2024-07-19', ?, ?)",
            (json.dumps(metric), now, now))
    else:
        metric = {"kind": "positions", "higher_is_better": True, "cost_bps": 1.0}
        O.db().execute(
            "INSERT INTO objectives (id, project_id, title, metric, dataset, time_column, split_date, "
            "created_at, updated_at) VALUES ('o1','p1','t',?, 'candles', 't', '2024-07-19', ?, ?)",
            (json.dumps(metric), now, now))
    O.db().commit()
    return project


def test_a_task_objective_smoke_test_mounts_the_task_rows_at_task(tmp_path, monkeypatch):
    """The bug: a task objective library_save ran the smoke test without /task, so any test
    that read ft.rows()/ft.rows_pl() -- what the brief points at -- failed with 'not scored
    by a task server'. Now the in-sample task rows land at /task/rows.parquet."""
    _project_with_objective(tmp_path, monkeypatch, task=True)
    task_folder = tmp_path / "export"
    task_folder.mkdir()

    async def export_dir(obj, cut):
        assert cut == "2024-07-19"                            # in-sample cut, never full/holdout
        return task_folder
    monkeypatch.setattr(T, "export_dir", export_dir)
    monkeypatch.setattr(TB, "sandbox_dataset", lambda obj: None)      # no book yet

    runs: list[dict] = []

    async def execute(code, *, timeout_s, files, mounts):
        runs.append({"files": files, "mounts": mounts})
        return {"ok": True, "stdout": "[lib] imported x\ndone", "stderr": "", "artifacts": [],
                "duration_s": 0.1, "run_dir": str(tmp_path / "run")}
    (tmp_path / "run" / ".ft").mkdir(parents=True)
    monkeypatch.setattr(O, "execute", execute)

    req = L.SaveModule(name="x", kind="util", description="", code="def helper():\n    return 1\n",
                       note="", test_code="print('hi')\n", objective_id="o1")
    out = asyncio.run(L.save_module("p1", req))

    assert out["saved"] is True and out["version"] == 1
    assert runs and runs[0]["mounts"] == [(str(task_folder), "/task")]
    # The project catalog is not mounted for a task run: the whole point is /task or nothing.
    assert not any(m for m in runs[0]["mounts"] if m[1].startswith("/data"))
    # Nothing under .ft/lib/x.py that carries the module's own code was blanked out.
    assert runs[0]["files"][".ft/lib/x.py"] == "def helper():\n    return 1\n"


def test_a_task_objectives_causality_harness_reads_task_rows_not_project_datasets(tmp_path, monkeypatch):
    """The causality (prefix-invariance) check in a task project reads /task/rows.parquet via
    ft.rows(); the config carries task=True so the harness picks that path over ft.load()."""
    _project_with_objective(tmp_path, monkeypatch, task=True)
    task_folder = tmp_path / "export"
    task_folder.mkdir()

    async def export_dir(obj, cut):
        return task_folder
    monkeypatch.setattr(T, "export_dir", export_dir)
    monkeypatch.setattr(TB, "sandbox_dataset", lambda obj: None)
    (tmp_path / "run" / ".ft").mkdir(parents=True)

    async def execute(code, *, timeout_s, files, mounts):
        return {"ok": True, "stdout": "[lib] imported s\n", "stderr": "", "artifacts": [],
                "duration_s": 0.1, "run_dir": str(tmp_path / "run"), "files": files, "mounts": mounts}
    captured: dict = {}

    async def execute_and_capture(code, *, timeout_s, files, mounts):
        captured.update({"files": files, "mounts": mounts})
        return await execute(code, timeout_s=timeout_s, files=files, mounts=mounts)
    monkeypatch.setattr(O, "execute", execute_and_capture)

    req = L.SaveModule(name="s", kind="signal", description="",
                       code=("def signal(df):\n"
                             "    import numpy as np\n"
                             "    return np.zeros(len(df))\n"),
                       note="", test_code="", objective_id="o1")
    asyncio.run(L.save_module("p1", req))

    cfg = json.loads(captured["files"][".ft/causality_cfg.json"])
    assert cfg["task"] is True and cfg["time_column"] == "t"
    # The causality harness file is shipped, and the branch it takes is by the cfg flag, not
    # by any hard-coded task/rows path in the smoke code -- so a plain objective keeps working.
    assert ".ft/causality_check.py" in captured["files"]
    assert 'CFG.get("task")' in captured["files"][".ft/causality_check.py"]


def test_the_trade_book_never_reaches_a_module_smoke_test(tmp_path, monkeypatch):
    """`scratch_python` mounts the trade book beside /task for EXPERIMENTS only. A module is code a
    scored candidate imports, and scored runs never see the book (a look-ahead cut cannot
    truncate it), so the smoke test must not offer it either -- a module reading it would pass
    here and the causality check, then fail (or leak) when scored."""
    _project_with_objective(tmp_path, monkeypatch, task=True)
    task_folder = tmp_path / "export"
    task_folder.mkdir()

    async def export_dir(obj, cut):
        return task_folder
    monkeypatch.setattr(T, "export_dir", export_dir)
    book = tmp_path / "data" / "trade_book" / "trades_o1.parquet"
    book.parent.mkdir()
    book.write_bytes(b"parquet-stub")
    monkeypatch.setattr(TB, "sandbox_dataset", lambda obj: {"view": "trade_book_trades_o1",
                                                             "path": "trade_book/trades_o1.parquet",
                                                             "format": "parquet", "root": "/",
                                                             "host": str(book)})
    (tmp_path / "run" / ".ft").mkdir(parents=True)
    runs: list[dict] = []

    async def execute(code, *, timeout_s, files, mounts):
        runs.append({"files": files, "mounts": mounts})
        return {"ok": True, "stdout": "", "stderr": "", "artifacts": [], "duration_s": 0.1,
                "run_dir": str(tmp_path / "run")}
    monkeypatch.setattr(O, "execute", execute)

    req = L.SaveModule(name="uses_book", kind="util", description="", code="def h():\n    return 1\n",
                       note="", test_code="", objective_id="o1")
    asyncio.run(L.save_module("p1", req))

    assert runs[0]["mounts"] == [(str(task_folder), "/task")]            # the task rows, nothing else
    assert json.loads(runs[0]["files"][".ft/catalog.json"]) == []


def test_a_plain_objective_smoke_test_still_mirrors_the_project_datasets(tmp_path, monkeypatch):
    """The fix only changes task objectives. A plain positions objective still gets the
    in-sample mirror at /data (never /task), same as before."""
    _project_with_objective(tmp_path, monkeypatch, task=False)
    # `export_dir` and `sandbox_dataset` must not be reached for a plain objective.
    monkeypatch.setattr(T, "export_dir",
                        lambda obj, cut: (_ for _ in ()).throw(AssertionError("must not export task rows")))
    monkeypatch.setattr(TB, "sandbox_dataset",
                        lambda obj: (_ for _ in ()).throw(AssertionError("no book for a plain objective")))
    # No mirror machinery is exercised here -- the smoke test just runs the container -- so a
    # trivial stub tells us the mirror side kicked in without asking build_mirror to shell out.
    monkeypatch.setattr(O, "build_mirror",
                        lambda obj, data_dir, cut=None, only=None: {"root": str(tmp_path / "m"), "items": []})
    (tmp_path / "run" / ".ft").mkdir(parents=True)
    runs: list[dict] = []

    async def execute(code, *, timeout_s, files, mounts):
        runs.append({"files": files, "mounts": mounts})
        return {"ok": True, "stdout": "", "stderr": "", "artifacts": [], "duration_s": 0.1,
                "run_dir": str(tmp_path / "run")}
    monkeypatch.setattr(O, "execute", execute)

    req = L.SaveModule(name="y", kind="util", description="", code="def h():\n    return 1\n",
                       note="", test_code="", objective_id="o1")
    asyncio.run(L.save_module("p1", req))

    assert not any(dst == "/task" for _, dst in runs[0]["mounts"])
    assert (str(tmp_path / "data"), "/data") in runs[0]["mounts"]


def test_the_error_message_names_the_loader_this_project_uses(tmp_path, monkeypatch):
    """A failure that points at the wrong loader (ft.rows() in a plain project, or a save
    without objective_id) comes back with a one-line clue that says which loader to use."""
    _project_with_objective(tmp_path, monkeypatch, task=False)
    monkeypatch.setattr(O, "build_mirror",
                        lambda obj, data_dir, cut=None, only=None: {"root": str(tmp_path / "m"), "items": []})
    (tmp_path / "run" / ".ft").mkdir(parents=True)

    async def execute(code, *, timeout_s, files, mounts):
        return {"ok": False, "stdout": "",
                "stderr": ('RuntimeError: ft.rows_pl(): this objective is not scored by a task '
                           'server -- use ft.load_pl() instead'),
                "artifacts": [], "duration_s": 0.1, "run_dir": str(tmp_path / "run")}
    monkeypatch.setattr(O, "execute", execute)

    req = L.SaveModule(name="wrong_loader", kind="util", description="", code="def h():\n    return 1\n",
                       note="", test_code="rows = None\n", objective_id="o1")
    out = asyncio.run(L.save_module("p1", req))
    assert out["saved"] is False and out["test_ok"] is False
    assert "ft.load" in out["error"] and "not scored by a task server" in out["error"].lower()

    # And when no objective_id was passed at all, the hint names the loader for THIS project's
    # kind -- task_server set on the project means ft.rows()/ft.rows_pl().
    monkeypatch.setattr(L, "_project", lambda pid: {"id": "p1", "data_dir": str(tmp_path / "data"),
                                                     "task_server": "gex"})

    async def execute2(code, *, timeout_s, files, mounts):
        return {"ok": False, "stdout": "", "stderr": "boom", "artifacts": [], "duration_s": 0.1,
                "run_dir": str(tmp_path / "run")}
    monkeypatch.setattr(O, "execute", execute2)

    req = L.SaveModule(name="no_oid", kind="util", description="", code="def h():\n    return 1\n",
                       note="", test_code="", objective_id=None)
    out = asyncio.run(L.save_module("p1", req))
    assert "ft.rows()" in out["error"] and "no objective_id" in out["error"]


def test_a_test_that_reaches_the_module_the_wrong_way_is_told_the_right_one(tmp_path, monkeypatch):
    """Bugs #85 and #16's tail: test code that calls signal(df) bare, or imports the module by its
    own name, failed with a NameError / ModuleNotFoundError that says neither that the test
    starts with `from lib import <name>` nor where library modules live."""
    _project_with_objective(tmp_path, monkeypatch, task=False)
    monkeypatch.setattr(O, "build_mirror",
                        lambda obj, data_dir, cut=None, only=None: {"root": str(tmp_path / "m"), "items": []})
    (tmp_path / "run" / ".ft").mkdir(parents=True)
    stderr = ["NameError: name 'signal' is not defined. Did you forget to import 'signal'?"]

    async def execute(code, *, timeout_s, files, mounts):
        return {"ok": False, "stdout": "[lib] imported sig_a", "stderr": stderr[0], "artifacts": [],
                "duration_s": 0.1, "run_dir": str(tmp_path / "run")}
    monkeypatch.setattr(O, "execute", execute)
    req = L.SaveModule(name="sig_a", kind="util", description="", note="", objective_id="o1",
                       code="import ft\n\ndef signal(df):\n    return df['Close'] * 0\n", test_code="pos = signal(df)\n")
    err = asyncio.run(L.save_module("p1", req))["error"]
    assert err.startswith("the smoke test failed -- fix the module and save again")
    assert "call sig_a.signal(...), not signal(...)" in err and "from lib import sig_a" in err

    stderr[0] = "ModuleNotFoundError: No module named 'sig_a'"
    assert "`from lib import sig_a`" in asyncio.run(L.save_module("p1", req))["error"]
    monkeypatch.setattr(L, "module_files", lambda pid: {".ft/lib/helper_b.py": "x = 1\n"})
    stderr[0] = "ModuleNotFoundError: No module named 'helper_b'"           # another module of the library
    assert "`from lib import helper_b`" in asyncio.run(L.save_module("p1", req))["error"]

    # Not the module's own names: the plain message, nothing invented.
    for other in ("NameError: name 'np' is not defined", "ModuleNotFoundError: No module named 'talib'", "boom"):
        stderr[0] = other
        assert asyncio.run(L.save_module("p1", req))["error"] == "the smoke test failed -- fix the module and save again"
