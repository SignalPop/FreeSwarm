"""Objectives scored by a task server (a data/action MCP) instead of the harness's own pricing.

A task objective names a registered MCP server and one of its tasks (metric.kind == "task",
metric.task_server, metric.task). The server owns the problem -- it serves the rows, defines
what an action means and scores the actions -- and this module is the harness's side of the
interface (mcp/README.md; the control plane never imports server code, it only calls the tools):

* rows: `harness_export_rows` writes the task's rows to a folder that is mounted read-only into
  the sandbox at /task (strategy code reads it with ft.rows()). The full export is cached per
  task version; for a look-ahead cut the export stops at the cut, so the code cannot see what
  comes after it -- the same guarantee truncated dataset mirrors give the positions harness.
* actions: the code reports one action per row with ft.report_actions, written to
  .ft/actions.parquet as (t, pos) -- the same shape as positions, so the proven positions
  look-ahead comparison applies unchanged.
* score: `harness_evaluate` returns in-sample and holdout segments; the leaderboard score is the
  weaker of the two (robust ranking) or the holdout (holdout ranking). Agents are shown only the
  in-sample numbers and the server's in-sample notes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from . import mcp_registry

ACTIONS_FILE = "actions.parquet"


def is_task(obj: dict) -> bool:
    return (obj.get("metric") or {}).get("kind") == "task"


def _norm_ts(v: str | None) -> str | None:
    """'2024-01-19T18:17:00' -> '2024-01-19 18:17:00' (the harness compares cut strings)."""
    if not v:
        return v
    s = str(v).replace("T", " ")
    return s[:19] if len(s) > 19 else s


async def call(server: str, tool: str, args: dict[str, Any], timeout_s: float = 600.0) -> dict[str, Any]:
    """Call a task-server tool and return its structured result; a transport failure or an
    {"error": ...} result is an HTTPException naming the server and tool."""
    specs = [s for s in mcp_registry.load_config() if s.enabled]
    spec = next((s for s in specs if s.name == server), None)
    if spec is None:
        raise HTTPException(status_code=409, detail=f"task server {server!r} is not registered (or disabled) in "
                                                    "ui/backend/mcp_servers.json")
    if spec.kind != "task":
        raise HTTPException(status_code=409, detail=f"{server!r} is a tool connector, not a data/action MCP "
                                                    '(register it with "kind": "task")')
    if spec.oauth and spec.transport != "stdio":
        from . import mcp_oauth

        if server not in mcp_oauth.authorised_servers():
            raise HTTPException(status_code=409, detail=f"task server {server!r} needs sign-in: open Connectors in the "
                                                        f"console, press Connect on {server!r} and approve with its "
                                                        "passphrase (from its make_oauth_secrets.py)")
    try:
        res = await asyncio.wait_for(mcp_registry.call_tool(specs, f"{server}__{tool}", args), timeout_s)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"task server {server!r}: {tool} took over {timeout_s:.0f}s") from None
    except Exception as exc:  # noqa: BLE001 -- the server being down must read as such
        raise HTTPException(status_code=502, detail=f"task server {server!r}: {tool} failed: "
                                                    f"{mcp_registry.describe_exception(exc)}") from None
    out = res.get("structured")
    if isinstance(out, dict) and set(out) == {"result"} and isinstance(out["result"], dict):
        out = out["result"]
    if not isinstance(out, dict):
        try:
            out = json.loads(res.get("content") or "")
        except ValueError:
            out = None
    if res.get("is_error") or not isinstance(out, dict):
        raise HTTPException(status_code=502, detail=f"task server {server!r}: {tool} returned "
                                                    f"{(res.get('content') or '')[:600]!r}")
    if out.get("error"):
        raise HTTPException(status_code=502, detail=f"task server {server!r}: {tool}: {out['error']}")
    return out


async def describe(server: str, task: str, target: str | None = None,
                   value_function: str | None = None, action_rule: str | None = None,
                   direction: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"task": task}
    if target:
        args["target"] = target
    if value_function:
        args["value_function"] = value_function
    if action_rule:
        args["action_rule"] = action_rule
    if direction:
        args["direction"] = direction
    return await call(server, "task_describe", args, timeout_s=900)


def _target(obj: dict) -> dict[str, Any]:
    """{"target": ...} when the objective values on a chosen target (a project setting), else {}."""
    t = (obj.get("metric") or {}).get("target")
    return {"target": t} if t else {}


def _choices(obj: dict) -> dict[str, Any]:
    """The objective's target, value function and action rule (project settings) for the calls that value."""
    m = obj.get("metric") or {}
    return {**_target(obj), **({"value_function": m["value_function"]} if m.get("value_function") else {}),
            **_rule(obj)}


def _rule(obj: dict) -> dict[str, Any]:
    """How the objective's actions are managed: its action rule and -- when the task offers a
    choice of directions -- the direction (long only / short only / both)."""
    m = obj.get("metric") or {}
    out = {"action_rule": m["action_rule"]} if m.get("action_rule") else {}
    if m.get("direction") and (m.get("task_info") or {}).get("directions"):
        out["direction"] = m["direction"]
    return out


def snapshot(d: dict[str, Any]) -> dict[str, Any]:
    """What an objective keeps of a task's description: enough for the agents' brief, the console
    and the look-ahead plan without calling the server for every view."""
    return {
        "title": d.get("title"), "description": d.get("description"), "brief": d.get("brief"),
        "target": d.get("target"), "action": d.get("action"), "score": d.get("score"),
        "target_options": d.get("target_options"), "valuation": d.get("valuation"), "shape": d.get("shape"),
        "value_function": d.get("value_function"), "value_functions": d.get("value_functions"),
        "action_rule": d.get("action_rule"), "action_rules": d.get("action_rules"),
        "direction": d.get("direction"), "directions": d.get("directions"), "guidance": d.get("guidance"),
        "display_tz": d.get("display_tz"),
        "rows": d.get("rows"), "in_sample_rows": d.get("in_sample_rows"), "first": d.get("first"),
        "last_in_sample": d.get("last_in_sample"), "holdout_from": d.get("holdout_from"),
        "version": d.get("version"),
        "columns": [{k: c.get(k) for k in ("name", "dtype", "role", "description")} for c in d.get("columns") or []],
    }


def with_trade_limit(rule: str | None, cap: int | None) -> str:
    """An action rule with its daily trade limit replaced ("intraday+max_3_trades"; None/0 = none).
    The limit is the "+max_<k>_trades" part of the rule (the task server's grammar)."""
    base = "+".join(p.strip() for p in str(rule or "intraday").split("+")
                    if p.strip() and trade_limit(p) is None) or "intraday"
    return base + (f"+max_{int(cap)}_trades" if cap else "")


def trade_limit(rule: str | None) -> int | None:
    """The daily trade limit in an action rule, or None."""
    for p in str(rule or "").split("+"):
        p = p.strip()
        if p.startswith("max_") and p.endswith(("_trades", "_trade", "_trades_per_day")):
            try:
                return int(p.split("_")[1])
            except ValueError:
                return None
    return None


GUIDANCE_TTL_S = 600
_guidance_checked: dict[str, float] = {}


async def refresh_guidance(obj: dict) -> dict[str, Any] | None:
    """The objective's metric with the task server's CURRENT agent guidance, or None when it is
    unchanged (or was checked less than GUIDANCE_TTL_S ago). The snapshot taken when the objective
    was created would otherwise keep briefing agents with advice the server has since improved;
    only `guidance` is refreshed -- the rest of the snapshot (the version, the columns) decides
    the exports and stays as it was."""
    oid, now = obj["id"], time.time()
    if now - _guidance_checked.get(oid, 0.0) < GUIDANCE_TTL_S:
        return None
    _guidance_checked[oid] = now
    m = obj.get("metric") or {}
    rule = _rule(obj)
    d = await describe(m["task_server"], m["task"], m.get("target"), m.get("value_function"),
                       rule.get("action_rule"), rule.get("direction"))
    info = m.get("task_info") or {}
    if d.get("guidance") is None or d.get("guidance") == info.get("guidance"):
        return None
    return {**m, "task_info": {**info, "guidance": d.get("guidance")}}


async def prepare_objective(metric: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Fill a new task objective's metric from the server's description; returns (metric,
    split_date). The split is the holdout boundary's date, the mid cut the task's in-sample cut."""
    server, task = metric.get("task_server"), metric.get("task")
    if not server or not task:
        raise HTTPException(status_code=400, detail="a task objective needs metric.task_server and metric.task")
    d = await describe(server, task, metric.get("target"), metric.get("value_function"), metric.get("action_rule"),
                       metric.get("direction"))
    metric = {**metric, "task_info": snapshot(d), "target": d.get("target") or metric.get("target"),
              "value_function": d.get("value_function") or metric.get("value_function"),
              "action_rule": d.get("action_rule") or metric.get("action_rule"),
              "direction": d.get("direction"),                    # None: the task offers no choice of sides
              "higher_is_better": bool((d.get("score") or {}).get("higher_is_better", True))}
    # Positions-harness settings do not apply: the task server prices the actions.
    metric["price_column"] = None
    hold = d.get("holdout_from")
    split = str(hold)[:10] if hold else None
    cuts = [_norm_ts(c) for c in d.get("cuts") or []]
    mid = next((c for c in cuts if split and c and c[:10] < split), None)
    if mid:
        metric["mid_cut"] = mid
    return metric, split


def _work(obj: dict) -> Path:
    from .objectives import WORK_ROOT

    return WORK_ROOT / obj["id"] / "task"


_EXPORT_LOCKS: dict[str, asyncio.Lock] = {}


def _safe_part(s: Any) -> str:
    """A server-given string made safe as one path component."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s or ""))[:64]


async def export_dir(obj: dict, cut: str | None = None, temporary: bool = False) -> Path:
    """A folder holding rows.parquet (+ task.json) with the task's rows -- all of them, or those
    before `cut`. Full and fixed-cut exports are cached per task version; a temporary one (a
    look-ahead cut placed after the candidate's own trades) is the caller's to discard."""
    m = obj["metric"]
    version = _safe_part((m.get("task_info") or {}).get("version")) or "v"
    tag = "full" if not cut else "cut-" + "".join(ch for ch in cut if ch.isdigit())
    # The target is part of the export's identity (task.json names it; a server may shape rows by it).
    tgt = "".join(ch if ch.isalnum() else "_" for ch in (m.get("target") or "default"))
    rule = _safe_part(m.get("action_rule") or "default")
    base = _work(obj) / ("tmp" if temporary else version) / tgt / rule / tag
    lock = _EXPORT_LOCKS.setdefault(str(base), asyncio.Lock())
    async with lock:
        if (base / "rows.parquet").is_file() and (base / "task.json").is_file():
            return base
        out = await call(m["task_server"], "harness_export_rows",
                         {"task": m["task"], "path": str(base / "rows.parquet"), "until": cut, **_choices(obj)}, timeout_s=900)
        if out.get("version") and out["version"] != version and not temporary:
            # The server's data changed since the objective was created: keep exporting (the
            # rows are what the server serves now) but note it for the operator.
            (base / "VERSION_CHANGED").write_text(f"objective created on {version}, server now {out['version']}")
        if not (base / "rows.parquet").is_file():
            raise HTTPException(status_code=502, detail=f"task server {m['task_server']!r} reported an export but "
                                                        f"wrote no {base / 'rows.parquet'}")
        return base


VIEW_DIR = "mcp_tasks"


async def ensure_view(obj: dict, data_dir: str) -> str | None:
    """The task's IN-SAMPLE rows as a dataset of the project (<data_dir>/mcp_tasks/<server>_<task>.parquet),
    so the analysis tools -- decile plots, field scans, regime maps, query_data, the field guide --
    see every column of the MCP's schema, delayed exactly as the server serves it. The file holds
    only rows before the split: the holdout cannot leak through any tool. Rebuilt when the server's
    data version changes. Returns the view name (None when there is no data folder)."""
    from . import datasource

    if not data_dir or not Path(data_dir).is_dir():
        return None
    m = obj["metric"]
    safe = lambda s: "".join(ch if ch.isalnum() else "_" for ch in str(s)).strip("_").lower()  # noqa: E731
    rel = Path(VIEW_DIR) / f"{safe(m['task_server'])}_{safe(m['task'])}.parquet"
    dst = Path(data_dir) / rel
    # One file per task, shared by every objective on it, so the stamp describes the file: what
    # it was exported for. An objective whose version, split or target differs rebuilds it. (Each
    # analysis also cuts at its own objective's split, so a later split never shows a holdout.)
    want = {"version": (m.get("task_info") or {}).get("version") or "v", "split": obj.get("split_date"),
            "target": m.get("target")}
    from .objectives import WORK_ROOT

    stamp_file = WORK_ROOT / "_task_views" / (hashlib.sha1(str(dst.resolve()).lower().encode()).hexdigest()[:16] + ".json")
    try:
        stamp = json.loads(stamp_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stamp = {}
    if not dst.is_file() or any(stamp.get(k) != v for k, v in want.items()):
        folder = await export_dir(obj, obj.get("split_date") or None)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(folder / "rows.parquet", tmp)
        tmp.replace(dst)
        stamp_file.parent.mkdir(parents=True, exist_ok=True)
        stamp_file.write_text(json.dumps({**want, "path": str(dst), "objective": obj.get("id")}), encoding="utf-8")
    return datasource._view_name(rel)


def field_guide(obj: dict) -> dict[str, Any]:
    """The field guide for a task objective, from the MCP's own column descriptions: every column,
    grouped by family (the prefix before "_"), with what the family is."""
    cols = (obj["metric"].get("task_info") or {}).get("columns") or []
    fams: dict[str, dict[str, Any]] = {}
    for c in cols:
        if c.get("role") == "key":
            continue
        name = c["name"]
        fam = name.split("_", 1)[0] if "_" in name else "(base)"
        f = fams.setdefault(fam, {"about": c.get("description") or "", "columns": []})
        f["columns"].append(name)
        if not f["about"] and c.get("description"):
            f["about"] = c["description"]
    return fams                       # the same shape as objectives.field_guide: {family: {about, columns}}


def discard(path: Path | None) -> None:
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def actions_hold(obj: dict) -> bool:
    """Whether the task's actions are levels held until the next one (a repeat changes nothing),
    not events such as orders (a repeat is another order). Unknown kinds count as events."""
    kind = (((obj.get("metric") or {}).get("task_info") or {}).get("action") or {}).get("kind")
    return kind in ("position", "setpoint", "value")


def actions_file(report: dict) -> Path | None:
    p = Path(report["run_dir"]) / ".ft" / ACTIONS_FILE
    return p if p.is_file() else None


async def action_log(obj: dict, actions: Path, start: str | None, end: str | None, limit: int = 500) -> dict[str, Any]:
    """What the server's action management made of a candidate's actions between start and end
    (harness_actions): trades, a charge schedule, ... and the managed state."""
    m = obj["metric"]
    return await call(m["task_server"], "harness_actions", {"task": m["task"], "actions_path": str(actions),
                                                             "start": start, "end": end, "limit": limit, **_target(obj),
                                                             **_rule(obj)},
                      timeout_s=300)


async def evaluate_actions(obj: dict, actions: Path) -> dict[str, Any]:
    m = obj["metric"]
    return await call(m["task_server"], "harness_evaluate", {"task": m["task"], "actions_path": str(actions), **_choices(obj)},
                      timeout_s=900)


def _sparse_hint(ins: dict[str, Any], hold: dict[str, Any]) -> str:
    """What the agent can learn from an unscorable holdout WITHOUT seeing it: when the in-sample
    trading rate, carried over the holdout's length, falls short of the active days a period needs
    (#138/#139 on 10-01: 30 and 29 active days of 223 in-sample, 15 and 7 in the holdout, 20 needed),
    say so from the in-sample numbers alone."""
    import re

    need = re.search(r"need (\d+)", str(hold.get("note") or ""))
    days, active, h_days = ins.get("days"), ins.get("active_days"), hold.get("days")
    if not (need and days and active is not None and h_days):
        return ""
    expect = active / days * h_days
    if expect >= int(need.group(1)) * 1.5:
        return ""
    return (f" -- the strategy is SPARSE: it trades on {active} of {days} days in-sample, which over the "
            f"holdout's {h_days} days is ~{expect:.0f}, near or under the {need.group(1)} active days a period "
            "needs to be scored. Trade more regularly (looser entry, both sides, more sessions)")


def score(obj: dict, ev: dict[str, Any]) -> tuple[float | None, float | None, str, dict[str, Any], list[list]]:
    """(leaderboard score, in-sample score, note, metrics, curve) from a harness_evaluate result.

    Robust ranking (the default) takes the weaker of the in-sample and holdout scores -- a
    strategy must work in both; holdout ranking takes the holdout alone. `unranked` from the
    server (e.g. a one-sided strategy) and a missing segment score leave it unranked."""
    m = obj["metric"]
    higher = bool(m.get("higher_is_better", True))
    seg = ev.get("segments") or {}
    ins, hold = seg.get("in_sample") or {}, seg.get("holdout") or {}
    is_score, ho_score = ins.get("score"), hold.get("score")
    note = ""
    if ev.get("unranked"):
        s, note = None, str(ev["unranked"])
    elif m.get("rank", "robust") == "holdout":
        s = ho_score
        note = "" if s is not None else "holdout: no score (the task server could not score that period)"
    elif is_score is None or ho_score is None:
        # The note reaches the agent: the holdout's own reason (row counts, active days) stays in
        # the metrics for the operator, the agent only learns that the holdout was not scorable.
        s = None
        note = (f"in-sample: {ins.get('note') or 'no score'}" if is_score is None
                else "holdout: no score (the task server could not score that period)" + _sparse_hint(ins, hold))
    else:
        s = min(is_score, ho_score) if higher else max(is_score, ho_score)
    few = too_few_trades(m, ev)
    if few and s is not None:
        s, note = None, few
    metrics: dict[str, Any] = {
        "task": {k: ev.get(k) for k in ("segments", "diagnostics", "notes", "unranked", "actions", "score_name",
                                         "higher_is_better", "curve_kind")},
        "in_sample": ins, "source": f"task server {m.get('task_server')} / {m.get('task')}",
    }
    if hold:
        metrics["holdout"] = hold
    if few:
        metrics["too_few_trades"] = few
    if s is not None and m.get("rank", "robust") != "holdout":
        metrics["rank"] = {"method": "robust", "base": s, "weaker": "in_sample" if s == is_score else "holdout",
                           "holdout": ho_score}
    curve = [[str(d)[:10], float(v)] for d, v in (ev.get("curve") or []) if isinstance(v, (int, float))]
    return (None if s is None else float(s)), (None if is_score is None else float(is_score)), note, metrics, curve


def in_sample_trades(ev: dict[str, Any]) -> float | None:
    """A task candidate's in-sample trade count: the server's per-side counts, else trades per day
    times the period's days; None when the server reports neither."""
    d = (ev.get("diagnostics") or {}).get("in_sample") or {}
    n = d.get("trades")
    if isinstance(n, dict):
        counts = [v for v in n.values() if isinstance(v, (int, float))]
        if counts:
            return float(sum(counts))
    elif isinstance(n, (int, float)):
        return float(n)
    tpd = d.get("trades_per_day")
    days = ((ev.get("segments") or {}).get("in_sample") or {}).get("days")
    return float(tpd) * float(days) if tpd is not None and days else None


def too_few_trades(m: dict[str, Any], ev: dict[str, Any]) -> str | None:
    """Why a task candidate trades too rarely to rank, or None. When every strategy loses after
    costs, trading less loses less, and the ranking would otherwise drift to strategies that barely
    trade at all -- so with min_trades set, its in-sample trades IN ALL must reach it: enough for the
    score to be more than luck, while a selective strategy that skips most days still ranks.
    min_trades_per_day (a daily quota, the older floor) is still honoured where it is set, but it
    works against selectivity: taking only the big winners means trading well under once a day."""
    floor = float(m.get("min_trades") or 0.0)
    if floor:
        n = in_sample_trades(ev)
        if n is not None and n < floor:
            return (f"too few trades: {n:g} in-sample -- at least {floor:g} in all are needed for the score to "
                    "be more than luck. Stay selective: let the same kind of entry fire on more days (a slightly "
                    "looser threshold, or a second setup that is just as good), never filler trades to reach "
                    "the count")
    per_day = float(m.get("min_trades_per_day") or 0.0)
    if not per_day:
        return None
    tpd = ((ev.get("diagnostics") or {}).get("in_sample") or {}).get("trades_per_day")
    if tpd is None or float(tpd) >= per_day:
        return None
    return (f"too few trades: {float(tpd):g} trades/day in-sample -- at least {per_day:g} a day are required. "
            "Check the signal THROUGHOUT the session (not once at the open or at one fixed time) and re-enter "
            "whenever it fires again after an exit")


def agent_notes(metrics: dict[str, Any]) -> dict[str, Any]:
    """What the agent is told about a task candidate: in-sample only."""
    t = metrics.get("task") or {}
    out: dict[str, Any] = {}
    if t.get("notes"):
        out["task_notes"] = t["notes"]
    d = (t.get("diagnostics") or {}).get("in_sample")
    if d:
        out["diagnostics_in_sample"] = d
    if t.get("unranked") or metrics.get("too_few_trades"):
        out["not_ranked_because"] = t.get("unranked") or metrics["too_few_trades"]
    return out


async def lookahead(obj: dict, code: str, full_actions: Path, run: Any, seed: Any = None,
                    concurrency: int = 2, progress: dict | None = None) -> tuple[str, str]:
    """The look-ahead test for a task candidate: re-run the code on rows exported only up to
    each cut, and require every action before the cut to be identical. `run(code, task_dir)` runs
    the candidate with that folder mounted at /task. Cuts: the task's fixed ones (holdout
    boundary, mid in-sample) plus ones placed just after the candidate's own action changes."""
    from .objectives import LOOKAHEAD_ACTIVE_CUTS, _active_cuts, _off, _positions_lookahead, cuts

    fixed = cuts(obj)
    active = await _off(_active_cuts, obj, full_actions, LOOKAHEAD_ACTIVE_CUTS, seed) if obj.get("split_date") else []
    plan = [(c, False) for c in fixed] + [(c, True) for c in active if c not in fixed]
    rows: list[dict] = []
    if progress is not None:
        progress.update(cuts_done=0, cuts_total=len(plan))
        rows = [{"label": f"cut at {c}", "kind": "after an action change" if t else "fixed", "state": "queued"}
                for c, t in plan]
        progress.setdefault("runs", []).extend(rows)
    sem = asyncio.Semaphore(concurrency)
    failed = asyncio.Event()

    async def one(i: int, cut: str, temporary: bool) -> tuple[str, str] | None:
        import time

        row = rows[i] if rows else {}
        async with sem:
            if failed.is_set():
                row["state"] = "skipped"
                return None
            row.update(state="running", started=time.time())
            folder, v = None, None
            try:
                folder = await export_dir(obj, cut, temporary)
                trunc = await run(code, folder)
                if not trunc["ok"]:
                    v = ("error", f"the script failed on rows cut at {cut}: " + trunc["stderr"][-600:])
                    return v
                t_act = actions_file(trunc)
                v = (await _off(_positions_lookahead, full_actions, t_act, cut)) if t_act \
                    else ("error", f"no actions reported on rows cut at {cut}")
                v = (v[0], v[1].replace("positions", "actions").replace("position ", "action "))
                if v[0] == "fail":
                    failed.set()
                return v
            except HTTPException as exc:
                v = ("error", f"rows cut at {cut} could not be exported: {exc.detail}")
                return v
            finally:
                if row:
                    row.update(state=v[0] if v else "error", seconds=round(time.time() - row["started"], 1))
                if progress is not None:
                    progress["cuts_done"] = progress.get("cuts_done", 0) + 1
                if temporary:
                    discard(folder)

    verdicts = [v for v in await asyncio.gather(*(one(i, c, t) for i, (c, t) in enumerate(plan))) if v]
    if not verdicts:
        return "error", "no look-ahead cut could be run"
    worst = "fail" if any(v == "fail" for v, _ in verdicts) else \
        "error" if any(v == "error" for v, _ in verdicts) else "pass"
    if worst == "pass":
        return worst, (f"{len(verdicts)} cuts ({len(active)} placed just after the candidate's own action changes, "
                       f"{len(verdicts) - len(active)} fixed): every earlier action identical with later rows removed")
    return worst, " | ".join(d for v, d in verdicts if v == worst)


async def project_mcp(project: dict) -> dict[str, Any]:
    """What the project's data/action MCP offers: every task described with the project's chosen
    target -- the data's shape, the target and its options, the actions, the value function."""
    server = project.get("task_server")
    if not server:
        return {"server": None, "tasks": [], "errors": []}
    listing = await call(server, "task_list", {}, timeout_s=900)
    options = project.get("task_options") or {}
    tasks, errors = [], list(listing.get("errors") or [])
    for t in listing.get("tasks") or []:
        name = t.get("name")
        if not name or t.get("error"):
            errors.append(f"{name}: {t.get('error')}")
            continue
        try:
            o = options.get(name) or {}
            d = await describe(server, name, o.get("target"), o.get("value_function"), o.get("action_rule"),
                               o.get("direction"))
            d.pop("cuts", None)
            d["columns"] = [{k: c.get(k) for k in ("name", "dtype", "role", "description")} for c in d.get("columns") or []]
            tasks.append(d)
        except HTTPException as exc:
            errors.append(f"{name}: {exc.detail}")
    return {"server": server, "tasks": tasks, "errors": errors}


async def list_servers() -> dict[str, Any]:
    """Every enabled MCP server that implements the task-server contract, with its tasks."""
    out, errors = [], []
    for spec in [s for s in mcp_registry.load_config() if s.enabled and s.kind == "task"]:
        if spec.oauth and spec.transport != "stdio":
            from . import mcp_oauth

            if spec.name not in mcp_oauth.authorised_servers():
                errors.append(f"{spec.name}: needs sign-in (Connectors -> {spec.name} -> Connect)")
                continue
        try:
            tools = await asyncio.wait_for(mcp_registry.list_tools(spec), 60)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec.name}: {mcp_registry.describe_exception(exc)}")
            continue
        names = {t["function"]["name"].split("__", 1)[-1] for t in tools}
        if not {"task_list", "task_describe", "harness_export_rows", "harness_evaluate"} <= names:
            continue
        try:
            listing = await call(spec.name, "task_list", {}, timeout_s=900)
            out.append({"server": spec.name, "tasks": listing.get("tasks") or [], "errors": listing.get("errors") or []})
        except HTTPException as exc:
            errors.append(f"{spec.name}: {exc.detail}")
    return {"servers": out, "errors": errors}
