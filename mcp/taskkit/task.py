"""The Task: one problem a data/action MCP offers the swarm -- rows in, actions out, a valuation back.

Every FreeSwarm task server speaks the same interface (see mcp/README.md), whatever it models:

1. **Rows.** A table, one row per time step, sorted by a timestamp column `t` (UTC). Every column
   is a signal a strategy may use; one of them is the **target** the actions are about (a price
   to trade, a power price to arbitrage, a quantity to forecast).
2. **Actions.** The strategy decides one number per row. The server MANAGES them: it turns them
   into the state they drive (a position, a battery's charge, a forecast) under the task's rules
   -- bounds, costs, forced exits -- and can list what happened (`action_log`). The action decided
   at row t may use rows up to and including t and takes effect from row t to row t+1. Actions
   may be sparse: an action holds until the next one; before the first it is `action.initial`.
3. **Valuation.** The managed actions produce a **curve** (daily returns, daily profit, ...) and a
   valuation per segment -- `in_sample` (before `holdout_from`) and `holdout` -- whose `score` ranks
   candidates, plus diagnostics and notes for the agent (in-sample only).

Subclass `Task`, set the class attributes, implement `load_rows()` and `evaluate()` (optionally
`action_log()`), and `taskkit.serve(...)` turns it into the MCP tools. Rows are kept as a polars
DataFrame; `load_rows` may return polars or pandas.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

KEY = "t"
DAY_NS = 86_400_000_000_000


def ts_ns(value: Any) -> int:
    """A timestamp (ISO string, datetime, pandas/numpy timestamp) as int64 nanoseconds UTC: a
    zone-aware value is converted, a naive one is taken as UTC already."""
    import datetime as dt

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        s = value.strip().replace(" ", "T")
        value = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[ns]").astype(np.int64))
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        value = dt.datetime(value.year, value.month, value.day)
    if value.tzinfo is not None:
        value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    epoch = dt.datetime(1970, 1, 1)
    delta = value - epoch
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def iso(ns: int) -> str:
    """int64 nanoseconds UTC -> 'YYYY-MM-DDTHH:MM:SS' (fractions only when present)."""
    s = str(np.datetime64(int(ns), "ns"))
    s = s.rstrip("0").rstrip(".") if "." in s else s
    return s


def cache_dir(default: Path | None = None) -> Path:
    """Where a task keeps generated or joined rows between calls (a stdio server is a fresh
    process per call, so an in-memory cache alone would rebuild on every call)."""
    d = Path(os.environ.get("TASKKIT_CACHE") or default or Path(__file__).resolve().parent.parent / ".cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def keys_ns(df: pl.DataFrame, col: str = KEY) -> np.ndarray:
    """A timestamp column as int64 nanoseconds (whatever its stored resolution)."""
    return df[col].cast(pl.Datetime("ns")).cast(pl.Int64).to_numpy()


def align_actions(keys: np.ndarray, t: np.ndarray, action: np.ndarray, initial: float = 0.0) -> np.ndarray:
    """One action per row: the latest action reported at or before the row's key (as-of,
    backward), `initial` before the first. `keys` and `t` are int64 nanoseconds, `t` sorted."""
    if len(action) == 0:
        return np.full(len(keys), float(initial))
    idx = np.searchsorted(t, keys, side="right") - 1
    out = np.where(idx >= 0, action[np.clip(idx, 0, None)], initial)
    return np.nan_to_num(out.astype(float), nan=initial)


def read_actions(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(t as int64 ns sorted, action as float) from an actions parquet written by
    ft.report_actions: columns `t` (timestamp) and `pos` (the action). Missing actions (NaN)
    are dropped -- the previous one holds; duplicate keys keep the last reported value."""
    df = pl.read_parquet(path)
    if "t" not in df.columns or "pos" not in df.columns:
        raise ValueError("actions file needs columns t and pos (write it with ft.report_actions)")
    tcol = df["t"]
    if isinstance(tcol.dtype, pl.Datetime) and tcol.dtype.time_zone:
        tcol = tcol.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    t = tcol.cast(pl.Datetime("ns")).cast(pl.Int64).to_numpy()
    a = df["pos"].cast(pl.Float64, strict=False).to_numpy()
    keep = np.isfinite(a)
    t, a = t[keep], a[keep]
    order = np.argsort(t, kind="stable")
    t, a = t[order], a[order]
    last = np.r_[t[1:] != t[:-1], True] if len(t) else np.zeros(0, bool)
    return t[last], a[last]


def apply_value_function(res: dict[str, Any], vf: dict[str, Any]) -> dict[str, Any]:
    """Make the chosen value function's statistic the score of every segment. A segment whose
    own score is unrankable for a reason (a `note`, e.g. too few active days) stays unrankable."""
    stat = vf.get("stat") or vf["name"]
    for seg in (res.get("segments") or {}).values():
        if seg.get("score") is None and seg.get("note"):
            continue
        v = seg.get(stat)
        seg["score"] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None
        if seg["score"] is None:
            seg["note"] = f"{vf['name']} is undefined here"
    res["score_name"] = vf["name"]
    res["higher_is_better"] = bool(vf.get("higher_is_better", True))
    res["value_function"] = vf["name"]
    return res


def records(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Rows as JSON-safe records: timestamps as ISO strings, NaN as null."""
    out = df.with_columns(
        [pl.col(c).dt.strftime("%Y-%m-%dT%H:%M:%S") for c, d in df.schema.items() if isinstance(d, (pl.Datetime, pl.Date))]
        + [pl.col(c).fill_nan(None) for c, d in df.schema.items() if d in (pl.Float32, pl.Float64)])
    return out.to_dicts()


class Task:
    """Base class for a task. Subclasses set the attributes and implement load_rows/evaluate."""

    #: Short unique id, [a-z0-9_]. Used in tool calls and objective settings.
    name: str = ""
    #: One line for people choosing a task.
    title: str = ""
    #: What the problem is, for agents and people: the setting, what the target means, what a
    #: good strategy does. Plain language; this is the first thing an agent reads.
    description: str = ""
    #: Rules the agents must follow beyond the generic ones (constraints, hints, pitfalls).
    brief: str = ""
    #: The column the actions are about.
    target: str = ""
    #: What an action is: kind ("position" | "order" | "setpoint" | "value" ...), min/max bounds,
    #: `initial` (the action before the first one), and a description.
    action: dict[str, Any] = {"kind": "value", "min": None, "max": None, "initial": 0.0, "description": ""}
    #: Rows from this DATE on (UTC) are the holdout. A time of day is ignored: the holdout starts at
    #: that day's midnight, so the harness's split, the in-sample rows and every day-level
    #: valuation agree on one boundary.
    holdout_from: str = ""
    #: Optional extra look-ahead cut inside the in-sample period (ISO timestamp); default: its middle.
    mid_cut: str | None = None
    #: Name of the score and its direction, for people and the leaderboard.
    score_name: str = "score"
    higher_is_better: bool = True
    #: Per-column descriptions shown to agents: {column: description}.
    column_notes: dict[str, str] = {}
    #: Columns legitimately known BEFORE their row's time -- forecasts published in advance (a
    #: day-ahead price, a weather forecast). They lead the target by design; the leak scan lists them
    #: apart instead of flagging them. `*` at the end matches a prefix.
    ahead_columns: list[str] = []
    #: Columns the operator may choose as the target instead (a project setting); [] = only `target`.
    target_options: list[str] = []
    #: The value function in words and numbers, for people choosing a task: {"summary": ..., ...}.
    valuation_info: dict[str, Any] = {}
    #: Time zone the console should draw this task's bars in (the data stays UTC).
    display_tz: str = "UTC"
    #: The value functions the operator may choose between (a project setting): each names the
    #: valuation statistic that becomes the score -- {"name", "title", "description",
    #: "higher_is_better", "stat" (a key of the segments; default: the name)}. The first is the
    #: default; [] = only the task's own score.
    value_functions: list[dict[str, Any]] = []
    value_function: str | None = None

    _lock = threading.Lock()

    # ---- to implement ------------------------------------------------------------------------
    def load_rows(self):
        """All rows (in-sample AND holdout), with a timestamp column named `t` (UTC). Polars or
        pandas. Called once per process; cache expensive work on disk under cache_dir()."""
        raise NotImplementedError

    def evaluate(self, rows: pl.DataFrame, actions: np.ndarray) -> dict[str, Any]:
        """Manage and value one action per row (already aligned and bounded: `actions[i]` was
        decided at row i and is in effect from row i to i+1). Return:

        * ``segments``: {"in_sample": {"score": float | None, "note": str?, ...valuation},
          "holdout": {...}} -- `score` None means "not rankable" and `note` says why.
        * ``curve``: [[label, value], ...] one entry per period (e.g. a day), and
          ``curve_kind``: "returns" (compounding) or "additive" (summed).
        * ``diagnostics``: {"in_sample": {...}, "holdout": {...}} free-form numbers.
        * ``notes``: text for the agent about its IN-SAMPLE result (never mention the holdout).
        * ``unranked`` (optional): a reason the candidate must not be ranked.
        """
        raise NotImplementedError

    def action_log(self, rows: pl.DataFrame, actions: np.ndarray, lo_ns: int, hi_ns: int,
                   limit: int) -> dict[str, Any]:
        """What the managed actions did between lo_ns and hi_ns: `events` (records -- trades, a
        charge schedule, ...) and optionally `state` ([[t, value], ...], the managed state per row).
        Default: the rows where the action changed."""
        k = keys_ns(rows)
        prev = np.r_[np.nan, actions[:-1]]
        ch = np.flatnonzero((actions != prev) & (k >= lo_ns) & (k < hi_ns))[:limit]
        return {"events": [{"t": iso(k[i]), "action": float(actions[i])} for i in ch],
                "state": [[iso(k[i]), float(actions[i])] for i in np.flatnonzero((k >= lo_ns) & (k < hi_ns))[:limit * 20]],
                "state_kind": "action", "bars": self.bars(rows, lo_ns, hi_ns)}

    def bars(self, rows: pl.DataFrame, lo_ns: int, hi_ns: int, max_points: int = 2000) -> dict[str, Any]:
        """The target between lo and hi for a drill-down chart: OHLC candles when the rows have
        Open/High/Low/Close, else a line of the target; thinned to at most max_points."""
        k = keys_ns(rows)
        a, b = int(np.searchsorted(k, lo_ns)), int(np.searchsorted(k, hi_ns))
        w = rows.slice(a, b - a)
        step = max(1, len(w) // max_points)
        w = w.gather_every(step) if step > 1 else w
        ohlc = [c for c in ("Open", "High", "Low", "Close") if c in w.columns]
        cols = ohlc if len(ohlc) == 4 else [self.target]
        kk = keys_ns(w)
        vals = [w[c].cast(pl.Float64, strict=False).to_numpy() for c in cols]
        return {"kind": "ohlc" if len(cols) == 4 else "line", "columns": ["t"] + cols,
                "rows": [[iso(kk[i])] + [None if not np.isfinite(v[i]) else round(float(v[i]), 6) for v in vals]
                         for i in range(len(w))], "tz": self.display_tz}

    def version_parts(self) -> list[Any]:
        """What the rows depend on; change any part and cached exports are rebuilt."""
        return [self.name]

    # ---- provided --------------------------------------------------------------------------
    def rows(self) -> pl.DataFrame:
        with self._lock:
            cached = self.__dict__.get("_rows")
            if cached is None:
                df = self.load_rows()
                if not isinstance(df, pl.DataFrame):
                    df = pl.from_pandas(df)
                if KEY not in df.columns:
                    raise ValueError(f"task {self.name}: rows need a timestamp column {KEY!r}")
                t = df[KEY]
                if t.dtype == pl.Utf8:
                    t = t.str.to_datetime()
                if isinstance(t.dtype, pl.Datetime) and t.dtype.time_zone:
                    t = t.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
                df = df.with_columns(t.cast(pl.Datetime("ns")).alias(KEY))
                df = df.sort(KEY, maintain_order=True).unique(KEY, keep="last", maintain_order=True)
                self.__dict__["_rows"] = cached = df
            return cached

    def options(self) -> list[str]:
        return list(self.target_options) or [self.target]

    def value_fns(self) -> list[dict[str, Any]]:
        return list(self.value_functions) or [{
            "name": self.score_name, "title": self.score_name, "stat": "score",
            "description": self.valuation_info.get("summary", ""), "higher_is_better": self.higher_is_better}]

    def active_value_fn(self) -> dict[str, Any]:
        fns = self.value_fns()
        return next((f for f in fns if f["name"] == self.value_function), fns[0])

    def with_value_function(self, name: str | None) -> "Task":
        """This task scored by another of its value functions (the operator's project setting)."""
        if not name or name == self.active_value_fn()["name"]:
            return self
        names = [f["name"] for f in self.value_fns()]
        if name not in names:
            raise ValueError(f"task {self.name}: value function {name!r} is not one of {names}")
        self.rows()
        t = copy.copy(self)
        t.value_function = name
        return t

    def with_options(self, target: str | None = None, value_function: str | None = None) -> "Task":
        return self.with_target(target).with_value_function(value_function)

    def with_target(self, target: str | None) -> "Task":
        """This task valued on another of its `target_options` (the operator's project setting)."""
        if not target or target == self.target:
            return self
        if target not in self.options():
            raise ValueError(f"task {self.name}: target {target!r} is not one of {self.options()}")
        self.rows()                                           # load once; the copy shares the rows
        t = copy.copy(self)
        t.target = target
        return t

    def version(self) -> str:
        return hashlib.sha1(json.dumps(self.version_parts(), default=str).encode()).hexdigest()[:12]

    def holdout_ns(self) -> int:
        if not self.holdout_from:
            return 2**62
        return ts_ns(self.holdout_from) // DAY_NS * DAY_NS          # midnight UTC of that day

    def in_sample(self) -> pl.DataFrame:
        r = self.rows()
        return r.filter(pl.col(KEY).cast(pl.Int64) < self.holdout_ns())

    def cuts(self) -> list[str]:
        """Look-ahead cuts: the holdout boundary and one inside the in-sample period."""
        out = [iso(self.holdout_ns())] if self.holdout_from else []
        ins = self.in_sample()
        mid = self.mid_cut or (iso(keys_ns(ins)[len(ins) // 2]) if len(ins) > 10 else None)
        if mid:
            out.append(iso(ts_ns(mid)))
        return out

    def columns(self) -> list[dict[str, Any]]:
        out = []
        for c, d in self.rows().schema.items():
            role = "key" if c == KEY else "target" if c == self.target else "signal"
            out.append({"name": c, "dtype": str(d), "role": role, "description": self.column_notes.get(c, "")})
        return out

    def describe(self) -> dict[str, Any]:
        r = self.rows()
        ins = self.in_sample()
        k, ki = keys_ns(r), keys_ns(ins)
        return {
            "name": self.name, "title": self.title, "description": " ".join(self.description.split()),
            "brief": " ".join(self.brief.split()), "key": KEY, "target": self.target,
            "target_options": self.options(), "valuation": dict(self.valuation_info), "display_tz": self.display_tz,
            "shape": {"rows": int(len(r)), "columns": int(len(r.columns)), "first": iso(k[0]) if len(k) else None,
                      "last": iso(k[-1]) if len(k) else None,
                      "step_s": float(np.median(np.diff(k)) / 1e9) if len(k) > 1 else None},
            "action": {"initial": 0.0, **self.action},
            "score": {"name": self.active_value_fn()["name"],
                      "higher_is_better": bool(self.active_value_fn().get("higher_is_better", True))},
            "value_function": self.active_value_fn()["name"],
            "value_functions": [{k: f.get(k) for k in ("name", "title", "description", "higher_is_better")}
                                for f in self.value_fns()],
            "rows": int(len(r)), "in_sample_rows": int(len(ins)),
            "first": iso(k[0]) if len(k) else None, "last_in_sample": iso(ki[-1]) if len(ki) else None,
            "holdout_from": iso(self.holdout_ns()) if self.holdout_from else None,
            "cuts": self.cuts(), "columns": self.columns(), "version": self.version(),
            "ahead_columns": list(self.ahead_columns),
        }

    def export(self, path: str | Path, until: str | None = None) -> dict[str, Any]:
        """Write the rows with key < `until` (all rows without it) to `path` as parquet, plus a
        task.json beside it. This is what strategy code sees -- truncated copies are how the
        harness proves a strategy does not use the future."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        r = self.rows()
        if until:
            r = r.filter(pl.col(KEY).cast(pl.Int64) < ts_ns(until))
        tmp = path.with_name(path.name + ".tmp")
        r.write_parquet(tmp)
        tmp.replace(path)
        # The copy strategy code reads (ft.task()): everything but the look-ahead cuts -- a
        # strategy that knew where the harness cuts could behave honestly only there.
        desc = {k: v for k, v in self.describe().items() if k not in ("cuts", "version")}
        (path.parent / "task.json").write_text(json.dumps(desc, indent=1), encoding="utf-8")
        return {"path": str(path), "rows": int(len(r)), "until": until, "version": self.version()}

    def managed_actions(self, actions_path: str | Path) -> tuple[pl.DataFrame, np.ndarray | None, str | None]:
        """(rows, one bounded action per row, problem) from an actions parquet."""
        r = self.rows()
        t, a = read_actions(actions_path)
        keys = keys_ns(r)
        inside = int(((t >= keys[0]) & (t <= keys[-1])).sum()) if len(t) and len(keys) else 0
        if len(t) == 0 or inside < max(1, 0.01 * len(t)):
            return r, None, (f"{len(t)} actions, {inside} of them inside the rows' time range -- index the actions "
                             "by the rows' `t` column (ft.report_actions(pd.Series(values, index=rows['t'])))")
        aligned = align_actions(keys, t, a, float(self.action.get("initial") or 0.0))
        lo, hi = self.action.get("min"), self.action.get("max")
        if lo is not None or hi is not None:
            aligned = np.clip(aligned, -np.inf if lo is None else lo, np.inf if hi is None else hi)
        return r, aligned, None

    def evaluate_file(self, actions_path: str | Path) -> dict[str, Any]:
        """Align an actions parquet onto the rows, bound it, and value it."""
        r, aligned, problem = self.managed_actions(actions_path)
        if problem:
            return {"problem": problem}
        res = self.evaluate(r, aligned)
        res.setdefault("score_name", self.score_name)
        res.setdefault("higher_is_better", self.higher_is_better)
        vf = self.active_value_fn()
        if vf.get("stat", vf["name"]) != "score" and "segments" in res:
            apply_value_function(res, vf)
        res["actions"] = {"reported": int(len(read_actions(actions_path)[0])),
                          "changes": int((np.diff(aligned) != 0).sum())}
        return res

    def action_log_file(self, actions_path: str | Path, start: str | None = None, end: str | None = None,
                        limit: int = 500) -> dict[str, Any]:
        r, aligned, problem = self.managed_actions(actions_path)
        if problem:
            return {"problem": problem}
        lo = ts_ns(start) if start else -(2**62)
        hi = ts_ns(end) if end else 2**62
        return self.action_log(r, aligned, lo, hi, max(1, min(limit, 5000)))

    def sample(self, limit: int = 50, offset: int = 0, columns: list[str] | None = None) -> dict[str, Any]:
        """In-sample rows only, as records -- never the holdout."""
        ins = self.in_sample()
        if columns:
            missing = [c for c in columns if c not in ins.columns]
            if missing:
                raise ValueError(f"unknown columns {missing}; see task_describe")
            ins = ins.select([KEY] + [c for c in columns if c != KEY])
        part = ins.slice(max(0, offset), max(1, min(limit, 500)))
        return {"rows": records(part), "offset": offset, "in_sample_rows": int(len(ins))}

    def query(self, sql: str, limit: int = 200) -> dict[str, Any]:
        """A read-only SQL SELECT over the IN-SAMPLE rows, registered as the table `rows`
        (polars SQL). At most `limit` (<= 1000) rows come back."""
        s = sql.strip().rstrip(";")
        if not s.lower().startswith(("select", "with")):
            raise ValueError("only a SELECT (or WITH ... SELECT) over the table `rows` is allowed")
        ctx = pl.SQLContext(rows=self.in_sample().lazy(), eager=False)
        out = ctx.execute(s).head(max(1, min(limit, 1000))).collect()
        return {"columns": out.columns, "rows": records(out), "returned": len(out),
                "note": "in-sample rows only (the holdout is hidden)"}

    def column_stats(self) -> dict[str, Any]:
        """Per-column summary over the in-sample rows."""
        ins = self.in_sample()
        out: dict[str, Any] = {}
        for c, d in ins.schema.items():
            s = ins[c]
            if c == KEY:
                k = keys_ns(ins)
                out[c] = {"first": iso(k[0]) if len(k) else None, "last": iso(k[-1]) if len(k) else None,
                          "median_step_s": float(np.median(np.diff(k)) / 1e9) if len(k) > 1 else None}
            elif d.is_numeric():
                x = s.cast(pl.Float64).fill_nan(None).drop_nulls()
                qs = x.quantile(0.25), x.median(), x.quantile(0.75)
                out[c] = {"count": int(len(x)), "missing": int(len(s) - len(x)),
                          **{k: (None if v is None else round(float(v), 6)) for k, v in
                             {"mean": x.mean(), "std": x.std(), "min": x.min(), "25%": qs[0], "50%": qs[1],
                              "75%": qs[2], "max": x.max()}.items()}}
            else:
                vc = s.value_counts(sort=True).head(5)
                out[c] = {"distinct": int(s.n_unique()), "missing": int(s.null_count()),
                          "top": {str(row[0]): int(row[1]) for row in vc.iter_rows()}}
        return {"columns": out, "in_sample_rows": int(len(ins))}

    def leak_scan(self, top: int = 15) -> dict[str, Any]:
        """Columns that look like they know the future, in-sample.

        For each numeric column, the correlation of its CHANGE at row t with the target's move
        over row t (into t) and over row t+1 (the next row). A column filed under the right time
        explains the current move at least as well as the next one; one whose change predicts the
        NEXT move clearly better was probably computed after its timestamp (a late snapshot), and
        a strategy trading it would be using the future. Such columns should be delayed until the
        peak sits at the current row. Heuristic: a genuinely leading signal can look the same, so
        read it as a prompt to check how the column is made."""
        ins = self.in_sample()
        if len(ins) < 3:
            return {"columns": [], "suspects": [], "declared_ahead": [], "how": "too few in-sample rows to scan"}
        y = ins[self.target].cast(pl.Float64, strict=False).to_numpy()
        day = keys_ns(ins) // DAY_NS
        same = np.r_[False, day[1:] == day[:-1]]
        with np.errstate(all="ignore"):
            if np.nanmin(y) > 0:                               # a price: its relative move
                move = np.where(same, np.r_[np.nan, y[1:] / y[:-1] - 1.0], np.nan)
            else:                                              # can be 0 or negative: its change
                move = np.where(same, np.r_[np.nan, np.diff(y)], np.nan)
        nxt = np.r_[move[1:], np.nan]

        def corr(a: np.ndarray, b: np.ndarray) -> float | None:
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 200 or a[m].std() == 0 or b[m].std() == 0:
                return None
            return round(float(np.corrcoef(a[m], b[m])[0, 1]), 4)

        rows = []
        for c, d in ins.schema.items():
            if c in (KEY, self.target) or not d.is_numeric():
                continue
            x = ins[c].cast(pl.Float64).to_numpy()
            with np.errstate(all="ignore"):
                dx = np.where(same, np.r_[np.nan, np.diff(x)], np.nan)
                now, ahead = corr(dx, move), corr(dx, nxt)
            if now is None or ahead is None:
                continue
            leads = abs(ahead) > max(0.05, 1.5 * abs(now))
            declared = any(c == p or (p.endswith("*") and c.startswith(p[:-1])) for p in self.ahead_columns)
            rows.append({"column": c, "change_vs_current_move": now, "change_vs_next_move": ahead,
                         "suspect": leads and not declared, **({"declared_ahead": True} if declared else {})})
        rows.sort(key=lambda r: -abs(r["change_vs_next_move"]))
        return {"columns": rows[:top], "suspects": [r["column"] for r in rows if r["suspect"]],
                "declared_ahead": [r["column"] for r in rows if r.get("declared_ahead")],
                "how": "a column whose change predicts the NEXT row's move better than the current one was "
                       "probably filed before it was known"}
