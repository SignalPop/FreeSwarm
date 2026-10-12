"""The data audit: verify a task's rows day by day and name the date range they can be trusted on.

A feed that dies quietly does not raise an error: its columns go empty (null) or freeze at one
value while the target keeps coming, so every strategy that reads them stops trading. If that
happens inside the holdout, every candidate scores badly and nothing in the scores says why.
The audit finds it from the rows themselves -- the same rows the server serves -- for any task:

* a **day** is one UTC calendar day of rows. A day is *short* when it has far fewer rows than
  the typical day (a gap in the recording), *bad* when it is very short or any checked column has
  a problem on it.
* a **column** has a problem on a day when it is *missing* (fewer than half the day's rows have a
  value) or *frozen* (one value all day, for a column that normally moves within a day).
  A column that is wrong on many scattered days besides its long outages (`isolated_share`) is
  patchy by nature -- a third-largest wall that often does not exist -- and is reported as
  *intermittent* instead of failing every day it is empty.
* the **verified range** is the longest stretch of days without a calendar gap longer than
  `max_gap_days` or a run of more than `max_bad_run` bad days (a feed that died), trimmed to its
  first and last good day. Bad days left inside it are
  listed as *exceptions*; the data outside it is listed as *excluded*, with the columns to blame.

`audit_rows(df, target, holdout_ns)` returns a JSON-able report (see the keys at the bottom of
`audit_rows`). Polars only; one group-by over the rows, then numpy over a days x columns grid.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl

KEY = "t"
DAY_NS = 86_400_000_000_000

DEFAULTS: dict[str, Any] = {
    "missing_share": 0.5,       # a column is missing on a day when < this share of its rows have a value
    "sparse_share": 0.2,        # a column with a value on fewer days than this is sparse: listed, not checked
    "moving_share": 0.95,       # a column that moves within the day on >= this share of its days (in a typical
                                #   month) can be frozen; one that often sits still (a strike level) cannot
    "frozen_min_rows": 10,      # days with fewer rows are never judged frozen (too few values to move)
    "run_days": 3,              # a problem lasting this many days in a row is a dead feed
    "isolated_share": 0.05,     # a column wrong on more of its days than this, apart from long outages, is
                                #   intermittent (patchy by nature): reported, never failing a day
    "short_share": 0.4,         # a day with < this share of the typical rows is bad
    "light_share": 0.9,         # ... and < this share is listed as a short day (half-days, gaps)
    "max_bad_run": 20,          # more bad days in a row than this (about a month) end the verified range;
                                #   shorter runs inside it are listed as exceptions
    "max_gap_days": 31,         # a calendar gap longer than this (or 5x the usual step) ends it too
    "bad_share": 0.25,          # the verdict is bad when more of the holdout (or in-sample) days than this
                                #   are unverified; warn for fewer
    "ignore_columns": [],       # columns (or prefixes ending in *) the audit leaves out
    "list_limit": 40,           # how many days / columns each list shows
}


def _match(c: str, pats: list[str]) -> bool:
    return any(c == p or (p.endswith("*") and c.startswith(p[:-1])) for p in pats)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end exclusive) of every run of True in a 1-d bool array."""
    if not len(mask):
        return []
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _long_runs(p: np.ndarray, n: int) -> np.ndarray:
    """Per row of a (columns x days) bool grid: True where the cell is in a run of >= n Trues."""
    out = np.zeros_like(p)
    for i in range(p.shape[0]):
        for a, b in _runs(p[i]):
            if b - a >= n:
                out[i, a:b] = True
    return out


def _names(cols: list[str], limit: int = 12) -> str:
    return ", ".join(cols[:limit]) + (f" and {len(cols) - limit} more" if len(cols) > limit else "")


def day_grid(df: pl.DataFrame, columns: list[str], key: str = KEY) -> pl.DataFrame:
    """One row per UTC day: `__n` rows, and per column i `n{i}` (values present) and `v{i}`
    (whether it moved within the day)."""
    exprs: list[pl.Expr] = [pl.len().alias("__n")]
    schema = df.schema
    for i, c in enumerate(columns):
        d = schema[c]
        if d.is_numeric() or d == pl.Boolean:
            x = pl.col(c).cast(pl.Float64)
            x = x.fill_nan(None)
            exprs += [x.count().alias(f"n{i}"), (x.max() > x.min()).fill_null(False).alias(f"v{i}")]
        else:
            exprs += [pl.col(c).count().alias(f"n{i}"), (pl.col(c).drop_nulls().n_unique() > 1).alias(f"v{i}")]
    t = pl.col(key)
    if df.schema[key] != pl.Date:
        t = t.dt.date()
    return df.lazy().group_by(t.alias("__d")).agg(exprs).sort("__d").collect()


def audit_rows(df: pl.DataFrame, target: str | None = None, holdout_ns: int | None = None,
               settings: dict[str, Any] | None = None, key: str = KEY) -> dict[str, Any]:
    """Verify `df` (a timestamp column `key`, UTC) day by day: see the module docstring."""
    s = {**DEFAULTS, **(settings or {})}
    lim = int(s["list_limit"])
    ignore = list(s.get("ignore_columns") or [])
    cols = [c for c, d in df.schema.items()
            if c != key and not _match(c, ignore) and (d.is_numeric() or d in (pl.Boolean, pl.Utf8, pl.Categorical))]
    if not len(df) or not cols:
        return {"status": "bad", "summary": "no rows to verify" if not len(df) else "no columns to verify",
                "verified": None, "data": {"rows": int(len(df)), "days": 0}, "settings": s}

    g = day_grid(df, cols, key)
    days = g["__d"].to_list()                                   # datetime.date, sorted
    nd = len(days)
    rows = g["__n"].to_numpy().astype(float)
    have = np.vstack([g[f"n{i}"].to_numpy() for i in range(len(cols))]).astype(float)   # columns x days
    moved = np.vstack([g[f"v{i}"].to_numpy() for i in range(len(cols))]).astype(bool)
    typical = float(np.median(rows))

    alive = have >= s["missing_share"] * rows                   # a value on enough of the day's rows
    alive_share = alive.mean(axis=1)
    checked = alive_share >= s["sparse_share"]
    if target in cols:
        checked[cols.index(target)] = True                      # the target is always checked
    # How often a column moves within its days, in a typical month: a feed frozen for a few months
    # of years must not make a moving column look sticky.
    month = np.array([d.year * 12 + d.month for d in days])
    per_month = []
    for m in np.unique(month):
        sel = month == m
        n_alive = alive[:, sel].sum(axis=1)
        per_month.append(np.where(n_alive > 0, (moved[:, sel] & alive[:, sel]).sum(axis=1) / np.maximum(n_alive, 1), np.nan))
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a column with no value in any month
        moving_share = np.nan_to_num(np.nanmedian(np.vstack(per_month), axis=0), nan=0.0)
    can_freeze = moving_share >= s["moving_share"]
    big_day = rows >= s["frozen_min_rows"]

    missing = ~alive
    frozen = alive & ~moved & can_freeze[:, None] & big_day[None, :]
    problem = (missing | frozen) & checked[:, None]

    long = _long_runs(problem, int(s["run_days"]))
    rest = (~long).sum(axis=1)
    iso_share = np.where(rest > 0, (problem & ~long).sum(axis=1) / np.maximum(rest, 1), 0.0)
    intermittent = checked & (iso_share > s["isolated_share"])
    flagged = problem & ~intermittent[:, None]

    very_short = rows < s["short_share"] * typical
    light = rows < s["light_share"] * typical
    bad = very_short | flagged.any(axis=0)
    good = ~bad

    # ---- the verified range: the longest stretch without a long gap or a long run of bad days
    ords = np.array([d.toordinal() for d in days])
    step = float(np.median(np.diff(ords))) if nd > 1 else 1.0
    max_gap = max(float(s["max_gap_days"]), 5 * step)
    breaks = np.zeros(nd, bool)                                 # True: a new stretch starts at this day
    long_bad: list[tuple[int, int]] = []                        # runs of bad days that end a stretch
    gaps = []
    for i in range(1, nd):
        gap = int(ords[i] - ords[i - 1])
        if gap > max_gap:
            breaks[i] = True
        if gap > max(4, 2 * step):
            gaps.append({"after": days[i - 1].isoformat(), "before": days[i].isoformat(), "calendar_days": gap - 1})
    for a, b in _runs(bad):
        if b - a > s["max_bad_run"]:
            long_bad.append((a, b))
            breaks[a] = True
            if b < nd:
                breaks[b] = True
    best: tuple[int, int, int] | None = None                    # (good days, start, end exclusive)
    starts = [0] + [i for i in range(1, nd) if breaks[i]] + [nd]
    for a, b in zip(starts[:-1], starts[1:]):
        idx = np.flatnonzero(good[a:b])
        if not len(idx):
            continue
        lo, hi = a + int(idx[0]), a + int(idx[-1]) + 1
        cand = (int(good[lo:hi].sum()), lo, hi)
        if best is None or cand[0] >= best[0]:                  # ties: the later stretch
            best = cand
    if best is None:
        lo = hi = 0
    else:
        _, lo, hi = best
    inside = np.zeros(nd, bool)
    inside[lo:hi] = True

    def why(i: int) -> dict[str, Any]:
        miss = [cols[j] for j in np.flatnonzero(flagged[:, i] & missing[:, i])]
        froz = [cols[j] for j in np.flatnonzero(flagged[:, i] & frozen[:, i])]
        out: dict[str, Any] = {"day": days[i].isoformat(), "rows": int(rows[i])}
        parts = []
        if very_short[i]:
            parts.append(f"{int(rows[i])} rows of a typical {typical:.0f}")
        if miss:
            parts.append(f"{len(miss)} column(s) empty: {_names(miss, 6)}")
            out["missing"] = miss[:lim]
        if froz:
            parts.append(f"{len(froz)} column(s) frozen at one value: {_names(froz, 6)}")
            out["frozen"] = froz[:lim]
        out["why"] = "; ".join(parts)
        return out

    def span(a: int, b: int) -> dict[str, Any] | None:
        """What is wrong with the days a..b-1 (outside the verified range), most-blamed columns first."""
        if b <= a:
            return None
        sl = slice(a, b)
        n_bad = int(bad[sl].sum())
        blame = flagged[:, sl].sum(axis=1)
        order = [j for j in np.argsort(-blame, kind="stable") if blame[j] > 0]
        cols_out = [{"column": cols[j], "bad_days": int(blame[j]),
                     "missing_days": int((flagged[j, sl] & missing[j, sl]).sum()),
                     "frozen_days": int((flagged[j, sl] & frozen[j, sl]).sum())} for j in order]
        short_n = int(very_short[sl].sum())
        parts = []
        if order:
            parts.append(f"{len(order)} column(s) empty or frozen: {_names([cols[j] for j in order], 8)}")
        if short_n:
            parts.append(f"{short_n} very short day(s)")
        if not parts and b - a:
            parts.append("separated from the verified range by a gap in the data")
        return {"from": days[a].isoformat(), "to": days[b - 1].isoformat(), "days": b - a, "bad_days": n_bad,
                "why": "; ".join(parts), "columns": cols_out[:lim]}

    excluded = [x for x in (span(0, lo), span(hi, nd)) if x]
    bad_runs = [x for x in (span(a, b) for a, b in long_bad) if x]
    exceptions = [why(i) for i in range(lo, hi) if bad[i]]
    short_days = [{"day": days[i].isoformat(), "rows": int(rows[i])} for i in range(lo, hi) if light[i] and not bad[i]]

    # ---- per column: where each checked column starts and stops being usable
    def group(idx_of: dict[str, int]) -> list[dict[str, Any]]:
        by: dict[int, list[str]] = {}
        for c, i in idx_of.items():
            by.setdefault(i, []).append(c)
        return [{"day": days[i].isoformat(), "count": len(cs), "columns": cs[:lim]} for i, cs in sorted(by.items())]

    ends, begins = {}, {}
    for j, c in enumerate(cols):
        if not checked[j]:
            continue
        ok = np.flatnonzero(~flagged[j])
        if not len(ok):
            ends[c] = begins[c] = -1
            continue
        if nd - 1 - ok[-1] >= s["run_days"]:
            ends[c] = int(ok[-1])
        if ok[0] >= s["run_days"]:
            begins[c] = int(ok[0])
    never = [c for c, i in ends.items() if i < 0]
    ends = {c: i for c, i in ends.items() if i >= 0}
    begins = {c: i for c, i in begins.items() if i >= 0}

    months: dict[str, list[int]] = {}
    for i, d in enumerate(days):
        m = months.setdefault(d.strftime("%Y-%m"), [0, 0, 0])
        m[0] += 1
        m[1] += int(good[i] and inside[i])
        m[2] += int(bad[i])

    first, last = days[0].isoformat(), days[-1].isoformat()
    verified = None if best is None else {
        "from": days[lo].isoformat(), "to": days[hi - 1].isoformat(), "days": hi - lo, "good_days": int(good[lo:hi].sum())}

    def segment(mask: np.ndarray, frm: str | None) -> dict[str, Any]:
        n = int(mask.sum())
        ok = int((mask & inside & good).sum())
        out = {"from": frm, "days": n, "verified_days": ok, "unverified_days": n - ok}
        if n and verified:
            if verified["to"] < last and mask[hi:].any():
                out["unverified_after"] = verified["to"]
            if lo > 0 and mask[:lo].any():
                out["unverified_before"] = verified["from"]
        return out

    hold_seg = in_seg = None
    if holdout_ns is not None and holdout_ns < 2**62:
        hday = dt.date(1970, 1, 1) + dt.timedelta(days=int(holdout_ns // DAY_NS))
        hmask = np.array([d >= hday for d in days])
        hold_seg = segment(hmask, hday.isoformat())
        in_seg = segment(~hmask, first)

    # ---- the verdict
    problems = []
    if verified is None:
        status = "bad"
        problems.append("no day passes the checks")
    else:
        status = "ok"
        if verified["from"] > first or verified["to"] < last or exceptions:
            status = "warn"
        for name, seg in (("holdout", hold_seg), ("in-sample", in_seg)):
            if not seg or not seg["days"] or not seg["unverified_days"]:
                continue
            share = seg["unverified_days"] / seg["days"]
            msg = f"{seg['unverified_days']} of the {name}'s {seg['days']} days are unverified ({share:.0%})"
            if share > s["bad_share"]:
                status = "bad"
                problems.append(msg + (" -- candidates are scored on data that is not there" if name == "holdout" else ""))
            elif name == "holdout":
                status = "warn" if status == "ok" else status
    summary = (f"verified {verified['from']} to {verified['to']} ({verified['good_days']} good of {nd} days; "
               f"data {first} to {last})" if verified else f"nothing verified in {nd} days ({first} to {last})")
    if exceptions:
        summary += f"; {len(exceptions)} bad day(s) inside"
    for x in excluded:
        summary += f"; excluded {x['from']} to {x['to']} ({x['days']} days): {x['why']}"
    if problems:
        summary += ". " + " ".join(problems)

    return {
        "status": status,
        "summary": summary,
        "verified": verified,
        "data": {"first": first, "last": last, "days": nd, "rows": int(len(df)), "typical_rows_per_day": typical,
                 "columns": len(cols), "checked_columns": int(checked.sum())},
        "in_sample": in_seg,
        "holdout": hold_seg,
        "exceptions": exceptions[:lim],
        "exceptions_total": len(exceptions),
        "short_days": short_days[:lim],
        "short_days_total": len(short_days),
        "excluded": excluded,
        "bad_runs": bad_runs[:lim],
        "gaps": gaps[:lim],
        "columns_end": group(ends),
        "columns_start": group(begins),
        "columns_never_usable": never[:lim],
        "intermittent_columns": [{"column": cols[j], "bad_share": round(float(iso_share[j]), 3)}
                                 for j in np.flatnonzero(intermittent)][:lim],
        "sparse_columns": [cols[j] for j in np.flatnonzero(~checked)][:lim],
        "months": [{"month": k, "days": v[0], "verified": v[1], "bad": v[2]} for k, v in months.items()],
        "settings": {k: v for k, v in s.items()},
    }
