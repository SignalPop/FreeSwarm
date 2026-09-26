"""Decile studies ("deci-plots"): does a signal's decile predict the forward return?

This file is the pure numpy/polars core. It is imported by the control plane's tests
directly, and shipped as text into the sandbox (inlined into the harness in deciplot.py)
where it runs on the IN-SAMPLE mirror -- so it must not import anything from the app.
Polars, not pandas: a study resamples 700k+ ten-second bars five ways, and pandas' groupby
and rolling quantiles made that the slow part.

A decile plot is the most honest first look at a signal: bucket every bar by the signal's
decile, and show the average forward return in each bucket. A real relationship climbs (or
falls) across the ten bars and holds in every sub-period; noise zig-zags. It is also the
easiest study to fool yourself with, in two ways this module is built to rule out:

* **Full-sample deciles are look-ahead.** A full-sample qcut places the edges using the
  whole period, so a bar in January is bucketed with knowledge of December's distribution.
  Here the edges for a bar come from PAST data only: by default the previous
  ``window_days`` complete sessions (never the current one), or with ``window_bars`` the
  trailing N bars excluding the current bar. The warm-up period, before the window is
  full, is skipped rather than bucketed on a partial window. Appending future rows never
  changes an earlier bar's decile (tested).
* **Forward returns can straddle what must stay unseen.** Rows at or after ``cut`` (the
  objective's split) are dropped before anything is computed, so no return is ever
  measured into the holdout; and a horizon that crosses the session close is dropped
  rather than measured across the overnight gap.

Timeframes resample the base bars exactly like ``ft.resample``: each bar takes the LAST
value of the signal and the price in it and is stamped at its LAST underlying bar -- the
moment it is complete -- so the signal at a bar is known when the forward return starts.

Times may be passed as a polars Series, a numpy datetime64 array or anything numpy can turn
into one (a pandas Series works; pandas is not needed).
"""

from __future__ import annotations

import ast
import math

import numpy as np
import polars as pl

CORE_VERSION = 2  # bump when the numbers a study produces change: it is part of the cache key
TIMEFRAMES = ("10s", "20s", "30s", "1min", "5min")
HORIZONS = (1, 3, 6, 12)            # in bars OF THE TIMEFRAME (1 = the next bar)
WINDOW_DAYS = 20
SUB_PERIODS = 3
MIN_DECILE_N = 30
QS = np.arange(1, 10) / 10.0

# ---------------------------------------------------------------------------------------
# Expressions: a column, or arithmetic over columns
# ---------------------------------------------------------------------------------------
# The same vocabulary as objectives.series_expression (which validates the SQL form on the
# host); evaluated here with numpy because the sandbox has no DuckDB. Only these nodes are
# allowed -- it is spliced into nothing, but an expression is still agent input.
_FUNCS = {
    "abs": lambda a: np.abs(a), "ln": lambda a: np.log(a), "log": lambda a: np.log(a),
    "sqrt": lambda a: np.sqrt(a), "exp": lambda a: np.exp(a), "sign": lambda a: np.sign(a),
    "power": lambda a, b: np.power(a, b), "pow": lambda a, b: np.power(a, b),
    "greatest": lambda *a: np.nanmax(np.vstack(np.broadcast_arrays(*a)), axis=0),
    "least": lambda *a: np.nanmin(np.vstack(np.broadcast_arrays(*a)), axis=0),
    "nullif": lambda a, b: np.where(np.asarray(a) == np.asarray(b), np.nan, a),
    "coalesce": lambda *a: _coalesce(*a),
}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}


def _coalesce(*arrays):
    out = np.array(np.broadcast_arrays(*arrays)[0], dtype=float)
    for a in arrays[1:]:
        a = np.broadcast_to(np.asarray(a, dtype=float), out.shape)
        out = np.where(np.isnan(out), a, out)
    return out


def _parse(expr: str) -> ast.Expression:
    try:
        return ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse the expression {expr!r}: {exc.msg}") from None


def expression_columns(expr: str, columns: list[str]) -> list[str]:
    """The columns an expression reads (validating it). A bare column name -- even one that
    is not a Python identifier -- is itself."""
    if expr in columns:
        return [expr]
    names = {c.lower(): c for c in columns}
    used: list[str] = []

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOPS:
                raise ValueError(f"{type(node.op).__name__} is not allowed -- use + - * / and power(a, b)")
            return walk(node.left) or walk(node.right)
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ValueError(f"{type(node.op).__name__} is not allowed")
            return walk(node.operand)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):          # "Col Name": a quoted column, as in SQL
                col = names.get(node.value.lower())
                if col is None:
                    raise ValueError(f"unknown column {node.value!r}")
                used.append(col)
            elif not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError(f"constant {node.value!r} is not allowed")
            return None
        if isinstance(node, ast.Name):
            col = names.get(node.id.lower())
            if col is None:
                raise ValueError(f"unknown column {node.id!r}")
            used.append(col)
            return None
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id.lower() not in _FUNCS or node.keywords:
                raise ValueError("only abs, ln, sqrt, exp, sign, power, greatest, least, nullif, coalesce may be called")
            for a in node.args:
                walk(a)
            return None
        raise ValueError(f"{type(node).__name__} is not allowed in a signal expression")

    walk(_parse(expr))
    if not used:
        raise ValueError("the expression reads no column")
    return list(dict.fromkeys(used))


def _floats(col) -> np.ndarray:
    """A column (polars, pandas or a sequence) as float64 numpy; unparseable -> NaN."""
    s = col if isinstance(col, pl.Series) else pl.Series(col)
    return s.cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy().astype(float, copy=True)


def eval_expression(df, expr: str) -> np.ndarray:
    """Evaluate a validated expression over df's columns (polars or pandas), as float64
    (inf -> NaN)."""
    cols = list(df.columns)
    if expr in cols:
        out = _floats(df[expr])
        out[~np.isfinite(out)] = np.nan
        return out
    expression_columns(expr, cols)  # validates
    names = {c.lower(): c for c in cols}

    def val(node):
        if isinstance(node, ast.Expression):
            return val(node.body)
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](val(node.left), val(node.right))
        if isinstance(node, ast.UnaryOp):
            v = val(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return _floats(df[names[node.value.lower()]])
            return float(node.value)
        if isinstance(node, ast.Name):
            return _floats(df[names[node.id.lower()]])
        if isinstance(node, ast.Call):
            return _FUNCS[node.func.id.lower()](*[val(a) for a in node.args])
        raise ValueError(type(node).__name__)

    with np.errstate(all="ignore"):
        out = np.broadcast_to(np.asarray(val(_parse(expr)), dtype=float), (len(df),)).copy()
    out[~np.isfinite(out)] = np.nan
    return out


# ---------------------------------------------------------------------------------------
# Times and bars
# ---------------------------------------------------------------------------------------
def _times(t) -> pl.Series:
    """Any time column as a polars Datetime series (naive, as the harness passes it)."""
    if isinstance(t, pl.Series):
        return t.cast(pl.Datetime("us")).alias("t")
    return pl.Series("t", np.asarray(t, dtype="datetime64[us]"))


def _every(rule: str) -> str:
    """ft/pandas-style bar sizes ("10s", "1min", "5min", "1h") in polars' duration syntax."""
    r = rule.strip().lower()
    return r[:-3] + "m" if r.endswith("min") else r


def resample_last(t, x: np.ndarray, price: np.ndarray, rule: str) -> pl.DataFrame:
    """Bars of `rule`: the signal's and the price's LAST value in each bar, stamped at the
    bar's LAST underlying timestamp (as ft.resample does -- causal as stamped). Columns
    t, x, p, day."""
    frame = pl.DataFrame({"t": _times(t), "x": np.asarray(x, dtype=float), "p": np.asarray(price, dtype=float)})
    frame = frame.drop_nulls("t").sort("t", maintain_order=True)
    # The latest KNOWN value in the bar (NaN/null skipped), still from inside the bar.
    known = lambda c: pl.col(c).fill_nan(None).drop_nulls().last().fill_null(float("nan"))  # noqa: E731
    out = (frame.group_by(pl.col("t").dt.truncate(_every(rule)).alias("_bin"), maintain_order=True)
           .agg(pl.col("t").last(), known("x").alias("x"), known("p").alias("p"))
           .sort("_bin").drop("_bin"))
    return out.with_columns(pl.col("t").dt.truncate("1d").alias("day"))


# ---------------------------------------------------------------------------------------
# Rolling (causal) decile edges
# ---------------------------------------------------------------------------------------
def rolling_edges(t, x: np.ndarray, window_days: int = WINDOW_DAYS,
                  window_bars: int | None = None, min_obs: int = 50) -> np.ndarray:
    """(n, 9) decile edges per bar, from PAST data only; NaN rows = warm-up (not bucketed).

    Default: the edges for every bar of session d are the deciles of the signal over the
    `window_days` complete sessions before d -- the current session is never included, so
    no intraday value leaks into its own bucket. With `window_bars`: the trailing N bars
    EXCLUDING the current one. Either way the first window is skipped, not estimated on less.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    edges = np.full((n, 9), np.nan)
    if n == 0:
        return edges
    if window_bars:
        w = int(window_bars)
        # shift(1) excludes the current bar; NaN -> null so a window needs w KNOWN values.
        s = pl.Series("x", x).fill_nan(None).shift(1)
        for j, q in enumerate(QS):
            edges[:, j] = (s.rolling_quantile(float(q), interpolation="linear", window_size=w, min_samples=w)
                           .fill_null(float("nan")).to_numpy())
        return edges
    day = _times(t).dt.truncate("1d").to_numpy()
    # Sessions in time order; rows must already be sorted by time.
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], n]
    for k in range(int(window_days), len(starts)):
        pool = x[starts[k - int(window_days)]:starts[k]]          # the previous sessions only
        pool = pool[np.isfinite(pool)]
        if len(pool) < min_obs:
            continue
        edges[starts[k]:ends[k]] = np.quantile(pool, QS)
    return edges


def assign_deciles(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Decile 0..9 of each value against its own row's edges; -1 where unknown (warm-up or
    a missing signal). A value equal to an edge goes to the lower bucket."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x) & np.isfinite(edges).all(axis=1)
    dec = np.full(len(x), -1, dtype=int)
    dec[ok] = (x[ok, None] > edges[ok]).sum(axis=1)
    return dec


# ---------------------------------------------------------------------------------------
# Forward returns (within the session, never past the cut)
# ---------------------------------------------------------------------------------------
def forward_returns_bps(bars: pl.DataFrame, h: int) -> np.ndarray:
    """Return from bar i's close to bar i+h's close, in bps; NaN when i+h is another session
    or does not exist (rows at/after the cut were removed before, so the last bars of the
    in-sample period simply have no forward return -- they are dropped, never filled)."""
    p = bars["p"].to_numpy().astype(float)
    day = bars["day"].to_numpy()
    n = len(p)
    out = np.full(n, np.nan)
    if n <= h:
        return out
    same = day[h:] == day[:-h]
    with np.errstate(all="ignore"):
        r = (p[h:] / p[:-h] - 1.0) * 1e4
    r[~same | ~np.isfinite(r) | (p[:-h] <= 0)] = np.nan
    out[:-h] = r
    return out


# ---------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    ra = pl.Series(np.asarray(a, dtype=float)).rank("average").to_numpy()
    rb = pl.Series(np.asarray(b, dtype=float)).rank("average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _r(v, d=3):
    return None if v is None or not math.isfinite(v) else round(float(v), d)


def decile_table(dec: np.ndarray, ret: np.ndarray, h: int) -> dict:
    """Per-decile mean / median / hit / n / t, monotonicity and the top-minus-bottom spread.

    Forward returns over h bars taken at every bar overlap h-fold, so t-stats use an
    effective sample of n / h (a conservative overlap correction): a t of 3 here is not
    three overlapping copies of a t of 1.7."""
    ok = (dec >= 0) & np.isfinite(ret)
    d, r = dec[ok], ret[ok]
    rows = []
    for k in range(10):
        v = r[d == k]
        n = len(v)
        if n == 0:
            rows.append({"decile": k + 1, "n": 0, "mean_bps": None, "median_bps": None, "hit": None, "t": None})
            continue
        sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
        n_eff = max(1.0, n / h)
        t = float(np.mean(v)) / (sd / math.sqrt(n_eff)) if sd > 0 else None
        rows.append({"decile": k + 1, "n": int(n), "mean_bps": _r(np.mean(v), 4), "median_bps": _r(np.median(v), 4),
                     "hit": _r(np.mean(v > 0), 4), "t": _r(t, 2)})
    usable = [x for x in rows if x["n"] >= MIN_DECILE_N and x["mean_bps"] is not None]
    rho = _spearman(np.array([x["decile"] for x in usable]), np.array([x["mean_bps"] for x in usable])) \
        if len(usable) >= 5 else None
    top, bot = r[d == 9], r[d == 0]
    spread = t_spread = None
    if len(top) >= MIN_DECILE_N and len(bot) >= MIN_DECILE_N:
        spread = float(np.mean(top) - np.mean(bot))
        se = math.sqrt(np.var(top, ddof=1) / max(1.0, len(top) / h) + np.var(bot, ddof=1) / max(1.0, len(bot) / h))
        t_spread = spread / se if se > 0 else None
    return {"deciles": rows, "n": int(len(r)), "spearman": _r(rho), "spread_bps": _r(spread, 4),
            "t_spread": _r(t_spread, 2), "mean_all_bps": _r(np.mean(r), 4) if len(r) else None}


def verdict(t_spread, spearman, consistency) -> str:
    """One word for a (timeframe, horizon) cell: can the team build on it?"""
    t, rho = abs(t_spread or 0.0), abs(spearman or 0.0)
    if t < 2 and rho < 0.5:
        return "flat"
    if t >= 2 and consistency is not None and consistency < 1:
        return "unstable"
    if t >= 3 and rho >= 0.7:
        return "monotone"
    if t >= 2 and rho < 0.5:
        return "extremes"
    return "weak"


def _stamp(v) -> str:
    """A timestamp as 'YYYY-MM-DD HH:MM:SS' (what the console and the brief show)."""
    return str(np.datetime64(v, "s")).replace("T", " ")


def study_timeframe(t, x: np.ndarray, price: np.ndarray, rule: str, horizons=HORIZONS,
                    window_days: int = WINDOW_DAYS, window_bars: int | None = None,
                    sub_periods: int = SUB_PERIODS) -> dict:
    """One timeframe: resample, bucket by rolling deciles, measure every horizon, and the
    same statistics in `sub_periods` sequential slices of the bucketed period."""
    bars = resample_last(t, x, price, rule)
    bx = bars["x"].to_numpy().astype(float)
    bt = bars["t"].to_numpy()
    edges = rolling_edges(bars["t"], bx, window_days, window_bars)
    dec = assign_deciles(bx, edges)
    live = np.flatnonzero(dec >= 0)
    out: dict = {"bars": int(len(bars)), "bucketed": int(len(live)),
                 "warmup_until": _stamp(bt[live[0]]) if len(live) else None,
                 "from": _stamp(bt[0]) if len(bars) else None,
                 "to": _stamp(bt[-1]) if len(bars) else None, "horizons": {}}
    if not len(live):
        return out
    # Sub-periods split the bucketed sessions into equal runs of whole days.
    days = bars["day"].to_numpy()
    live_days = np.unique(days[live])
    chunks = [c for c in np.array_split(live_days, max(1, sub_periods)) if len(c)]
    for h in horizons:
        ret = forward_returns_bps(bars, int(h))
        cell = decile_table(dec, ret, int(h))
        periods = []
        for c in chunks:
            m = np.isin(days, c)
            sub = decile_table(np.where(m, dec, -1), ret, int(h))
            periods.append({"from": str(np.datetime64(c[0], "D")), "to": str(np.datetime64(c[-1], "D")),
                            "spread_bps": sub["spread_bps"], "t_spread": sub["t_spread"], "spearman": sub["spearman"],
                            "mean_by_decile": [d["mean_bps"] for d in sub["deciles"]]})
        consistency = None
        if cell["spread_bps"] and cell["spread_bps"] != 0:
            signs = [p["spread_bps"] for p in periods if p["spread_bps"] is not None]
            if signs:
                consistency = round(sum(1 for s in signs if s * cell["spread_bps"] > 0) / len(signs), 3)
        cell["periods"] = periods
        cell["consistency"] = consistency
        cell["verdict"] = verdict(cell["t_spread"], cell["spearman"], consistency)
        out["horizons"][str(int(h))] = cell
    return out


def study(t, x: np.ndarray, price: np.ndarray, *, cut: str | None = None,
          timeframes=TIMEFRAMES, horizons=HORIZONS, window_days: int = WINDOW_DAYS,
          window_bars: int | None = None, sub_periods: int = SUB_PERIODS) -> dict:
    """The full study of one signal: every timeframe x horizon, plus a summary.

    Rows at or after `cut` are removed FIRST -- whatever the caller passed, nothing after the
    split is resampled, bucketed or used as an outcome."""
    tt = _times(t).to_numpy()
    x = np.asarray(x, dtype=float)
    price = np.asarray(price, dtype=float)
    keep = ~np.isnat(tt)
    if cut:
        keep = keep & (tt < np.datetime64(str(cut).replace(" ", "T"), "us"))
    tt, x, price = tt[keep], x[keep], price[keep]
    order = np.argsort(tt, kind="stable")
    tt, x, price = tt[order], x[order], price[order]
    res = {"core_version": CORE_VERSION, "cut": cut, "rows": int(len(tt)),
           "coverage": _r(float(np.isfinite(x).mean()) if len(x) else 0.0, 4),
           "window": {"days": None if window_bars else int(window_days), "bars": int(window_bars) if window_bars else None},
           "timeframes": {}}
    for rule in timeframes:
        res["timeframes"][rule] = study_timeframe(tt, x, price, rule, horizons, window_days, window_bars, sub_periods)
    res["summary"] = summarize(res)
    return res


def summarize(res: dict) -> dict:
    """The strongest (timeframe, horizon) cell, per-timeframe bests, and an overall verdict."""
    cells = []
    for rule, tf in (res.get("timeframes") or {}).items():
        for h, c in (tf.get("horizons") or {}).items():
            cells.append({"timeframe": rule, "horizon": int(h), "spread_bps": c.get("spread_bps"),
                          "t_spread": c.get("t_spread"), "spearman": c.get("spearman"),
                          "consistency": c.get("consistency"), "verdict": c.get("verdict"), "n": c.get("n")})
    scored = [c for c in cells if c["t_spread"] is not None]
    best = max(scored, key=lambda c: abs(c["t_spread"]), default=None)
    by_tf = {}
    for rule in res.get("timeframes") or {}:
        own = [c for c in scored if c["timeframe"] == rule]
        b = max(own, key=lambda c: abs(c["t_spread"]), default=None)
        if b:
            by_tf[rule] = {k: b[k] for k in ("horizon", "spread_bps", "t_spread", "spearman", "consistency", "verdict")}
    verdicts = [c["verdict"] for c in cells]
    if any(v == "monotone" for v in verdicts):
        overall = "monotone"
    elif cells and all(v == "flat" for v in verdicts):
        overall = "flat"
    elif any(v == "unstable" for v in verdicts):
        overall = "unstable"
    elif any(v == "extremes" for v in verdicts):
        overall = "extremes"
    else:
        overall = "weak" if cells else "no data"
    return {"best": best, "by_timeframe": by_tf, "verdict": overall,
            "direction": (None if not best or not best["spread_bps"] else ("higher -> up" if best["spread_bps"] > 0 else "higher -> down"))}
