"""Ready-made evaluators: turn one action per row into the result a Task.evaluate returns.

* `trading`  -- actions are POSITIONS in the target (a price). Marked to market exactly as the
  FreeSwarm harness prices positions: the position decided at row i earns row i+1's price move,
  costs are paid on every change of position, returns compound per UTC day.
* `forecast` -- actions are PREDICTIONS of the target `horizon` rows ahead. Scored by skill
  against the naive "no change" forecast.

Both split results into in_sample / holdout at the task's `holdout_from`, and keep the agent
notes to in-sample numbers only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from . import metrics as M


def _ns(rows: pl.DataFrame, key: str) -> np.ndarray:
    """Timestamps as int64 NANOSECONDS whatever their stored resolution."""
    return rows[key].cast(pl.Datetime("ns")).cast(pl.Int64).to_numpy()


def _num(rows: pl.DataFrame, col: str) -> np.ndarray:
    return rows[col].cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy()


def _days(t_ns: np.ndarray) -> np.ndarray:
    return (t_ns // 86_400_000_000_000).astype(np.int64)


def _day_label(day: int) -> str:
    return str(np.datetime64(int(day), "D"))


def trading(rows: pl.DataFrame, pos: np.ndarray, *, price: str, holdout_ns: int, cost_bps: float = 1.0,
            flat_each_day: bool = False, periods_per_year: float = 252.0, score: str = "sharpe",
            min_active_days: int = 20, min_side_share: float = 0.0, swing_pct: float = 0.25,
            key: str = "t") -> dict[str, Any]:
    """Score positions in `price`. `pos` is already aligned (one per row) and bounded.

    `flat_each_day`: the position is forced to 0 at each UTC day's last row, so nothing is held
    overnight -- every trade opens and closes the same day. `min_side_share`: longs and shorts
    must each be at least this share of the in-sample trades or the result is `unranked`."""
    t = _ns(rows, key)
    p = _num(rows, price)
    ok = np.isfinite(p) & (p > 0)
    t, p, q = t[ok], p[ok], np.asarray(pos, dtype=float)[ok]
    if len(t) < 3:
        return {"problem": "fewer than 3 rows with a positive price"}
    day = _days(t)
    if flat_each_day:
        last = np.r_[day[1:] != day[:-1], True]
        q = np.where(last, 0.0, q)
    cost = cost_bps / 10_000.0
    q1 = np.r_[0.0, q[:-1]]                       # held over row i (decided at i-1)
    q2 = np.r_[0.0, 0.0, q[:-2]]                  # the one before: q1 - q2 is the trade at i-1
    move = np.r_[0.0, p[1:] / p[:-1] - 1.0]
    gross_bar = q1 * move
    net_bar = gross_bar - cost * np.abs(q1 - q2)
    flip_bar = -gross_bar - cost * np.abs(q1 - q2)
    days, first = np.unique(day, return_index=True)
    bounds = np.r_[first, len(day)]

    def per_day(bar: np.ndarray) -> np.ndarray:
        lg = np.log1p(np.maximum(bar, -0.999999))
        c = np.r_[0.0, np.cumsum(lg)]
        return np.expm1(c[bounds[1:]] - c[bounds[:-1]])

    net, gross, flip = per_day(net_bar), per_day(gross_bar), per_day(flip_bar)
    hday = int(holdout_ns // 86_400_000_000_000)
    seg_masks = {"in_sample": days < hday, "holdout": days >= hday}

    # Trades opened per side (a flip opens one), per segment by the row the trade opened on.
    s, sp = np.sign(q), np.sign(np.r_[0.0, q[:-1]])
    opens = (s != 0) & (s != sp)
    long_open, short_open = opens & (s > 0), opens & (s < 0)
    swing = M.swings(t / 1e9, p, q, swing_pct / 100.0, holdout_ns / 1e9)

    segments: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for name, dm in seg_masks.items():
        if not dm.any():
            continue
        st = M.stats(net[dm], periods_per_year)
        v = st.get(score)
        note = ""
        if st.get("active_days", 0) < min_active_days:
            v, note = None, f"only {st.get('active_days', 0)} active days (need {min_active_days})"
        segments[name] = {**st, "score": v, **({"note": note} if note else {})}
        rm = (t < holdout_ns) if name == "in_sample" else (t >= holdout_ns)
        g = M.stats(gross[dm], periods_per_year)
        f = M.stats(flip[dm], periods_per_year)
        diagnostics[name] = {
            "before_costs": g.get(score), "after_costs": st.get(score), "flipped": f.get(score),
            "trades": {"long": int((long_open & rm).sum()), "short": int((short_open & rm).sum())},
            "trades_per_day": round(float(opens[rm].sum()) / max(1, int(dm.sum())), 3),
            "exposure": round(float((q[rm] != 0).mean()), 4) if rm.any() else None,
            "swings": swing.get(name),
        }
    unranked = None
    ins = diagnostics.get("in_sample") or {}
    if min_side_share and ins:
        lo, sh = ins["trades"]["long"], ins["trades"]["short"]
        n = lo + sh
        if not n:
            unranked = ("no trades: the strategy never opened a position in-sample -- its entry conditions never "
                        "fire together. Loosen the strictest threshold or drop a gate, and check each condition's "
                        "hit rate on its own before combining them.")
        elif min(lo, sh) < min_side_share * n:
            weak = "short" if sh <= lo else "long"
            unranked = (f"one-sided: {lo} long and {sh} short trades in-sample -- {weak} trades must be at least "
                        f"{min_side_share:.0%} of them. Add the mirrored {weak} entry (the same conditions reversed).")
    notes = []
    if ins:
        notes.append(f"In-sample {score}: {ins['after_costs']} after costs, {ins['before_costs']} before, "
                     f"{ins['flipped']} with every position flipped; {ins['trades']['long']} long / "
                     f"{ins['trades']['short']} short trades, {ins['trades_per_day']} trades/day, "
                     f"in the market {ins['exposure']:.0%} of rows." if ins.get("exposure") is not None else "")
        sw = ins.get("swings")
        if sw:
            notes.append(f"Swings (legs of >= {swing_pct}%): net {sw['net']:+d} of {sw['legs']} legs -- up legs caught "
                         f"long {sw['up_caught_long']}, down legs caught short {sw['down_caught_short']}, up legs while "
                         f"short {sw['up_while_short']}, down legs while long {sw['down_while_long']}.")
        if unranked:
            notes.append(unranked)
    return {
        "segments": segments, "diagnostics": diagnostics, "unranked": unranked,
        "curve": [[_day_label(d), round(float(r), 10)] for d, r in zip(days, net)],
        "curve_kind": "returns", "notes": " ".join(n for n in notes if n),
    }


def forecast(rows: pl.DataFrame, pred: np.ndarray, *, target: str, holdout_ns: int, horizon: int = 1,
             key: str = "t", min_rows: int = 50) -> dict[str, Any]:
    """Score predictions of `target` `horizon` rows ahead: pred[i] is the forecast, made at row
    i, of target[i + horizon]. Skill = 1 - SSE / SSE(naive), naive = "stays at target[i]";
    above 0 beats doing nothing. Direction accuracy counts rows where the naive change is not 0."""
    t = _ns(rows, key)
    y = _num(rows, target)
    a = np.asarray(pred, dtype=float)
    n = len(y) - horizon
    if n < 2:
        return {"problem": "not enough rows for the horizon"}
    yt, y0, ai, ti = y[horizon:], y[:n], a[:n], t[:n]
    ok = np.isfinite(yt) & np.isfinite(y0) & np.isfinite(ai)
    se, se0 = (yt - ai) ** 2, (yt - y0) ** 2
    day = _days(ti)
    segments, diagnostics = {}, {}
    for name, m in {"in_sample": ti < holdout_ns, "holdout": ti >= holdout_ns}.items():
        k = m & ok
        if k.sum() < min_rows:
            if m.any():
                segments[name] = {"score": None, "rows": int(k.sum()), "note": f"only {int(k.sum())} scorable rows"}
            continue
        sse, sse0 = float(se[k].sum()), float(se0[k].sum())
        ch = np.sign(yt[k] - y0[k])
        nz = ch != 0
        dir_acc = float((np.sign(ai[k] - y0[k])[nz] == ch[nz]).mean()) if nz.any() else None
        skill = (1.0 - sse / sse0) if sse0 > 0 else None
        segments[name] = {"score": None if skill is None else round(skill, 6), "rows": int(k.sum()),
                          "rmse": round(float(np.sqrt(se[k].mean())), 6),
                          "rmse_naive": round(float(np.sqrt(se0[k].mean())), 6),
                          "direction_accuracy": None if dir_acc is None else round(dir_acc, 4)}
        diagnostics[name] = {k2: segments[name][k2] for k2 in ("rmse", "rmse_naive", "direction_accuracy")}
    scale = float(se0[ok].mean()) or 1.0
    days, first = np.unique(day, return_index=True)
    bounds = np.r_[first, n]
    gain = np.where(ok, se0 - se, 0.0) / scale
    c = np.r_[0.0, np.cumsum(gain)]
    curve = [[_day_label(d), round(float(c[b] - c[a0]), 6)] for d, a0, b in zip(days, bounds[:-1], bounds[1:])]
    ins = segments.get("in_sample") or {}
    notes = (f"In-sample skill {ins.get('score')} (RMSE {ins.get('rmse')} vs {ins.get('rmse_naive')} for 'no change'), "
             f"direction right {ins.get('direction_accuracy')} of the time.") if ins else ""
    return {"segments": segments, "diagnostics": diagnostics, "curve": curve, "curve_kind": "additive",
            "notes": notes}
