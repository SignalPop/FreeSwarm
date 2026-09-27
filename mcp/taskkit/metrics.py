"""Scoring helpers shared by task servers and the FreeSwarm harness.

Everything here is plain numpy on already-aligned arrays -- no I/O, no MCP -- so a task server
can use it to score actions, and the harness can use the same code to describe positions it
marks to market itself. Two families:

* **Return statistics** (`stats`, `smoothness`): the numbers a daily return stream is judged by
  -- Sharpe, Sortino, total return, drawdown, how straight the equity curve is.
* **Swing capture** (`zigzag`, `swings`): the price cut into its high/low legs, and how often a
  position sat on the right side of each leg. Hindsight by design -- it measures what the
  market did -- so it is a yardstick for actions, never a feature a strategy may see.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------------------------
# Return statistics
# ---------------------------------------------------------------------------------------------
def smoothness(rets: list[float] | np.ndarray) -> float | None:
    """How straight the equity curve is: R^2 of log equity against time, signed by its slope.

    1.0 is a steady climb; near 0 is a random walk or a curve that went nowhere for months and
    then jumped; negative is a steady decline. Flat periods count as time, so a strategy that
    earned everything in one burst scores low even if the burst was large."""
    r = np.asarray(rets, dtype=float)
    n = len(r)
    if n < 3:
        return None
    eq = np.cumsum(np.log1p(np.maximum(r, -0.999999)))
    t = np.arange(n, dtype=float)
    tc, ec = t - t.mean(), eq - eq.mean()
    ss_e = float((ec * ec).sum())
    if ss_e < 1e-18:
        return None
    c = float((tc * ec).sum()) / math.sqrt(float((tc * tc).sum()) * ss_e)
    return math.copysign(c * c, c)


def stats(rets: list[float] | np.ndarray, periods_per_year: float = 252.0) -> dict[str, Any]:
    """Summary statistics of one segment of per-period (usually daily) returns.

    Keys: periods, active_periods (non-zero), sharpe, sortino, total_return, cagr, max_drawdown,
    calmar, volatility, win_rate, smoothness. Undefined values (no variance, no drawdown) are
    None. `days`/`active_days` repeat the counts under the names the console already shows."""
    r = np.asarray(rets, dtype=float)
    n = len(r)
    active = int((r != 0).sum())
    out: dict[str, Any] = {"periods": n, "active_periods": active, "days": n, "active_days": active}
    if n < 2:
        return out
    mean = float(r.mean())
    sd = float(r.std(ddof=1))
    down = math.sqrt(float((np.minimum(r, 0.0) ** 2).mean()))
    eq = np.cumprod(1.0 + r)
    mdd = float((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1.0).min())
    mdd = min(0.0, mdd)
    total = float(eq[-1] - 1.0)
    years = n / periods_per_year
    cagr = (float(eq[-1]) ** (1.0 / years) - 1.0) if eq[-1] > 0 and years > 0 else None
    nz = r[r != 0]
    out.update({
        "sharpe": (mean / sd * math.sqrt(periods_per_year)) if sd > 1e-12 else None,
        "sortino": (mean / down * math.sqrt(periods_per_year)) if down > 1e-12 else None,
        "total_return": total,
        "cagr": cagr,
        "max_drawdown": mdd,
        "calmar": (cagr / abs(mdd)) if (cagr is not None and mdd < -1e-9) else None,
        "volatility": sd * math.sqrt(periods_per_year),
        "win_rate": float((nz > 0).mean()) if len(nz) else None,
        "smoothness": smoothness(r),
    })
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------
# Swing capture
# ---------------------------------------------------------------------------------------------
def zigzag(p: np.ndarray, thr: float) -> list[tuple[int, int, int]]:
    """The swing legs of one stretch of prices: (start, end, +1 up / -1 down), pivot to pivot.

    A pivot is confirmed when price reverses at least `thr` (a fraction) from the running
    extreme; each leg runs from one confirmed pivot to the next, so every leg moved at least
    `thr`. Before the first confirmed move the start is the running low (or high); the last leg
    ends at its extreme and whatever drifts after it is left out."""
    n = len(p)
    legs: list[tuple[int, int, int]] = []
    if n < 2:
        return legs
    d, piv, ext, lo, hi = 0, 0, 0, 0, 0
    for i in range(1, n):
        x = p[i]
        if d == 0:
            if x < p[lo]:
                lo = i
            if x > p[hi]:
                hi = i
            if x >= p[lo] * (1 + thr):
                d, piv, ext = 1, lo, i
            elif x <= p[hi] * (1 - thr):
                d, piv, ext = -1, hi, i
        elif d == 1:
            if x > p[ext]:
                ext = i
            elif x <= p[ext] * (1 - thr):
                legs.append((piv, ext, 1))
                d, piv, ext = -1, ext, i
        else:
            if x < p[ext]:
                ext = i
            elif x >= p[ext] * (1 + thr):
                legs.append((piv, ext, -1))
                d, piv, ext = 1, ext, i
    if d:
        legs.append((piv, ext, d))
    return legs


def swings(t: np.ndarray, p: np.ndarray, pos: np.ndarray, thr: float,
           split_t: float | None) -> dict[str, dict[str, Any]]:
    """How well positions sat on the right side of the price's swings, per segment.

    `t` is epoch seconds (UTC), `p` the price, `pos` the position DECIDED at each row -- the one
    held over the following row, which is how it is priced. The prices are cut into swing legs
    day by day (`zigzag`). Over a leg's rows the alignment is the mean of sign(position) x the
    leg's direction, from -1 (always the wrong side) to +1 (always the right side), flat counting
    0. A leg is a HIT when alignment is at least 0.5 -- long through most of an up leg, short
    through most of a down leg -- a MISS at -0.5 or below, and neither when mostly flat or mixed.
    `net` is hits minus misses, `net_per_leg` that over all legs, and `capture` the alignment
    weighted by each leg's move. Riding the drift (always long) nets about zero here, so this
    rewards reading the turns, both ways. Segments: in_sample / holdout split at `split_t`
    (epoch seconds), or one `full` segment without a split."""
    out: dict[str, dict[str, Any]] = {}
    if len(t) < 2:
        return out
    held = np.sign(np.r_[0.0, pos[:-1]])            # the position earning row i was set at i-1
    cs = np.r_[0.0, np.cumsum(held)]
    day = (t // 86400).astype(np.int64)
    cuts = np.flatnonzero(np.diff(day)) + 1
    legs: list[tuple[int, int, int]] = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        legs += [(a + i, a + j, d) for i, j, d in zigzag(p[a:b], thr)]
    if not legs:
        return out
    L = np.array(legs)
    i, j, d = L[:, 0], L[:, 1], L[:, 2]
    align = (cs[j + 1] - cs[i + 1]) / np.maximum(1, j - i) * d   # rows i+1..j earn the leg
    move = np.abs(p[j] / p[i] - 1)
    segs = ({"in_sample": t[i] < split_t, "holdout": t[i] >= split_t} if split_t
            else {"full": np.ones(len(L), bool)})
    for name, k in segs.items():
        if not k.any():
            continue
        hit, miss, up = align[k] >= 0.5, align[k] <= -0.5, d[k] > 0
        n = int(k.sum())
        out[name] = {
            "legs": n, "up_legs": int(up.sum()), "down_legs": int((~up).sum()),
            "up_caught_long": int((hit & up).sum()), "down_caught_short": int((hit & ~up).sum()),
            "up_while_short": int((miss & up).sum()), "down_while_long": int((miss & ~up).sum()),
            "hits": int(hit.sum()), "misses": int(miss.sum()), "net": int(hit.sum() - miss.sum()),
            "net_per_leg": round(float((hit.sum() - miss.sum()) / n), 4),
            "capture": round(float((align[k] * move[k]).sum() / move[k].sum()), 4) if move[k].sum() else None,
            "legs_per_day": round(n / max(1, len(np.unique(day[i[k]]))), 2),
        }
    return out
