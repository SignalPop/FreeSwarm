"""The numbers behind a forecast request, for the agent inspector.

``objectives.input_streams`` describes WHICH streams a request sends and over which dates; this
records WHAT was sent and what came back, at the request's few sample anchors: per input stream
the context window the model read (decimated), and for each target the forecast the model
returned -- median path and 10-90% band over the horizon -- plus, where the build also ran
without its inputs (the Chronos-2 lift), that forecast too, and the values that actually
followed when they are in-sample.

Rules:
- No observed value at or after the objective's split date is ever recorded -- not context,
  not realized values, not known-ahead inputs. A sample anchor whose as-of sits in the holdout is
  replaced by the last in-sample anchor (and says so), so every recorded forecast was made from
  in-sample rows only; its horizon may run past the split (it is the model's answer, not data).
- Each array is decimated to at most MAX_POINTS (first and last kept, min/max per bucket), and
  the whole payload is shrunk until it fits MAX_BYTES.
- Capturing must never fail or slow down a forecast: every public method swallows its errors.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("freetoken.forecast_values")

MAX_POINTS = 300
MAX_BYTES = 200_000


# =======================================================================================
# Decimation
# =======================================================================================
def decimate_idx(v: Any, max_points: int = MAX_POINTS) -> np.ndarray:
    """Indices to keep so a line of len(v) points draws the same at <= max_points: the first
    and last point, and each bucket's min and max (so spikes survive)."""
    a = np.asarray(v, dtype=np.float64)
    n = len(a)
    max_points = max(2, int(max_points))
    if n <= max_points:
        return np.arange(n)
    buckets = max(1, (max_points - 2) // 2)
    edges = np.linspace(1, n - 1, buckets + 1).astype(np.int64)
    keep = [0, n - 1]
    lo_fill = np.where(np.isnan(a), np.inf, a)
    hi_fill = np.where(np.isnan(a), -np.inf, a)
    for b in range(buckets):
        s, e = int(edges[b]), int(edges[b + 1])
        if e <= s:
            continue
        keep.append(s + int(np.argmin(lo_fill[s:e])))
        keep.append(s + int(np.argmax(hi_fill[s:e])))
    return np.unique(np.asarray(keep, dtype=np.int64))


def decimate_lines(lines: dict[str, Any], max_points: int = MAX_POINTS) -> np.ndarray:
    """Shared indices for several lines on one time axis: each line gets an equal share of the
    budget and the union is kept (<= max_points in total)."""
    if not lines:
        return np.arange(0)
    n = len(next(iter(lines.values())))
    if n <= max_points:
        return np.arange(n)
    share = max(2, max_points // len(lines))
    idx = np.unique(np.concatenate([decimate_idx(v, share) for v in lines.values()]))
    if len(idx) > max_points:   # tiny budgets: fall back to an even stride that keeps the ends
        idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(np.int64))
    return idx


def _num(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return float(f"{f:.6g}")


def _nums(v: Any) -> list[float | None]:
    return [_num(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]


def _epochs(times: Any) -> np.ndarray:
    return np.asarray(times, dtype="datetime64[s]").astype(np.int64)


def _iso(sec: int) -> str:
    return str(np.datetime64(int(sec), "s"))


def _split_epoch(split: str | None) -> int | None:
    if not split:
        return None
    try:
        return int(np.datetime64(str(split).replace("Z", ""), "s").astype(np.int64))
    except (TypeError, ValueError):
        logger.warning("forecast values: unreadable split %r -- recording nothing past the first row", split)
        return -(2**62)


# =======================================================================================
# The capture
# =======================================================================================
class ValueCapture:
    """Collects the values of one request's sample anchors while the builder runs.

    `detail` is the dict ``input_streams`` returned (and that was handed to note_inputs);
    `times` the row timestamps the anchors index; `inclusive` as in input_streams (feature
    builders read rows up to AND including the anchor, the Forecast Lab the rows before it).
    """

    def __init__(self, detail: dict, times: Any, anchors: list[int], *, context: int, horizon: int,
                 inclusive: bool = True, split: str | None = None) -> None:
        self.ok = False
        self.detail = detail
        self.horizon = int(horizon)
        self.context = int(context)
        self.inclusive = inclusive
        self.split = split
        self.samples: list[dict] = []
        self._by_anchor: dict[int, dict] = {}
        try:
            self.t = _epochs(times)
            n = len(self.t)
            if not anchors or n == 0:
                return
            diffs = np.diff(self.t[-200:]) if n > 2 else np.array([60])
            pos = diffs[diffs > 0]
            self.step = int(np.median(pos)) if len(pos) else 60
            self.cut = _split_epoch(split)
            picks = sorted({0, len(anchors) // 2, len(anchors) - 1})
            for k, pick in enumerate(picks):
                a, note = anchors[pick], None
                if self.cut is not None and self.t[self._hi(a)] >= self.cut:
                    sub = self._last_in_sample(anchors)
                    if sub is None:
                        continue
                    note = (f"the sample at {_iso(self.t[self._hi(a)])} is at/after the split "
                            f"(holdout hidden): shown at the last in-sample forecast instead")
                    a = sub
                if a in self._by_anchor:
                    continue
                s = {"sample": k, "anchor": a, "note": note, "streams": {}, "forecasts": {}, "realized": {}}
                self._by_anchor[a] = s
                self.samples.append(s)
            self.ok = bool(self.samples)
        except Exception:  # noqa: BLE001 -- observing must never fail the forecast
            logger.warning("forecast values: capture setup failed", exc_info=True)
            self.ok = False

    # ----------------------------------------------------------------- index helpers
    def _hi(self, a: int) -> int:
        return a if self.inclusive else a - 1

    def _last_in_sample(self, anchors: list[int]) -> int | None:
        """The latest anchor whose as-of (preferably its whole horizon) is before the split."""
        n = len(self.t)
        fallback = None
        for a in reversed(anchors):
            hi = self._hi(a)
            if hi < 0 or self.t[hi] >= self.cut:
                continue
            if fallback is None:
                fallback = a
            end = hi + self.horizon
            if end < n and self.t[end] < self.cut:
                return a
        return fallback

    def _keep(self, secs: np.ndarray) -> np.ndarray:
        return np.ones(len(secs), dtype=bool) if self.cut is None else secs < self.cut

    def _ahead_times(self, a: int) -> np.ndarray:
        """Timestamps of the horizon steps: the real rows where the data has them, the grid's
        typical spacing beyond."""
        hi = self._hi(a)
        n = len(self.t)
        out = np.empty(self.horizon, dtype=np.int64)
        for k in range(1, self.horizon + 1):
            r = hi + k
            out[k - 1] = self.t[r] if r < n else self.t[n - 1] + (r - (n - 1)) * self.step
        return out

    # ----------------------------------------------------------------- recording
    def wants(self, a: int) -> bool:
        return self.ok and a in self._by_anchor

    def context_values(self, name: str, role: str, lines: dict[str, Any]) -> None:
        """An input stream: {label: full series aligned with `times`}. The context window of
        each sample anchor is recorded (rows before the split only)."""
        if not self.ok:
            return
        try:
            for s in self.samples:
                hi = self._hi(s["anchor"])
                lo = max(0, hi - self.context + 1)
                secs = self.t[lo:hi + 1]
                keep = self._keep(secs)
                vals = {lab: np.asarray(v[lo:hi + 1], dtype=np.float64)[keep] for lab, v in lines.items()}
                secs = secs[keep]
                idx = decimate_lines(vals, MAX_POINTS)
                s["streams"][(name, role)] = {"t": secs[idx], "lines": {lab: v[idx] for lab, v in vals.items()},
                                              "sent": int(hi - lo + 1)}
        except Exception:  # noqa: BLE001
            logger.warning("forecast values: context capture of %s failed", name, exc_info=True)

    def ahead_values(self, name: str, role: str, fn: Callable[[int], dict[str, Any]]) -> None:
        """A known-ahead input (future covariates): fn(anchor) -> {label: values over the horizon}."""
        if not self.ok:
            return
        try:
            for s in self.samples:
                secs = self._ahead_times(s["anchor"])
                raw = {lab: np.asarray(v, dtype=np.float64) for lab, v in fn(s["anchor"]).items()}
                m = min([len(secs)] + [len(v) for v in raw.values()])
                secs = secs[:m]
                keep = self._keep(secs)
                s["streams"][(name, role)] = {"t": secs[keep], "lines": {lab: v[:m][keep] for lab, v in raw.items()},
                                              "sent": self.horizon}
        except Exception:  # noqa: BLE001
            logger.warning("forecast values: known-ahead capture of %s failed", name, exc_info=True)

    def output(self, a: int, target: str, fc: dict, with_inputs: bool = True) -> None:
        """The model's answer for anchor `a` (a forecast dict: median + quantiles {"0.1", "0.9"})."""
        if not self.wants(a):
            return
        try:
            s = self._by_anchor[a]
            med = np.asarray(fc.get("median") or [], dtype=np.float64)
            q = fc.get("quantiles") or {}
            q10 = np.asarray(q.get("0.1") or fc.get("q10") or med, dtype=np.float64)
            q90 = np.asarray(q.get("0.9") or fc.get("q90") or med, dtype=np.float64)
            secs = self._ahead_times(a)
            m = min(len(secs), len(med), len(q10), len(q90))
            # Not clipped at the split: sample anchors are in-sample (as-of before the split), so
            # the model read no holdout row -- its answer reveals nothing of the holdout even
            # where its horizon runs past the split date.
            entry = s["forecasts"].setdefault(target, {})
            entry["with" if with_inputs else "without"] = {
                "t": secs[:m], "median": med[:m], "q10": q10[:m], "q90": q90[:m]}
        except Exception:  # noqa: BLE001
            logger.warning("forecast values: output capture failed", exc_info=True)

    def realized(self, target: str, values: Any) -> None:
        """What actually followed each sample anchor, where it is in the data AND before the split."""
        if not self.ok:
            return
        try:
            n = len(self.t)
            for s in self.samples:
                hi = self._hi(s["anchor"])
                lo_r, hi_r = hi + 1, min(n, hi + 1 + self.horizon)
                if lo_r >= hi_r:
                    continue
                secs = self.t[lo_r:hi_r]
                keep = self._keep(secs)
                v = np.asarray(values[lo_r:hi_r], dtype=np.float64)[keep]
                if len(v):
                    s["realized"][target] = {"t": secs[keep], "v": v}
        except Exception:  # noqa: BLE001
            logger.warning("forecast values: realized capture failed", exc_info=True)

    # ----------------------------------------------------------------- the payload
    def payload(self, max_bytes: int = MAX_BYTES) -> dict | None:
        """The JSON-ready record, shrunk until it fits `max_bytes` (None if nothing fits)."""
        if not self.ok:
            return None
        try:
            index = {(st.get("name"), st.get("role")): i for i, st in enumerate(self.detail.get("streams") or [])}
            plans = [(pts, True, None) for pts in (MAX_POINTS, 150, 75, 40, 20)] + [(20, False, None), (20, False, 1)]
            for pts, all_streams, anchors_cap in plans:
                doc = self._build(index, pts, all_streams, anchors_cap)
                size = len(json.dumps(doc, separators=(",", ":")))
                if size <= max_bytes:
                    doc["bytes"] = size
                    return doc
            logger.warning("forecast values: payload does not fit %d bytes even when shrunk", max_bytes)
            return None
        except Exception:  # noqa: BLE001
            logger.warning("forecast values: payload build failed", exc_info=True)
            return None

    def _build(self, index: dict, pts: int, all_streams: bool, anchors_cap: int | None) -> dict:
        def thin(t: np.ndarray, lines: dict[str, np.ndarray]) -> tuple[list, dict[str, list]]:
            idx = decimate_lines(lines, pts) if lines else np.arange(len(t))
            return [int(x) for x in t[idx]], {k: _nums(v[idx]) for k, v in lines.items()}

        out = []
        for s in self.samples[:anchors_cap]:
            hi = self._hi(s["anchor"])
            streams = []
            for (name, role), st in s["streams"].items():
                if not all_streams and role != "target":
                    continue
                t, lines = thin(st["t"], st["lines"])
                streams.append({"stream": index.get((name, role)), "name": name, "role": role, "t": t,
                                "lines": [{"label": k, "v": v} for k, v in lines.items()], "sent": st["sent"]})
            fcs = []
            for target, got in s["forecasts"].items():
                row: dict[str, Any] = {"target": target}
                for key in ("with", "without"):
                    f = got.get(key)
                    if f is None:
                        continue
                    t, lines = thin(f["t"], {"median": f["median"], "q10": f["q10"], "q90": f["q90"]})
                    row["with_inputs" if key == "with" else "without_inputs"] = {"t": t, **lines}
                if "with_inputs" in row:
                    fcs.append(row)
            realized = []
            for target, r in s["realized"].items():
                t, lines = thin(r["t"], {"v": r["v"]})
                realized.append({"target": target, "t": t, "v": lines["v"]})
            out.append({"sample": s["sample"], "as_of": _iso(self.t[hi]), "as_of_t": int(self.t[hi]),
                        "note": s["note"], "streams": streams, "forecasts": fcs, "realized": realized})
        return {"version": 1, "horizon": self.horizon, "context": self.context, "step_seconds": self.step,
                "split": self.split, "max_points": pts, "anchors": out}

    def publish(self, model: str) -> None:
        """Hand the finished record to the agent inspector (never raises)."""
        if not self.ok:
            return
        try:
            doc = self.payload()
            if doc is None:
                return
            from .agent_activity import note_values

            note_values(model, self.detail, doc)
        except Exception:  # noqa: BLE001
            logger.warning("forecast values: publish failed", exc_info=True)


class _Off(ValueCapture):
    """A capture that records nothing (when even setting one up failed)."""

    def __init__(self) -> None:  # noqa: D401 -- deliberately skips the parent's setup
        self.ok = False
        self.samples = []
        self._by_anchor = {}


def start(detail: dict, times: Any, anchors: list[int], **kw: Any) -> ValueCapture:
    """A ValueCapture, or an inert one if anything goes wrong. Never raises."""
    try:
        return ValueCapture(detail, times, anchors, **kw)
    except Exception:  # noqa: BLE001
        logger.warning("forecast values: could not start a capture", exc_info=True)
        return _Off()
