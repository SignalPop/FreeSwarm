"""A stored forecast feature, drawn: what went in and what came out, at a few anchors.

For the objective's Forecasts tab. Per anchor: every input series the forecast read (the target
and each covariate) over its context window -- raw values plus the window's mean and std, so the
console can z-score them onto one axis and still show raw values on hover -- and the forecast:
the stored value at the horizon (exactly what the feature file holds), and, when the feature's
model is loaded, the full median path and 10-90% band re-run from the same inputs. Overlaid, the
actual target values over the horizon.

In-sample only: the data is read with rows at or after the objective's split removed, and only
anchors made before the split are drawn, so no observed value at or after the split date is ever
sent. An anchor near the split shows its actual values truncated at it. The forecast path is the
model's answer from in-sample rows and may run past the split. Payload: at most MAX_ANCHORS
anchors, arrays decimated to <= 300 points (min/max kept), shrunk until MAX_BYTES.
"""

from __future__ import annotations

import collections
import json
import logging
from typing import Any

import numpy as np

from .forecast_values import MAX_POINTS, _num, _nums, decimate_lines

logger = logging.getLogger("freetoken.forecast_view")

MAX_ANCHORS = 12
MAX_BYTES = 300_000
# One categorical colour each, never cycled: the target(s) and covariates past 8 are not drawn.
MAX_SERIES = 8
_cache: collections.OrderedDict[tuple, dict] = collections.OrderedDict()


def _secs(t: Any) -> np.ndarray:
    return np.asarray(t, dtype="datetime64[s]").astype(np.int64)


def _cut(split: str | None) -> int | None:
    if not split:
        return None
    return int(np.datetime64(str(split).replace("Z", ""), "s").astype(np.int64))


def pick_anchors(fc_t: Any, frame_t: Any, context: int, n: int, split: str | None = None) -> list[tuple[int, int]]:
    """(row in the feature file, row in the data frame) for up to n anchors, evenly spread over
    the stored forecasts made before the split whose context is fully in the frame. The last
    one is the latest in-sample forecast (its actuals may be cut short by the split)."""
    fs, ts = _secs(fc_t), _secs(frame_t)
    if not len(fs) or not len(ts):
        return []
    pos = np.searchsorted(ts, fs)
    safe = np.minimum(pos, len(ts) - 1)
    ok = (pos < len(ts)) & (ts[safe] == fs) & (pos >= context - 1)
    cut = _cut(split)
    if cut is not None:
        ok &= fs < cut
    rows = np.nonzero(ok)[0]
    if not len(rows):
        return []
    sel = np.unique(np.linspace(0, len(rows) - 1, min(n, len(rows))).round().astype(np.int64))
    return [(int(rows[k]), int(pos[rows[k]])) for k in sel]


def _ahead(ts: np.ndarray, i: int, h: int, step: int) -> np.ndarray:
    n = len(ts)
    return np.array([ts[i + k] if i + k < n else ts[n - 1] + (i + k - (n - 1)) * step for k in range(1, h + 1)],
                    dtype=np.int64)


def build_view(params: dict, fc: dict[str, np.ndarray], frame_t: Any, cols: dict[str, np.ndarray],
               picks: list[tuple[int, int]], split: str | None, paths: dict | None = None,
               max_bytes: int = MAX_BYTES) -> dict:
    """The JSON-ready view. `fc` holds the feature file's columns (t, [pre]last, [pre]fc_median,
    [pre]fc_q10, [pre]fc_q90); `cols` the input series on the frame's rows (cut at the split);
    `paths` {frame row: {target: {"median", "q10", "q90"}}} from a re-run, if any."""
    ts = _secs(frame_t)
    cut = _cut(split)
    if cut is not None:   # belt and braces: the frame was read cut at the split already
        keep = ts < cut
        ts = ts[keep]
        cols = {k: np.asarray(v, dtype=np.float64)[keep] for k, v in cols.items()}
    n = len(ts)
    targets = [c for c in params.get("targets") or [] if c in cols]
    covs = [c for c in params.get("covariates") or [] if c in cols and c not in targets]
    series = (targets + covs)[:MAX_SERIES]
    h = int(params.get("horizon") or 1)
    context = int(params.get("context") or 512)
    diffs = np.diff(ts[-200:]) if n > 2 else np.array([60])
    step = int(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 60
    single = len(params.get("targets") or []) == 1

    def pre(c: str) -> str:
        from .objectives import _slug

        return "" if single else f"{_slug(c, 24)}_"

    def one(pts: int) -> dict:
        anchors, ep_n, ep_in, path_n, path_in = [], 0, 0, 0, 0
        for r, i in picks:
            if i >= n:
                continue
            lo = max(0, i - context + 1)
            win = {c: np.asarray(cols[c][lo:i + 1], dtype=np.float64) for c in series}
            idx = decimate_lines(win, pts)
            inputs = []
            for c in series:
                v = win[c]
                fin = v[np.isfinite(v)]
                sd = float(fin.std()) if len(fin) > 1 else 0.0
                inputs.append({"name": c, "role": "target" if c in targets else "covariate",
                               "mean": _num(fin.mean()) if len(fin) else 0.0, "std": _num(sd) if sd > 0 else 1.0,
                               "v": _nums(v[idx])})
            ahead = _ahead(ts, i, h, step)
            outs = []
            for c in targets:
                p = pre(c)
                a_hi = min(n, i + 1 + h)
                act_t = ts[i + 1:a_hi]
                act_v = np.asarray(cols[c][i + 1:a_hi], dtype=np.float64)
                if cut is not None:
                    m = act_t < cut
                    act_t, act_v = act_t[m], act_v[m]
                end = {"t": int(ahead[-1]), "median": _num(fc.get(f"{p}fc_median", [None] * (r + 1))[r]),
                       "q10": _num(fc.get(f"{p}fc_q10", [None] * (r + 1))[r]),
                       "q90": _num(fc.get(f"{p}fc_q90", [None] * (r + 1))[r])}
                actual_h = float(act_v[-1]) if len(act_v) == h else None
                if actual_h is not None and end["q10"] is not None and end["q90"] is not None:
                    ep_n += 1
                    end["inside"] = bool(end["q10"] <= actual_h <= end["q90"])
                    ep_in += end["inside"]
                row: dict[str, Any] = {"target": c, "last": _num(cols[c][i]), "end": end,
                                       "actual": {"t": [int(x) for x in act_t], "v": _nums(act_v)}}
                got = (paths or {}).get(i, {}).get(c)
                if got:
                    med = np.asarray(got["median"], dtype=np.float64)[:h]
                    q10 = np.asarray(got.get("q10") or med, dtype=np.float64)[:h]
                    q90 = np.asarray(got.get("q90") or med, dtype=np.float64)[:h]
                    k = min(len(med), len(q10), len(q90), h)
                    row["path"] = {"t": [int(x) for x in ahead[:k]], "median": _nums(med[:k]),
                                   "q10": _nums(q10[:k]), "q90": _nums(q90[:k])}
                    m = min(k, len(act_v))
                    if m:
                        a = act_v[:m]
                        path_n += int(np.isfinite(a).sum())
                        path_in += int(((a >= q10[:m]) & (a <= q90[:m])).sum())
                outs.append(row)
            anchors.append({"as_of": str(np.datetime64(int(ts[i]), "s")), "as_of_t": int(ts[i]),
                            "t": [int(x) for x in ts[lo:i + 1][idx]], "sent": int(i - lo + 1),
                            "inputs": inputs, "forecasts": outs,
                            "near_split": bool(cut is not None and len(ahead) and ahead[-1] >= cut)})
        return {"horizon": h, "context": context, "step_seconds": step, "split": split, "max_points": pts,
                "targets": targets, "covariates": covs, "calendar": bool(params.get("calendar")),
                "dropped_series": max(0, len(targets + covs) - MAX_SERIES),
                "anchors": anchors,
                "coverage": {"endpoint": {"n": ep_n, "inside": ep_in, "share": round(ep_in / ep_n, 3) if ep_n else None},
                             "path": {"n": path_n, "inside": path_in,
                                      "share": round(path_in / path_n, 3) if path_n else None}}}

    for pts in (MAX_POINTS, 150, 75, 40, 20):
        doc = one(pts)
        size = len(json.dumps(doc, separators=(",", ":")))
        if size <= max_bytes:
            doc["bytes"] = size
            return doc
    while picks:   # still too big: fewer anchors
        picks = picks[:-1]
        doc = one(20)
        size = len(json.dumps(doc, separators=(",", ":")))
        if size <= max_bytes:
            doc["bytes"] = size
            return doc
    return {"anchors": [], "bytes": 0, "error": "too large to draw"}


# =======================================================================================
# I/O: read the feature, the data, and (optionally) re-run the forecaster
# =======================================================================================
def _read_feature(obj: dict, view: str) -> tuple[dict, dict[str, np.ndarray]]:
    import pyarrow.parquet as pq
    from fastapi import HTTPException

    from .objectives import _features_root, list_features

    meta = next((f for f in list_features(obj["id"]) if f.get("view") == view), None)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"no forecast feature {view!r}")
    table = pq.read_table(_features_root(obj["id"]) / meta["file"])
    fc = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}
    return meta, fc


async def _rerun(params: dict, ts: np.ndarray, cols: dict[str, np.ndarray], rows: list[int]) -> tuple[dict, str | None]:
    """{frame row: {target: {median, q10, q90}}} from the feature's own model, same inputs."""
    from fastapi import HTTPException

    from .objectives import FEATURE_QUANTILES, _calendar, _forecaster

    model = params.get("model")
    if str((params.get("series") or [""])[0]).startswith("candles@"):
        return {}, "candle models (Kronos) are drawn from their stored values only"
    try:
        mgr, info = _forecaster(model)
    except HTTPException:
        return {}, f"{model} is not loaded: showing the stored forecast at the horizon only (load it for full paths)"
    targets = [c for c in params.get("targets") or [] if c in cols]
    covs = [c for c in params.get("covariates") or [] if c in cols]
    h, context = int(params["horizon"]), int(params["context"])
    rows = [i for i in rows if i - context + 1 >= 0]
    out: dict[int, dict] = {i: {} for i in rows}
    try:
        if covs or params.get("calendar"):
            items = []
            for a in rows:
                s0 = a - context + 1
                tgt = [cols[c][s0:a + 1].tolist() for c in targets]
                it: dict[str, Any] = {"target": tgt if len(tgt) > 1 else tgt[0]}
                past = {c: cols[c][s0:a + 1].tolist() for c in covs}
                if params.get("calendar"):
                    ph, pf = _calendar(np.asarray(ts[s0:a + 1]).astype("datetime64[s]"), h)
                    past.update({k: v.tolist() for k, v in ph.items()})
                    it["future_covariates"] = {k: v.tolist() for k, v in pf.items()}
                if past:
                    it["past_covariates"] = past
                items.append(it)
            res = await mgr.forecast(info["model"], {"inputs": items, "horizon": h, "quantiles": FEATURE_QUANTILES})
            for a, f in zip(rows, res.get("forecasts") or []):
                for c, v in zip(targets, f.get("variates") or [f]):
                    q = v.get("quantiles") or {}
                    out[a][c] = {"median": v["median"], "q10": q.get("0.1"), "q90": q.get("0.9")}
        else:
            order = [(a, c) for c in targets for a in rows]
            res = await mgr.forecast(info["model"], {
                "series": [cols[c][a - context + 1:a + 1].tolist() for a, c in order],
                "horizon": h, "quantiles": FEATURE_QUANTILES})
            for (a, c), f in zip(order, res.get("forecasts") or []):
                q = f.get("quantiles") or {}
                out[a][c] = {"median": f["median"], "q10": q.get("0.1"), "q90": q.get("0.9")}
    except Exception as exc:  # noqa: BLE001 -- the stored values still draw
        logger.warning("forecast view: re-run failed", exc_info=True)
        return {}, f"re-running the forecaster failed ({exc}): showing the stored forecast at the horizon only"
    return out, None


async def feature_view(obj: dict, view: str, n_anchors: int = 5, rerun: bool = True) -> dict:
    import asyncio

    from fastapi import HTTPException

    from . import projects
    from .tslab import load_frame

    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    split = obj.get("split_date")
    meta, fc = await asyncio.to_thread(_read_feature, obj, view)
    p = dict(meta.get("params") or {})
    series = list(p.get("series") or [])
    kronos = bool(series) and str(series[0]).startswith("candles@")
    targets = ["close"] if kronos else series
    params = {**p, "targets": targets, "covariates": list(p.get("covariates") or [])}
    n_anchors = max(1, min(MAX_ANCHORS, int(n_anchors)))
    key = (obj["id"], view, meta.get("created_at"), split, n_anchors, rerun)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    dataset = p.get("dataset") or obj.get("dataset")
    t, cols = await asyncio.to_thread(load_frame, project, obj, dataset, [*targets, *params["covariates"]],
                                      p.get("bar"), split)
    if "t" not in fc:
        raise HTTPException(status_code=400, detail=f"{view} has no t column")
    picks = pick_anchors(fc["t"], t, int(p.get("context") or 512), n_anchors, split)
    note = None
    paths: dict = {}
    if rerun and picks:
        paths, note = await _rerun(params, t, cols, [i for _, i in picks])
    doc = await asyncio.to_thread(build_view, params, fc, t, cols, picks, split, paths)
    doc.update(view=view, model=p.get("model"), dataset=dataset, bar=p.get("bar"),
               path_source="re-run of the feature's model on the same inputs" if paths else "stored value at the horizon",
               note=note if note else (None if picks else "no stored forecast before the split to draw"))
    if paths or not rerun or not picks:   # a fallback (model not loaded) is retried next time
        _cache[key] = doc
        while len(_cache) > 32:
            _cache.popitem(last=False)
    return doc
