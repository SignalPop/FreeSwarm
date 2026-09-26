"""The Forecasts tab's drawing of a stored feature (app/forecast_view.py): inputs over the
context, the forecast cone, and what followed -- never a value at or after the split, and a
bounded payload."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app import forecast_view as V
from app import objectives as O

SPLIT = "2024-01-03"
H = 6
CTX = 128


def _frame(n=4000, covs=("GEX",)):
    t = np.datetime64("2024-01-01T00:00:00", "us") + np.arange(n) * np.timedelta64(60, "s")
    rng = np.random.default_rng(3)
    cols = {"Close": np.cumsum(rng.normal(size=n)) + 100}
    for k, c in enumerate(covs):
        cols[c] = np.sin(np.arange(n) / (k + 5)) * (k + 1) * 1000
    return t, cols


def _stored(t, cols, every=20):
    anchors = list(range(CTX - 1, len(t), every))
    close = cols["Close"]
    return {"t": t[anchors], "last": close[anchors], "fc_median": close[anchors] + 0.1,
            "fc_q10": close[anchors] - 2, "fc_q90": close[anchors] + 2}


def _all_times(doc):
    for a in doc["anchors"]:
        yield from a["t"]
        for f in a["forecasts"]:
            yield from f["actual"]["t"]


PARAMS = {"targets": ["Close"], "covariates": ["GEX"], "horizon": H, "context": CTX, "calendar": False}


def test_anchors_are_in_sample_and_nothing_observed_reaches_the_split():
    t, cols = _frame()
    fc = _stored(t, cols, every=11)   # the last in-sample anchor is 3 bars before the split
    picks = V.pick_anchors(fc["t"], t, CTX, 5, SPLIT)          # the frame here is NOT cut: pick_anchors cuts
    cut = int(np.datetime64(SPLIT, "s").astype(np.int64))
    assert len(picks) == 5
    doc = V.build_view(PARAMS, fc, t, cols, picks, SPLIT)
    assert len(doc["anchors"]) == 5 and doc["bytes"] <= V.MAX_BYTES
    assert max(_all_times(doc)) < cut
    assert all(a["as_of_t"] < cut for a in doc["anchors"])
    last = doc["anchors"][-1]
    assert last["near_split"] and len(last["forecasts"][0]["actual"]["t"]) < H     # truncated at the split
    first = doc["anchors"][0]
    ins = {i["name"]: i for i in first["inputs"]}
    assert set(ins) == {"Close", "GEX"} and ins["GEX"]["std"] > 100      # raw values + stats to z-score
    assert len(ins["Close"]["v"]) == len(first["t"]) <= 300 and first["sent"] == CTX
    f0 = first["forecasts"][0]
    assert f0["end"]["median"] == pytest.approx(cols["Close"][CTX - 1] + 0.1, rel=1e-5)
    assert f0["end"]["inside"] is True and doc["coverage"]["endpoint"]["n"] >= 4


def test_rerun_paths_are_drawn_and_scored():
    t, cols = _frame()
    fc = _stored(t, cols)
    picks = V.pick_anchors(fc["t"], t, CTX, 3, SPLIT)
    paths = {i: {"Close": {"median": [cols["Close"][i]] * H, "q10": [cols["Close"][i] - 50] * H,
                           "q90": [cols["Close"][i] + 50] * H}} for _, i in picks}
    doc = V.build_view(PARAMS, fc, t, cols, picks, SPLIT, paths)
    p = doc["anchors"][0]["forecasts"][0]["path"]
    assert len(p["median"]) == H and p["t"][0] > doc["anchors"][0]["as_of_t"]
    assert doc["coverage"]["path"]["share"] == 1.0


def test_payload_is_bounded_with_many_long_inputs():
    covs = tuple(f"c{k}" for k in range(30))
    t, cols = _frame(n=60_000, covs=covs)
    fc = _stored(t, cols, every=500)
    params = {**PARAMS, "covariates": list(covs), "context": 8192}
    picks = V.pick_anchors(fc["t"], t, 8192, 12, None)
    doc = V.build_view(params, fc, t, cols, picks, None)
    assert doc["bytes"] <= V.MAX_BYTES and len(json.dumps(doc, separators=(",", ":"))) <= V.MAX_BYTES + 20
    assert doc["dropped_series"] == 31 - V.MAX_SERIES == 23


def test_feature_view_end_to_end_reads_only_in_sample_rows(monkeypatch, tmp_path):
    from app import tslab

    t, cols = _frame()
    fc = _stored(t, cols)
    root = tmp_path / "features"
    root.mkdir()
    pq.write_table(pa.table({"t": pa.array(fc["t"], type=pa.timestamp("us")),
                             **{k: v for k, v in fc.items() if k != "t"}}), root / "close_x.parquet")
    meta = {"view": "fc_close_x", "file": "close_x.parquet", "created_at": 1.0,
            "params": {"dataset": "bars", "series": ["Close"], "covariates": ["GEX"], "calendar": False,
                       "horizon": H, "every": 20, "context": CTX, "model": "amazon/chronos-2", "bar": None}}
    (root / "close_x.json").write_text(json.dumps(meta), encoding="utf-8")
    seen = {}

    def fake_frame(project, obj, dataset, columns, bar, end):
        seen["end"] = end
        keep = t < np.datetime64(end)
        return t[keep], {c: cols[c][keep] for c in columns}

    class Mgr:
        async def forecast(self, model, payload):
            return {"forecasts": [{"median": [it["target"][-1]] * H,
                                   "quantiles": {"0.1": [it["target"][-1] - 1] * H, "0.9": [it["target"][-1] + 1] * H}}
                                  for it in payload["inputs"]]}

    monkeypatch.setattr(O, "_features_root", lambda oid: root)
    monkeypatch.setattr(O.projects, "get", lambda pid: {"data_dir": str(tmp_path)})
    monkeypatch.setattr(tslab, "load_frame", fake_frame)
    monkeypatch.setattr(O, "_forecaster", lambda model: (Mgr(), {"model": model}))
    V._cache.clear()
    obj = {"id": "o1", "project_id": "p1", "split_date": SPLIT, "dataset": "bars"}
    doc = asyncio.run(V.feature_view(obj, "fc_close_x", 5))
    assert seen["end"] == SPLIT and doc["path_source"].startswith("re-run") and doc["note"] is None
    cut = int(np.datetime64(SPLIT, "s").astype(np.int64))
    assert len(doc["anchors"]) == 5 and max(_all_times(doc)) < cut
    assert all("path" in a["forecasts"][0] for a in doc["anchors"])
    # model not loaded: the stored horizon values still draw
    from fastapi import HTTPException

    def not_loaded(model):
        raise HTTPException(status_code=400, detail="not loaded")

    monkeypatch.setattr(O, "_forecaster", not_loaded)
    V._cache.clear()
    doc2 = asyncio.run(V.feature_view(obj, "fc_close_x", 3))
    assert "not loaded" in doc2["note"] and all("path" not in a["forecasts"][0] for a in doc2["anchors"])
    assert all(a["forecasts"][0]["end"]["median"] is not None for a in doc2["anchors"])
