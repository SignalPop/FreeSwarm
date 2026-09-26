"""The values behind a forecast request (app/forecast_values.py) for the agent inspector:
decimation bounds, the holdout rule, the payload size bound, and that capturing can never
fail the forecast it watches."""

from __future__ import annotations

import asyncio
import datetime as dt
import json

import numpy as np
import pytest

from app import agent_activity as A
from app import forecast_values as FV
from app import objectives as O

SPLIT = "2024-01-03"


def _grid(n: int, start: str = "2024-01-01T00:00:00", step_s: int = 60):
    return np.datetime64(start, "s") + np.arange(n) * np.timedelta64(step_s, "s")


def _epoch(s: str) -> int:
    return int(np.datetime64(s, "s").astype(np.int64))


def _fc(h: int, level: float = 1.0) -> dict:
    med = [level + 0.01 * k for k in range(h)]
    return {"median": med, "quantiles": {"0.1": [m - 1 for m in med], "0.5": med, "0.9": [m + 1 for m in med]}}


# ---------------------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [5, 300, 301, 1000, 50_000])
def test_decimation_is_bounded_and_keeps_the_ends_and_the_extremes(n):
    rng = np.random.default_rng(n)
    v = np.cumsum(rng.normal(size=n))
    if n > 10:
        v[n // 3] = 1e6        # a spike must survive
        v[2 * n // 3] = -1e6
    idx = FV.decimate_idx(v, 300)
    assert len(idx) <= 300 and idx[0] == 0 and idx[-1] == n - 1
    assert np.all(np.diff(idx) > 0)
    assert v[idx].max() == v.max() and v[idx].min() == v.min()
    lines = FV.decimate_lines({"a": v, "b": -v, "c": v * 2}, 300)
    assert len(lines) <= 300 and lines[0] == 0 and lines[-1] == n - 1


def test_decimation_survives_nans():
    v = np.full(5000, np.nan)
    v[100] = 3.0
    idx = FV.decimate_idx(v, 100)
    assert len(idx) <= 100 and 100 in idx


# ---------------------------------------------------------------------------------------
# The capture: the holdout rule
# ---------------------------------------------------------------------------------------
def _capture_run(n=5000, context=512, horizon=12, split=SPLIT, n_covs=1):
    t = _grid(n)
    anchors = list(range(context - 1, n, 50))
    streams = [("Close", "target")] + [(f"cov{i}", "past covariate") for i in range(n_covs)]
    detail = O.input_streams("feature", "m", "bars", t, anchors, context, horizon, streams)
    cap = FV.start(detail, t, anchors, context=context, horizon=horizon, split=split)
    close = np.cumsum(np.random.default_rng(0).normal(size=n)) + 100
    cap.context_values("Close", "target", {"Close": close})
    for i in range(n_covs):
        cap.context_values(f"cov{i}", "past covariate", {f"cov{i}": np.sin(np.arange(n) / (i + 3))})
    for a in anchors:
        cap.output(a, "Close", _fc(horizon, close[a]))
        cap.output(a, "Close", _fc(horizon, close[a] - 1), with_inputs=False)
    cap.realized("Close", close)
    return detail, cap, anchors, t


def test_no_observed_value_at_or_after_the_split():
    detail, cap, anchors, t = _capture_run()
    doc = cap.payload()
    cut = _epoch(SPLIT)
    assert doc and doc["bytes"] <= FV.MAX_BYTES and len(doc["anchors"]) == 3
    # the last sample anchor is in the holdout (the data runs past the split): replaced, and said so
    assert detail["samples"][-1]["as_of"] >= SPLIT
    assert doc["anchors"][-1]["note"] and "split" in doc["anchors"][-1]["note"]
    for a in doc["anchors"]:
        assert a["as_of_t"] < cut
        for s in a["streams"]:
            assert s["t"] and max(s["t"]) < cut
            assert all(len(line["v"]) == len(s["t"]) <= FV.MAX_POINTS for line in s["lines"])
            assert s["t"][-1] == a["as_of_t"]          # the context ends AT the as-of (inclusive)
        for r in a["realized"]:
            assert max(r["t"]) < cut and min(r["t"]) > a["as_of_t"]
        f = a["forecasts"][0]
        assert f["with_inputs"]["t"][0] > a["as_of_t"] and len(f["with_inputs"]["median"]) == 12
        assert f["without_inputs"]["median"] != f["with_inputs"]["median"]
    # each captured stream maps back to its block on the timeline
    assert all(s["stream"] is not None for a in doc["anchors"] for s in a["streams"])


def test_everything_in_the_holdout_records_nothing():
    t = _grid(2000, start="2024-02-01T00:00:00")
    anchors = list(range(99, 2000, 100))
    detail = O.input_streams("feature", "m", "bars", t, anchors, 100, 6, [("Close", "target")])
    cap = FV.start(detail, t, anchors, context=100, horizon=6, split=SPLIT)
    cap.context_values("Close", "target", {"Close": np.arange(2000.0)})
    assert not cap.ok and cap.payload() is None


def test_payload_fits_the_byte_bound_even_with_many_long_streams():
    detail, cap, _, _ = _capture_run(n=40_000, context=8192, horizon=256, split=None, n_covs=40)
    doc = cap.payload()
    assert doc is not None and doc["bytes"] <= FV.MAX_BYTES
    assert len(json.dumps(doc, separators=(",", ":"))) <= FV.MAX_BYTES + 20
    assert doc["max_points"] < FV.MAX_POINTS          # it had to shrink


# ---------------------------------------------------------------------------------------
# Stored with the forecast row, fetched on demand
# ---------------------------------------------------------------------------------------
def test_values_attach_to_the_request_row_and_are_served_by_id():
    A.reset()
    token = A._caller.set({"rid": 4242, "agent": "qwen", "path": "/api/objectives/o1/features", "recipe": {}})
    try:
        detail, cap, _, _ = _capture_run()
        A.note_inputs("m", detail)
        A.note_forecast("m", {"series": [[1.0] * 10], "horizon": 12}, 0.1, response=object())
        cap.publish("m")
    finally:
        A._caller.reset(token)
    row = A.forecasts("m")[0]
    summary = row["inputs"][0]["values"]
    assert summary["bytes"] <= A.MAX_VALUES_BYTES and len(summary["anchors"]) == 3
    assert "streams" not in json.dumps(summary)            # the poll carries only the summary
    assert A.values(summary["id"])["anchors"][0]["streams"]
    assert "values" not in detail                           # copy-on-write: the builder's dict is untouched
    A.reset()


def test_a_row_without_values_still_reads():
    A.reset()
    A.note_forecast("m", {"series": [[1.0] * 10], "horizon": 3}, 0.1, response=object())
    assert A.forecasts("m")[0]["inputs"] == [] and A.values("nope") is None


# ---------------------------------------------------------------------------------------
# A feature build end to end with a fake forecaster -- and capture failures that must not matter
# ---------------------------------------------------------------------------------------
class FakeMgr:
    def __init__(self):
        self.calls = 0

    async def forecast(self, model, payload):
        self.calls += 1
        A.note_forecast(model, payload, 0.01, response=object())
        h = payload["horizon"]
        return {"forecasts": [_fc(h, s[-1]) for s in payload["series"]]}


def _build(monkeypatch, tmp_path, mgr):
    n = 4000
    start = dt.datetime(2024, 1, 1)
    times = [start + dt.timedelta(minutes=k) for k in range(n)]
    values = list(np.cumsum(np.random.default_rng(1).normal(size=n)) + 50)
    monkeypatch.setattr(O.projects, "get", lambda pid: {"data_dir": str(tmp_path)})
    monkeypatch.setattr(O, "_forecaster", lambda model: (mgr, {"model": "fake/m", "context_length": 256}))
    monkeypatch.setattr(O, "_load_series", lambda *a, **k: (times, values, "t"))
    monkeypatch.setattr(O, "_features_root", lambda oid: tmp_path / "features")
    obj = {"id": "o1", "project_id": "p1", "split_date": SPLIT, "dataset": "bars"}
    req = O.FeatureReq(column="Close", horizon=6, context=256, every=40, name="t_close")
    A.reset()
    token = A._caller.set({"rid": 77, "agent": "qwen", "path": "/api/objectives/o1/features", "recipe": {}})
    try:
        return asyncio.run(O._build_feature(obj, req))
    finally:
        A._caller.reset(token)


def test_feature_build_records_its_values(monkeypatch, tmp_path):
    meta = _build(monkeypatch, tmp_path, FakeMgr())
    assert meta["rows"] > 0
    row = A.forecasts("fake/m")[0]
    doc = A.values(row["inputs"][0]["values"]["id"])
    cut = _epoch(SPLIT)
    assert doc["bytes"] <= FV.MAX_BYTES
    for a in doc["anchors"]:
        assert a["as_of_t"] < cut
        assert max(a["streams"][0]["t"]) < cut and len(a["streams"][0]["t"]) <= FV.MAX_POINTS
        assert a["forecasts"][0]["target"] == "Close" and len(a["forecasts"][0]["with_inputs"]["median"]) == 6
        for r in a["realized"]:
            assert max(r["t"]) < cut
    A.reset()


@pytest.mark.parametrize("where", ["setup", "decimate", "publish", "payload"])
def test_a_capture_failure_never_fails_the_build(monkeypatch, tmp_path, where):
    def boom(*a, **k):
        raise RuntimeError("capture exploded")

    if where == "setup":
        monkeypatch.setattr(FV.ValueCapture, "__init__", boom)
    elif where == "decimate":
        monkeypatch.setattr(FV, "decimate_lines", boom)
    elif where == "publish":
        monkeypatch.setattr(A, "note_values", boom)
    else:
        monkeypatch.setattr(FV, "_nums", boom)                  # the payload cannot be built
    mgr = FakeMgr()
    meta = _build(monkeypatch, tmp_path, mgr)
    assert meta["rows"] > 0 and mgr.calls > 0 and (tmp_path / "features" / meta["file"]).is_file()
    row = A.forecasts("fake/m")[0]
    assert row["calls"] == mgr.calls and row["inputs"]          # the row and its dates are still there
    assert "values" not in row["inputs"][0]
    A.reset()
