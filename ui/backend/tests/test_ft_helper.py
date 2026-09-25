"""The sandbox helper `ft` (ui/sandbox/ft.py): load(prefix=...) and forecast().

ft runs inside the candidate sandbox, where pandas is installed; these tests skip without it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

FT_PATH = Path(__file__).resolve().parents[2] / "sandbox" / "ft.py"


@pytest.fixture
def ft(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ft_under_test", FT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    work = tmp_path / ".ft"
    work.mkdir()
    monkeypatch.setattr(mod, "_FT", str(work))
    monkeypatch.setattr(mod, "_REQUESTS", str(work / "forecast_requests.json"))
    monkeypatch.setattr(mod, "_CATALOG", [])
    mod._root = tmp_path
    return mod


def _add_feature(ft, name: str, frame) -> None:
    frame.to_parquet(ft._root / f"{name}.parquet")
    ft._CATALOG.append({"view": f"fc_{name}", "path": f"{name}.parquet", "format": "parquet", "root": str(ft._root)})


def _fc_frame(times, medians):
    return pd.DataFrame({"t": pd.to_datetime(times), "last": [1.0] * len(times), "fc_median": medians,
                         "fc_q10": medians, "fc_q90": medians})


def test_load_prefix_renames_all_but_t(ft):
    _add_feature(ft, "a", _fc_frame(["2024-01-01 10:00"], [2.0]))
    cols = list(ft.load("fc_a", prefix="a_").columns)
    assert cols == ["t", "a_last", "a_fc_median", "a_fc_q10", "a_fc_q90"]
    assert list(ft.load("fc_a").columns)[1] == "last"  # no prefix: unchanged


def test_recipe_name_is_stable_and_order_independent(ft):
    r1 = {"column": "GEX", "horizon": 6, "covariates": ["A", "B"], "model": "m"}
    r2 = {"model": "m", "covariates": ["A", "B"], "horizon": 6, "column": "GEX"}
    assert ft.recipe_name(r1) == ft.recipe_name(r2)
    assert ft.recipe_name(r1).startswith("auto_")
    assert ft.recipe_name(r1) != ft.recipe_name({**r1, "horizon": 7})
    # Keys outside the recipe (e.g. join options) never change the name.
    assert ft.recipe_name(r1) == ft.recipe_name({**r1, "prefix": "x_"})


def test_forecast_not_built_records_request_once_and_raises(ft):
    for _ in range(2):
        with pytest.raises(ft.ForecastPending):
            ft.forecast("GEX", inputs=["Pressure_Total"], horizon=6, model="amazon/chronos-2")
    asked = json.loads(Path(ft._REQUESTS).read_text(encoding="utf-8"))
    assert len(asked) == 1
    assert asked[0]["recipe"]["covariates"] == ["Pressure_Total"]
    assert asked[0]["name"] == ft.recipe_name(asked[0]["recipe"])


def test_forecast_needs_a_series(ft):
    with pytest.raises(ValueError):
        ft.forecast()


def test_forecast_join_is_backward_and_prefixed(ft):
    recipe_kwargs = {"horizon": 6, "model": "amazon/chronos-2"}
    with pytest.raises(ft.ForecastPending):
        ft.forecast("GEX", **recipe_kwargs)
    name = json.loads(Path(ft._REQUESTS).read_text(encoding="utf-8"))[0]["name"]
    _add_feature(ft, name, _fc_frame(["2024-01-01 10:00:00", "2024-01-01 10:00:20"], [1.0, 2.0]))

    df = pd.DataFrame({"SlotUtc": pd.to_datetime(["2024-01-01 10:00:10", "2024-01-01 09:59:50",
                                                  "2024-01-01 10:00:30"]), "x": [1, 2, 3]})
    out = ft.forecast("GEX", join=df, time_col="SlotUtc", **recipe_kwargs)
    # Original row order kept; each row sees only the forecast made at or before it.
    assert list(out["x"]) == [1, 2, 3]
    med = list(out["GEX_fc_median"])
    assert med[0] == 1.0          # 10:00:10 -> forecast made at 10:00:00, never the 10:00:20 one
    assert pd.isna(med[1])        # 09:59:50 -> no forecast existed yet
    assert med[2] == 2.0
    assert "GEX_t" in out.columns and "t" not in out.columns


def test_two_joined_forecasts_do_not_clash(ft):
    for col, val in (("GEX", 1.0), ("Imb", 5.0)):
        with pytest.raises(ft.ForecastPending):
            ft.forecast(col, horizon=3)
    for entry in json.loads(Path(ft._REQUESTS).read_text(encoding="utf-8")):
        v = 1.0 if entry["recipe"]["column"] == "GEX" else 5.0
        _add_feature(ft, entry["name"], _fc_frame(["2024-01-01 10:00"], [v]))
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01 10:01"])})
    out = ft.forecast("Imb", horizon=3, join=ft.forecast("GEX", horizon=3, join=df, time_col="ts"), time_col="ts")
    assert out.loc[0, "GEX_fc_median"] == 1.0
    assert out.loc[0, "Imb_fc_median"] == 5.0


def test_loads_are_recorded_for_the_scoreboard(ft):
    _add_feature(ft, "a", _fc_frame(["2024-01-01 10:00"], [2.0]))
    ft.load("fc_a")
    ft.load("fc_a", prefix="x_")
    assert json.loads((Path(ft._FT) / "used.json").read_text(encoding="utf-8")) == ["fc_a"]
