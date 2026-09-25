"""`import ft` -- what a swarm candidate uses to read project data and report its result.

Shipped into each objective run as /work/.ft/ft.py (it is not baked into the image, so it
changes without a rebuild). The harness that scores a candidate is NOT in here: this file only
loads data and writes /work/.ft/result.json. Scoring happens on the host, from that file, with
code the candidate cannot see or change.

    import ft
    df = ft.load("sql_exports_dbo_gexbar10s")          # pandas DataFrame
    ...
    ft.report_returns(daily_returns)                  # pd.Series indexed by date
    ft.report(trades=n_trades, turnover=0.8)          # optional extra numbers

The project's code library is importable too: ``from lib import some_module``.
"""

from __future__ import annotations

import json
import math
import os

_DATA = "/data"
_FT = "/work/.ft"
_RESULT = os.path.join(_FT, "result.json")

try:
    with open(os.path.join(_FT, "catalog.json"), encoding="utf-8") as _fh:
        _CATALOG = json.load(_fh)
except OSError:
    _CATALOG = []


def datasets() -> list[str]:
    """The view names you can pass to load()."""
    return [c["view"] for c in _CATALOG]


def _find(name: str) -> dict:
    for c in _CATALOG:
        if name in (c["view"], c["path"]):
            return c
    raise KeyError(f"no dataset {name!r}; available: {', '.join(datasets()) or '(none)'}")


def path(name: str) -> str:
    """Filesystem path of a dataset inside the sandbox (a folder for exported datasets).
    Forecast features (views named fc_...) live under /features, project data under /data."""
    item = _find(name)
    rel = item["path"]
    if rel.endswith("/*.parquet"):
        rel = rel[: -len("/*.parquet")]
    return os.path.join(item.get("root") or _DATA, rel)


def _note_used(view: str) -> None:
    """Record which datasets the script loaded (the harness reads it: which forecasts a
    candidate actually used is what the forecast scoreboard is built from)."""
    used_path = os.path.join(_FT, "used.json")
    try:
        with open(used_path, encoding="utf-8") as fh:
            used = json.load(fh)
    except (OSError, ValueError):
        used = []
    if view not in used:
        used.append(view)
        try:
            with open(used_path, "w", encoding="utf-8") as fh:
                json.dump(used, fh)
        except OSError:
            pass


def load(name: str, columns: list[str] | None = None, prefix: str | None = None):
    """Load a dataset as a pandas DataFrame. `columns` limits what is read (parquet only).

    `prefix` renames every column except the time column ``t``: forecast features all share
    column names (fc_median, fc_q10, ...), so merging two of them leaves pandas' fc_median_x /
    fc_median_y and ``df["fc_median"]`` fails. ``ft.load("fc_a", prefix="a_")`` gives a_fc_median.
    """
    import pandas as pd

    item = _find(name)
    _note_used(item["view"])
    p = path(name)
    fmt = item.get("format", "")
    if fmt == "parquet":
        df = pd.read_parquet(p, columns=columns)
    elif fmt in ("csv", "tsv"):
        df = pd.read_csv(p, sep="\t" if fmt == "tsv" else ",", usecols=columns)
    elif fmt in ("jsonl", "ndjson"):
        df = pd.read_json(p, lines=True)
    else:
        df = pd.read_json(p)
    if prefix:
        df = df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c != "t"})
    return df


# What makes one forecast different from another. The name of a forecast built from a recipe is
# a hash of exactly these keys (canonical JSON), computed identically by the harness -- so the
# same call always finds the same stored forecast, and the recipe is on record to rebuild it.
_RECIPE_KEYS = ("dataset", "column", "columns", "covariates", "calendar", "horizon", "every", "context", "model", "bar")
_REQUESTS = os.path.join(_FT, "forecast_requests.json")


class ForecastPending(RuntimeError):
    """The forecast this script asked for is not built yet. The harness builds it (causally,
    with the loaded forecaster) and runs the script again -- nothing to catch or handle."""


def recipe_name(recipe: dict) -> str:
    import hashlib

    canon = json.dumps({k: recipe.get(k) for k in _RECIPE_KEYS}, sort_keys=True, separators=(",", ":"))
    return "auto_" + hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]


def forecast(column: str | None = None, *, columns: list[str] | None = None, inputs: list[str] | None = None,
             calendar: bool = False, horizon: int = 12, every: int = 0, context: int = 512,
             model: str | None = None, dataset: str | None = None, bar: str | None = None,
             join=None, time_col: str | None = None, prefix: str | None = None):
    """A time-series model's forecasts, from inside your script.

        fc = ft.forecast("Imb_OINet_D0", inputs=["GEX", "Pressure_Total"], horizon=6,
                         model="amazon/chronos-2", join=df, time_col="SlotUtc")
        df["edge"] = fc["Imb_OINet_D0_fc_median"] - fc["Imb_OINet_D0_last"]

    Made CAUSALLY: the forecast stamped at bar t read data up to and including t only.
    `column` (or several `columns`) is what is forecast -- a column or an expression over
    columns; `inputs` are other columns the model READS (Chronos-2 only); `calendar` adds time
    of day / weekday. `every` = bars between forecasts (0 = automatic). The first run that asks
    for a new recipe stops with ForecastPending; the harness builds it and re-runs the script,
    and every later run loads it at once. Identical calls share one stored forecast.

    Returns the forecast frame (t, last, fc_median, fc_q10, fc_q90, fc_path_mean, fc_change),
    or -- with `join=df` -- df with those columns attached by an as-of BACKWARD join on
    `time_col` (the only direction that cannot leak), prefixed so two forecasts never clash
    (default prefix: the series name + "_").
    """
    recipe = {"dataset": dataset, "column": column, "columns": list(columns) if columns else None,
              "covariates": list(inputs) if inputs else None, "calendar": bool(calendar),
              "horizon": int(horizon), "every": int(every), "context": int(context), "model": model, "bar": bar}
    if not (column or columns):
        raise ValueError("ft.forecast needs the `column` (or `columns`) to forecast")
    name = recipe_name(recipe)
    view = f"fc_{name}"
    if view not in datasets():
        try:
            with open(_REQUESTS, encoding="utf-8") as fh:
                pending = json.load(fh)
        except (OSError, ValueError):
            pending = []
        if not any(p.get("name") == name for p in pending):
            pending.append({"name": name, "recipe": recipe})
            with open(_REQUESTS, "w", encoding="utf-8") as fh:
                json.dump(pending, fh)
        print(f"[ft] forecast {view} is not built yet: the harness builds it and runs this script again")
        raise ForecastPending(view)
    if join is None:
        return load(view, prefix=prefix)
    import pandas as pd

    if prefix is None:
        base = column or "_".join(columns or [])
        prefix = "".join(ch if ch.isalnum() else "_" for ch in base).strip("_")[:40] + "_"
    fc = load(view, prefix=prefix).rename(columns={"t": f"{prefix}t"}).sort_values(f"{prefix}t")
    tc = time_col or next((c for c in join.columns if str(join[c].dtype).startswith("datetime")), None)
    if tc is None:
        raise ValueError("ft.forecast(join=df) needs time_col= (df has no datetime column)")
    left = join.copy()
    left[tc] = pd.to_datetime(left[tc])
    fc[f"{prefix}t"] = pd.to_datetime(fc[f"{prefix}t"])
    if getattr(left[tc].dt, "tz", None) is not None and getattr(fc[f"{prefix}t"].dt, "tz", None) is None:
        fc[f"{prefix}t"] = fc[f"{prefix}t"].dt.tz_localize(left[tc].dt.tz)
    elif getattr(left[tc].dt, "tz", None) is None and getattr(fc[f"{prefix}t"].dt, "tz", None) is not None:
        fc[f"{prefix}t"] = fc[f"{prefix}t"].dt.tz_localize(None)
    order = left.index
    out = pd.merge_asof(left.sort_values(tc), fc, left_on=tc, right_on=f"{prefix}t", direction="backward")
    out.index = left.sort_values(tc).index
    return out.loc[order]


_OHLC = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample(df, rule: str, time_col: str | None = None):
    """Bars of another timeframe: "20s", "30s", "1min", "5min", "15min", "1h", ...

    Open/High/Low/Close/Volume columns (any capitalisation) aggregate as OHLCV; every other
    numeric column takes its LAST value in the bar. Each bar is stamped with the timestamp of
    the LAST underlying bar it contains -- the moment it is complete and known -- never the
    bucket's start, so a decision made on a resampled bar is causal as stamped. The returned
    frame keeps the time column (and `bar_rows`, the number of base bars per bar).
    """
    import pandas as pd

    tc = time_col or next((c for c in df.columns if str(df[c].dtype).startswith("datetime")), None)
    if tc is None:
        raise ValueError("resample needs a datetime column; pass time_col=")
    d = df.sort_values(tc)
    t = pd.to_datetime(d[tc])
    bucket = t.dt.floor(rule)
    agg = {}
    for c in d.columns:
        if c == tc:
            continue
        how = _OHLC.get(str(c).lower())
        if how:
            agg[c] = how
        elif pd.api.types.is_numeric_dtype(d[c]):
            agg[c] = "last"
    g = d.groupby(bucket.values)
    out = g.agg(agg)
    out[tc] = g[tc].last().values          # stamped at the bar's LAST underlying timestamp
    out["bar_rows"] = g.size().values
    return out.reset_index(drop=True)[[tc] + [c for c in out.columns if c != tc]]


def align(values, value_times, base_times):
    """Carry values stamped at `value_times` (e.g. positions from resampled bars) onto the
    base bar timestamps `base_times`: each base bar gets the latest value at or before it."""
    import numpy as np
    import pandas as pd

    src = pd.DataFrame({"t": pd.to_datetime(pd.Series(np.asarray(value_times))), "v": np.asarray(values, dtype=float)})
    dst = pd.DataFrame({"t": pd.to_datetime(pd.Series(np.asarray(base_times)))})
    order = np.argsort(dst["t"].values, kind="stable")
    merged = pd.merge_asof(dst.iloc[order], src.sort_values("t"), on="t", direction="backward")
    out = np.empty(len(dst))
    out[order] = merged["v"].fillna(0.0).to_numpy()
    return pd.Series(out, index=pd.to_datetime(dst["t"]))


def route(regimes, mapping: dict, default: float = 0.0):
    """Pick each bar's position from the signal mapped to that bar's regime.

        regime = lib.regime_gex_vol.detect(df)
        pos = ft.route(regime, {"neg_gamma_volatile": momentum.signal(df),
                                "pos_gamma_calm": mean_revert.signal(df)})   # others -> default

    `regimes` is one label per row; each mapping value is a Series (or array) of positions
    aligned to the same rows, or a constant. Returns a float Series on the regimes' index.
    """
    import numpy as np
    import pandas as pd

    labels = pd.Series(np.asarray(regimes)).astype(str)
    out = np.full(len(labels), float(default))
    for label, pos in mapping.items():
        mask = (labels == str(label)).to_numpy()
        vals = np.broadcast_to(np.asarray(pos, dtype=float), (len(labels),)) if np.ndim(pos) else np.full(len(labels), float(pos))
        out[mask] = vals[mask]
    index = regimes.index if isinstance(regimes, pd.Series) else None
    return pd.Series(np.nan_to_num(out), index=index)


def _merge(update: dict) -> None:
    os.makedirs(_FT, exist_ok=True)
    try:
        with open(_RESULT, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        doc = {}
    doc.update(update)
    with open(_RESULT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)


def _position_times(index):
    """The bar timestamps a positions series is indexed by, or a ValueError that says why not.

    pd.to_datetime accepts a RangeIndex without complaint and turns row 0..N-1 into
    1970-01-01 00:00:00.000000000 .. +N ns. The usual way to get there is reset_index() before
    reporting; the harness then sees half a million positions inside one microsecond of 1970
    and every downstream check fails with an error that names none of this. So a numeric
    index is refused outright; anything else (DatetimeIndex, tz-aware, strings, dates,
    Timestamps in an object index, periods) is converted as before.
    """
    import pandas as pd

    if isinstance(index, pd.MultiIndex):
        raise ValueError("report_positions: positions must be indexed by the bar timestamp alone, "
                         f"not a MultiIndex ({', '.join(str(n) for n in index.names)}) -- "
                         "e.g. series.droplevel(...) or set_index(time_col)")
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    if isinstance(index, pd.DatetimeIndex):
        return pd.to_datetime(index)
    if pd.api.types.is_numeric_dtype(index) or pd.api.types.is_bool_dtype(index):
        kind = ("a RangeIndex (row numbers) -- did you call reset_index() or pass a list/array?"
                if isinstance(index, pd.RangeIndex) else f"an index of dtype {index.dtype} (row numbers or epoch values?)")
        raise ValueError(
            "report_positions: positions must be indexed by the bar timestamp "
            "(e.g. df.set_index(time_col)['pos'] or pd.Series(pos.values, index=df[time_col])); "
            f"got {kind}")
    try:
        return pd.to_datetime(index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"report_positions: the index is not bar timestamps ({exc}); index the "
                         "series by the time column, e.g. pd.Series(pos.values, index=df[time_col])") from None


def report_positions(positions) -> None:
    """Report the strategy's POSITIONS -- the preferred way to report a trading strategy.

    `positions` is a pandas Series indexed by bar timestamp: the position (e.g. -1, 0, 0.5, 1)
    decided at the close of that bar using only data up to and including that bar. A series
    indexed by row numbers (e.g. after reset_index()) is refused -- it has no times. The harness
    holds each position until the next reported one and computes the returns itself, from the
    dataset's prices, with trading costs -- so you never compute or report returns yourself.
    """
    import pandas as pd

    s = positions if isinstance(positions, pd.Series) else pd.Series(positions)
    if len(s) == 0:
        raise ValueError("report_positions got an empty series")
    df = pd.DataFrame({
        "t": _position_times(s.index),
        "pos": pd.to_numeric(pd.Series(s.values), errors="coerce").fillna(0.0).astype("float64"),
    })
    if getattr(df["t"].dt, "tz", None) is not None:
        # The harness compares in UTC wall time; match it.
        df["t"] = df["t"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=["t"]).drop_duplicates("t", keep="last").sort_values("t")
    if df.empty:
        raise ValueError("report_positions: every timestamp in the index is missing (NaT)")
    lo, hi = df["t"].iloc[0], df["t"].iloc[-1]
    if lo < pd.Timestamp("1990-01-01") or hi > pd.Timestamp("2100-01-01"):
        # Integers read as timestamps are nanoseconds after 1970-01-01, so row numbers all land
        # on the first second of 1970 -- a series that "reports" fine and then marks to market
        # as one constant position. Refuse it here, where the cause can still be named.
        raise ValueError(
            f"report_positions: positions are dated {lo} .. {hi}, which is not the data's time range. "
            f"Index the series by the bar timestamp (e.g. pd.Series(pos.values, index=df[time_col])), "
            f"not by row numbers or epoch integers.")
    os.makedirs(_FT, exist_ok=True)
    df.to_parquet(os.path.join(_FT, "positions.parquet"), index=False)
    print(f"[ft] reported {len(df)} positions ({df['t'].iloc[0]} .. {df['t'].iloc[-1]}), "
          f"{int((df['pos'].diff().abs() > 0).sum())} changes")


def report_returns(returns, dates=None) -> None:
    """Report the strategy's return stream, net of costs.

    `returns` is a pandas Series indexed by date/timestamp (or a sequence, with `dates`).
    Several returns on the same day (intraday bars or trades) are compounded into that day's
    return, so the harness always scores DAILY returns. Days with no position should be 0.0.
    """
    import pandas as pd

    s = returns if isinstance(returns, pd.Series) else pd.Series(list(returns), index=list(dates or []))
    if len(s) == 0:
        raise ValueError("report_returns got an empty series")
    idx = pd.to_datetime(s.index)
    s = pd.Series(pd.to_numeric(s.values, errors="coerce"), index=idx).dropna()
    daily = (1.0 + s).groupby(s.index.normalize()).prod() - 1.0
    daily = daily.sort_index()
    out = [[d.strftime("%Y-%m-%d"), float(r)] for d, r in daily.items() if math.isfinite(float(r))]
    print(f"[ft] reported {len(out)} daily returns ({out[0][0]} .. {out[-1][0]})" if out else "[ft] no finite returns")
    _merge({"returns": out, "intraday_points": int(len(s))})


def report_score(value: float) -> None:
    """Report a single score (for objectives measured by a reported score)."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError("score must be a finite number")
    _merge({"score": v})


def report(**numbers) -> None:
    """Extra numbers worth showing next to the score (trade count, turnover, ...)."""
    extra = {}
    for k, v in numbers.items():
        try:
            f = float(v)
            if math.isfinite(f):
                extra[str(k)[:40]] = f
        except (TypeError, ValueError):
            extra[str(k)[:40]] = str(v)[:200]
    _merge({"extra": extra})
