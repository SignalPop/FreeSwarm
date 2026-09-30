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

Task objectives (scored by a task server) read rows and report actions instead:

    rows = ft.rows()                                  # the task's time-aligned rows
    spec = ft.task()                                  # target, action meaning and bounds
    ft.report_actions(pd.Series(values, index=rows["t"]))

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


def _pandas_compat() -> None:
    """The sandbox runs pandas 3, but models write pandas 2 from memory: fillna(method="ffill"),
    .pad(), .backfill(), .applymap() were removed there and kill an otherwise working candidate.
    Map them onto their replacements (same results) once, on `import ft`."""
    try:
        import pandas as pd
    except ImportError:
        return
    fills = {"ffill": "ffill", "pad": "ffill", "bfill": "bfill", "backfill": "bfill"}
    for cls in (pd.Series, pd.DataFrame):
        orig = cls.fillna
        if getattr(orig, "_ft_compat", False):
            continue

        def fillna(self, value=None, *args, method=None, _orig=orig, **kw):
            if method is None:
                return _orig(self, value, *args, **kw)
            if method not in fills:
                raise ValueError(f"fillna(method={method!r}): use 'ffill' or 'bfill'")
            return getattr(self, fills[method])(**{k: v for k, v in kw.items() if k in ("axis", "inplace", "limit")})

        fillna._ft_compat = True
        cls.fillna = fillna
        if not hasattr(cls, "pad"):
            cls.pad = cls.ffill
        if not hasattr(cls, "backfill"):
            cls.backfill = cls.bfill
    if not hasattr(pd.DataFrame, "applymap"):
        pd.DataFrame.applymap = pd.DataFrame.map


_pandas_compat()


def datasets() -> list[str]:
    """The view names you can pass to load()."""
    return [c["view"] for c in _CATALOG]


def _unique(columns):
    """`columns` without repeats, in order. Models list a column twice in long column lists
    (['IV_AtmD0', ..., 'IV_AtmD0']); polars refuses that with a DuplicateError deep in its planner
    and the run is lost for nothing."""
    return None if columns is None else list(dict.fromkeys(columns))


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

    columns = _unique(columns)
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


def load_pl(name: str, columns: list[str] | None = None, prefix: str | None = None):
    """Load a dataset as a POLARS DataFrame -- several times faster than load() on the 10s bar
    data (700k+ rows). Same arguments as load(); convert with .to_pandas() if you need pandas.

        df = ft.load_pl("sql_exports_dbo_gexbar10s", columns=["ts", "Close", "GEX"])
    """
    import polars as pl

    columns = _unique(columns)
    item = _find(name)
    _note_used(item["view"])
    p = path(name)
    fmt = item.get("format", "")
    if fmt == "parquet":
        src = os.path.join(p, "*.parquet") if os.path.isdir(p) else p
        df = pl.scan_parquet(src)
        df = (df.select(columns) if columns else df).collect()
    elif fmt in ("csv", "tsv"):
        df = pl.read_csv(p, separator="\t" if fmt == "tsv" else ",", columns=columns, try_parse_dates=True)
    elif fmt in ("jsonl", "ndjson"):
        df = pl.read_ndjson(p)
    else:
        df = pl.read_json(p)
    if prefix:
        df = df.rename({c: f"{prefix}{c}" for c in df.columns if c != "t"})
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
    if tc in df.index.names:
        # df.index = df[tc] (or set_index(tc, drop=False)): pandas cannot tell the index level
        # from the column and refuses to sort by it. The column is all that is used here.
        df = df.reset_index(drop=True)
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
    # A frame with the time column alone has nothing to aggregate (pandas: "No objects to
    # concatenate"); its bars still have a timestamp and a row count.
    out = g.agg(agg) if agg else pd.DataFrame(index=g.size().index)
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


def _naive_times(values):
    """Timestamps as tz-naive UTC wall time (the harness compares in UTC everywhere)."""
    import pandas as pd

    t = pd.DatetimeIndex(pd.to_datetime(values))
    return t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t


def _is_time_index(index) -> bool:
    import pandas as pd

    return isinstance(index, pd.DatetimeIndex)


def route(regimes, mapping: dict, default: float = 0.0):
    """Pick each bar's position from the signal mapped to that bar's regime.

        regime = lib.regime_gex_vol.detect(df)
        pos = ft.route(regime, {"neg_gamma_volatile": momentum.signal(df),
                                "pos_gamma_calm": mean_revert.signal(df)})   # others -> default

    `regimes` is one label per row; each mapping value is a Series (or array) of positions
    aligned to the same rows, or a constant. Returns a float Series on the regimes' index.

    When `regimes` is indexed by bar time (ft.regime_grid(..., time=...) is) a value that is
    also time-indexed -- another timeframe's positions, or ft.candidate_positions(n) -- is
    carried onto the regime's bars as-of BACKWARD (the latest position at or before each bar),
    so strategies on 15-minute bars route cleanly on a 10-second regime.
    """
    import numpy as np
    import pandas as pd

    import types

    for label, pos in mapping.items():
        if callable(pos) or isinstance(pos, types.ModuleType):
            what = getattr(pos, "__name__", type(pos).__name__)
            raise TypeError(
                f"ft.route: the value for regime {label!r} is {what!r}, a function or module, not positions. "
                f"Pass what it RETURNS, e.g. {{{label!r}: my_signal.signal(df)}} or ft.candidate_positions(1233).")
    labels = pd.Series(np.asarray(regimes)).astype(str)
    index = regimes.index if isinstance(regimes, pd.Series) else None
    times = _naive_times(index) if index is not None and _is_time_index(index) else None
    _ROUTED.clear()
    _ROUTED.update(labels=labels.to_numpy(), index=index,
                   routes={str(k): str(getattr(v, "name", None) or k) for k, v in mapping.items()},
                   name=str(getattr(regimes, "name", None) or "regime"))
    out = np.full(len(labels), float(default))
    for label, pos in mapping.items():
        mask = (labels == str(label)).to_numpy()
        if not mask.any():
            continue
        if (times is not None and isinstance(pos, pd.Series) and _is_time_index(pos.index)
                and not (len(pos) == len(times) and _naive_times(pos.index).equals(times))):
            vals = align(pos.to_numpy(dtype=float), _naive_times(pos.index), times).to_numpy()
        elif np.ndim(pos):
            arr = np.asarray(pos, dtype=float)
            if len(arr) != len(labels):
                raise ValueError(
                    f"ft.route: positions for regime {label!r} have {len(arr)} rows but the regime has {len(labels)}. "
                    "Compute both on the same rows, or index both by bar time (the regime with "
                    "ft.regime_grid(..., time=TIME), the positions with pd.Series(pos, index=df[TIME])) so route aligns them.")
            vals = arr
        else:
            vals = np.full(len(labels), float(pos))
        out[mask] = vals[mask]
    return pd.Series(np.nan_to_num(out), index=index)


def _bucket_names(n: int) -> list[str]:
    return {2: ["low", "high"], 3: ["low", "mid", "high"]}.get(int(n)) or [f"q{i + 1}" for i in range(int(n))]


def regime_grid(df, fields, n: int = 3, time: str | None = None, smooth: int = 360, window_days: float = 20,
                window: int | None = None):
    """Causal regimes from one or more columns, crossed into a grid.

        regime = ft.regime_grid(df, {"GEX": 3, "IntrVol": 3}, time="SlotUtc")
        # -> "GEX:high|IntrVol:low", ... (9 regimes), "warmup" before there is history

    Each field is first smoothed by a trailing mean over `smooth` bars (360 = 1 hour of 10 s bars;
    0 = raw), then ranked against its OWN trailing window -- `window` bars, or `window_days`
    sessions of bars when `time` is given -- and cut into equal buckets (2: low/high, 3:
    low/mid/high, more: q1..qn). Every step reads only bars at or before the one labelled, so a
    label never knows the future. The label of several fields is their buckets joined by "|".

    With `time` the result is indexed by that column's timestamps, which lets ft.route align
    positions from any timeframe onto it; without, by df's own index. This is exactly the
    labelling the Regime Lab measures, so its routes carry over to a script unchanged.
    """
    import numpy as np
    import pandas as pd

    spec = ({str(k): int(v) for k, v in fields.items()} if isinstance(fields, dict)
            else {str(f): int(n) for f in ([fields] if isinstance(fields, str) else fields)})
    if not spec:
        raise ValueError("regime_grid needs at least one field")
    order = np.arange(len(df))
    index = df.index
    bars_per_day = None
    if time is not None:
        t = pd.to_datetime(df[time])
        order = np.argsort(t.to_numpy(), kind="stable")
        index = pd.DatetimeIndex(t)
        bars_per_day = float(t.dt.normalize().value_counts().median()) if len(t) else None
    win = int(window or (window_days * bars_per_day if bars_per_day else 2000))
    win = max(20, win)
    label = None
    warm = np.zeros(len(df), dtype=bool)
    for field, k in spec.items():
        if k < 2:
            raise ValueError(f"regime_grid: {field} needs at least 2 buckets, got {k}")
        x = pd.Series(pd.to_numeric(df[field], errors="coerce").to_numpy(dtype="float64")[order])
        if smooth and int(smooth) > 1:
            x = x.rolling(int(smooth), min_periods=1).mean()
        pct = x.rolling(win, min_periods=max(20, win // 4)).rank(pct=True).to_numpy()
        b = np.clip(np.ceil(pct * k) - 1, 0, k - 1)
        warm |= np.isnan(b)
        names = np.asarray([f"{field}:{nm}" for nm in _bucket_names(k)], dtype=object)
        part = names[np.nan_to_num(b).astype(int)]
        label = part if label is None else label + "|" + part
    label = np.where(warm, "warmup", label).astype(object)
    out = np.empty(len(df), dtype=object)
    out[order] = label
    return pd.Series(out, index=index, name="regime")


# While another candidate's script runs inside this one (candidate_positions), what it reports
# is captured here instead of being written as this run's result.
_CAPTURE: dict | None = None
_MEMBERS: dict = {}


def candidate_positions(seq: int):
    """The positions another VERIFIED candidate of this objective reports, computed here and now.

        a = ft.candidate_positions(1233)          # a pd.Series indexed by bar time
        pos = ft.route(regime, {"GEX:high|IntrVol:low": a, ...})

    The candidate's own script runs inside this one, on the same data this run sees (so a
    look-ahead test that truncates the data truncates it too), and whatever it reports is
    captured instead of reported. Only candidates that scored, passed the look-ahead test and
    were not disqualified are available; a script that reports returns rather than positions
    cannot be routed. Each one runs once per script however often it is asked for.
    """
    global _CAPTURE
    seq = int(str(seq).lstrip("#"))
    if seq in _MEMBERS:
        return _MEMBERS[seq].copy()
    src_path = os.path.join(_FT, "members", f"{seq}.py")
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"candidate #{seq} is not available: only verified candidates of this objective (scored, look-ahead "
            "passed, not disqualified, not an ensemble) can be used, and the call must name the number literally, "
            "e.g. ft.candidate_positions(1233)")
    with open(src_path, encoding="utf-8") as fh:
        src = fh.read()
    saved, prev = dict(_ROUTED), _CAPTURE
    _CAPTURE = {}
    try:
        print(f"[ft] running candidate #{seq} for its positions")
        exec(compile(src, f"candidate_{seq}.py", "exec"), {"__name__": "__main__", "__file__": f"candidate_{seq}.py"})
        got = _CAPTURE.get("positions")
    finally:
        _CAPTURE = prev
        _ROUTED.clear()
        _ROUTED.update(saved)
    if got is None:
        raise ValueError(f"candidate #{seq} reported no positions (it reports returns or a score), so it cannot be routed")
    got.name = f"#{seq}"
    _MEMBERS[seq] = got
    return got.copy()


# What the last ft.route call routed, so report_positions can record the regimes for the
# equity chart without the script having to report them separately.
_ROUTED: dict = {}


def regimes(signal, n: int = 3, window: int = 2000, labels: list[str] | None = None, min_periods: int | None = None):
    """Causal regime labels from any series: where each bar's value sits among the previous
    `window` bars of the same series, cut into `n` equal buckets.

        vol_regime = ft.regimes(df["IntrVol"], n=3, window=6 * 390)   # low / mid / high vs ~last 6 days

    Default labels are low/mid/high for n=3 and q1..qn otherwise. The rank uses only bars up
    to and including this one (a rolling window, never the full sample), so the first part
    of the data is not labelled with knowledge of the rest; warm-up bars are labelled "warmup".
    Name the series (``.rename("vol_regime")``) and that name is shown on the chart.
    """
    import numpy as np
    import pandas as pd

    x = signal.astype("float64") if isinstance(signal, pd.Series) else pd.Series(signal, dtype="float64")
    pct = x.rolling(int(window), min_periods=int(min_periods or max(20, window // 4))).rank(pct=True)
    names = labels or (["low", "mid", "high"] if n == 3 else [f"q{i + 1}" for i in range(n)])
    if len(names) != n:
        raise ValueError(f"regimes: {n} buckets need {n} labels, got {len(names)}")
    idx = np.clip(np.ceil(pct.to_numpy() * n) - 1, 0, n - 1)
    out = np.where(np.isnan(idx), "warmup", np.asarray(names, dtype=object)[np.nan_to_num(idx).astype(int)])
    return pd.Series(out, index=x.index, name=getattr(signal, "name", None) and f"{signal.name}_regime")


def report_regime(labels, signal=None, name: str | None = None, routes: dict | None = None) -> None:
    """Show which regime was active when, on the candidate's equity curve and a plot beneath it.

    `labels` is one regime label per bar, indexed by bar timestamp like report_positions.
    `signal` (optional, same index) is the number the regime was derived from, drawn under
    the equity curve so you can see what the regime was responding to. `routes` maps each
    label to the signal traded in it (for the legend). ft.route records this automatically
    when its labels line up with the positions you report; call this to override or add
    `signal`.
    """
    import pandas as pd

    lab = labels if isinstance(labels, pd.Series) else pd.Series(labels)
    df = pd.DataFrame({"t": _position_times(lab.index), "label": lab.astype(str).to_numpy()})
    if signal is not None:
        sig = signal if isinstance(signal, pd.Series) else pd.Series(signal, index=lab.index)
        df["value"] = pd.to_numeric(pd.Series(sig.reindex(lab.index).to_numpy()), errors="coerce")
    _write_regime(df, name or str(getattr(labels, "name", None) or "regime"), routes)


def _write_regime(df, name: str, routes: dict | None) -> None:
    if _CAPTURE is not None:
        return
    os.makedirs(_FT, exist_ok=True)
    if getattr(df["t"].dt, "tz", None) is not None:
        df["t"] = df["t"].dt.tz_convert("UTC").dt.tz_localize(None)
    df.to_parquet(os.path.join(_FT, "regime.parquet"), index=False)
    _merge({"regime": {"name": name[:80], "routes": {str(k)[:60]: str(v)[:80] for k, v in (routes or {}).items()}}})


def inverse_vol(price, lookback: int = 60, clip: tuple[float, float] = (0.25, 3.0), halflife: int | None = None):
    """A causal inverse-volatility multiplier per bar: above 1 when recent volatility is low,
    below 1 when it is high, about 1 on average.

        scale = ft.inverse_vol(df["Close"], lookback=60)   # 60 bars of whatever timeframe df is

    Volatility is the rolling std of log returns over the last `lookback` bars (including this
    one); it is compared with its own long exponential average (half-life `halflife` bars,
    default 20 x lookback), so no full-sample statistic leaks in. NaN warm-up bars get 1.0.
    """
    import numpy as np
    import pandas as pd

    p = pd.Series(price, dtype="float64") if not isinstance(price, pd.Series) else price.astype("float64")
    r = np.log(p.where(p > 0)).diff()
    vol = r.rolling(int(lookback), min_periods=max(2, int(lookback) // 2)).std()
    ref = vol.ewm(halflife=int(halflife or 20 * lookback), min_periods=int(lookback)).mean()
    return (ref / vol.where(vol > 0)).clip(*clip).fillna(1.0)


def size(direction, scale=1.0, *, base: float = 1.0, step: float = 0.5, rebalance: str = "entry",
         band: float = 0.5, max_leverage: float | None = None):
    """Turn a direction (-1/0/+1, or any signed signal) and a size multiplier into positions
    that do not pay costs for nothing.

        pos = ft.size(direction, ft.inverse_vol(df["Close"], 60), base=2.0)

    Every size change is a trade, so the size is NOT rescaled every bar:
    * rebalance="entry" (default): the size is chosen when a trade opens or flips side
      -- base * |direction| * scale at that bar, rounded to `step` -- and held until it closes.
    * rebalance="band": the size is also reset while in a trade, but only when the target
      has drifted at least `band` away from the size held.
    Sizes are rounded to multiples of `step` (0 disables rounding) and capped at `max_leverage`.
    Causal: each bar uses only its own direction and scale. Returns a float Series.
    `direction` and `scale` are one value per bar of the SAME frame; either may be one number.
    """
    import numpy as np
    import pandas as pd

    n_scale = len(scale) if np.ndim(scale) else None
    if n_scale is not None and np.ndim(direction) == 0:
        # One direction for every bar (ft.size(1, scale)): it takes the scale's bars. The
        # direction used to become a one-row series here, and the scale then failed to
        # broadcast onto it with a numpy error naming neither argument.
        d = pd.Series(float(direction), index=scale.index if isinstance(scale, pd.Series) else range(n_scale))
    else:
        d = pd.Series(direction, dtype="float64") if not isinstance(direction, pd.Series) else direction.astype("float64")
    if n_scale not in (None, 1, len(d)):
        raise ValueError(f"ft.size: direction has {len(d)} values but scale has {n_scale} -- both must be one value "
                         "per bar of the same frame (or a single number)")
    s = np.broadcast_to(np.asarray(scale, dtype=float), (len(d),)) if np.ndim(scale) else np.full(len(d), float(scale))
    target = np.nan_to_num(d.to_numpy()) * base * np.nan_to_num(s, nan=1.0)
    if step:
        target = np.round(target / step) * step
    if max_leverage is not None:
        target = np.clip(target, -max_leverage, max_leverage)
    sign = np.sign(np.nan_to_num(d.to_numpy()))
    out = np.zeros(len(d))
    held = 0.0
    for i in range(len(d)):
        if sign[i] == 0:
            held = 0.0
        elif np.sign(held) != sign[i]:
            held = target[i] if target[i] != 0 else sign[i] * (step or base)  # opening / flipping
        elif rebalance == "band" and abs(target[i] - held) >= band:
            held = target[i]
        out[i] = held
    return pd.Series(out, index=d.index)


# ---------------------------------------------------------------------------------------------
# Intraday trend tools: the session clock, VWAP, the gamma regime, trailing-stop trade management,
# a published breakout baseline, and walk-forward meta-labelling. All causal: the value at a row
# uses that row and earlier rows only -- except label_outcomes, which is for RESEARCH (it reads the
# future on purpose) and which meta_filter only ever uses for days that have already closed.
# ---------------------------------------------------------------------------------------------
_TZ = "America/New_York"


def _hhmm(s) -> float:
    """"15:55" (or 15.55 hours, or minutes past midnight) -> minutes past midnight."""
    if isinstance(s, str):
        h, _, m = s.partition(":")
        return int(h) * 60 + float(m or 0)
    return float(s) * 60 if float(s) < 24 else float(s)


def clock(times, tz: str = _TZ):
    """(session, minute) of each timestamp: the session's LOCAL date (numpy datetime64[D]) and the
    local time of day in minutes past midnight (9:30 = 570.0). Timestamps are the rows' `t`
    (naive UTC); daylight saving is handled by the time zone, so 15:55 is always 15:55 New York.

        session, minute = ft.clock(rows["t"])
        late = minute >= 15 * 60 + 55
    """
    import numpy as np
    import pandas as pd

    t = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
    t = (t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")).tz_convert(tz)
    minute = (t.hour * 60 + t.minute + t.second / 60.0).to_numpy(dtype=float)
    session = t.tz_localize(None).normalize().to_numpy().astype("datetime64[D]")
    return session, minute


def _col(rows, name, default=None):
    import numpy as np

    if name in rows.columns:
        return np.asarray(rows[name], dtype=float)
    if default is not None:
        return default
    raise KeyError(f"no column {name!r} in the rows")


def _session_starts(session):
    import numpy as np

    return np.flatnonzero(np.r_[True, session[1:] != session[:-1]])


def session_vwap(rows, price: str = "Close", volume: str = "Volume", time: str = "t", tz: str = _TZ):
    """The volume-weighted average price since each session's first row, up to and including
    this row (the running mean price where there is no volume). A float Series on rows' index.

        above = rows["Close"] > ft.session_vwap(rows)
    """
    import numpy as np
    import pandas as pd

    p = _col(rows, price)
    v = _col(rows, volume, np.ones(len(p)))
    v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
    ok = np.isfinite(p)
    session, _ = clock(rows[time], tz)
    grp = np.cumsum(np.r_[True, session[1:] != session[:-1]])
    df = pd.DataFrame({"g": grp, "pv": np.where(ok, p * v, 0.0), "v": np.where(ok, v, 0.0),
                       "p": np.where(ok, p, 0.0), "n": ok.astype(float)})
    c = df.groupby("g")[["pv", "v", "p", "n"]].cumsum()
    with np.errstate(all="ignore"):
        out = np.where(c["v"] > 0, c["pv"] / c["v"], c["p"] / c["n"])
    return pd.Series(out, index=rows.index, name="vwap")


def gamma_regime(rows, column: str = "GEX", smooth: int = 360):
    """The dealer-gamma regime of each row: -1 when dealers are SHORT gamma (the column's trailing
    mean over `smooth` rows is below 0: moves tend to extend -- follow breakouts), +1 when LONG
    gamma (moves tend to fade and pin), 0 while there is no value yet. 360 rows = 1 hour of 10 s bars.

        short_gamma = ft.gamma_regime(rows) < 0
    """
    import numpy as np
    import pandas as pd

    g = pd.Series(_col(rows, column)).rolling(int(max(1, smooth)), min_periods=max(1, int(smooth) // 10)).mean()
    out = np.sign(g.fillna(0.0).to_numpy())
    return pd.Series(out, index=rows.index, name="gamma_regime")


def trend_exits(entries, rows, price: str = "Close", time: str = "t", *, size=1.0, stop_mult: float = 1.5,
                vol_window: int = 360, stop_pct: float | None = None, trail: bool = True,
                breakeven_at: float | None = 1.0, vwap_exit: bool = False, flat_at: str = "15:55",
                no_entry_before: str = "09:45", no_entry_after: str = "15:00", max_trades_per_day: int | None = None,
                reverse: bool = True, retrigger: bool = False, tz: str = _TZ):
    """Manage trades from ENTRY signals with trend-style exits: a stop that trails the best price,
    no profit target, a clock exit. Returns positions (a float Series indexed by rows[time]) ready
    for ft.report_actions.

        want = np.sign(my_signal)                    # +1 long, -1 short, 0/NaN no new trade
        pos = ft.trend_exits(want, rows, size=ft.inverse_vol(rows["Close"], 360) * 2)
        ft.report_actions(pos)

    * A change of `entries` to +1/-1 ARMS an entry (retrigger=True: every +1/-1 row does); it opens
      at the first row allowed -- flat, or with reverse=True in the opposite trade; between
      `no_entry_before` and `no_entry_after` (local clock); any number a session unless
      `max_trades_per_day` is given -- and is then used up: a stopped-out trade is not re-entered
      until the signal changes (retrigger=True re-enters while it stays on). A signal still on at a
      new session re-arms.
    * The stop distance is fixed at entry: `stop_pct` percent of the price, or `stop_mult` times
      the typical move over `vol_window` rows (std of log returns x sqrt(vol_window)). It starts at
      entry -/+ distance and, with trail=True, follows the best price since entry. With
      `breakeven_at` the stop moves to the entry price once the trade is that many distances in
      profit; with vwap_exit=True a close through the session VWAP also exits.
    * Everything is flat from `flat_at` (local clock) to the session end.
    `size` is a number or one per row (read at entry, then held). Row i's decision uses rows <= i.
    """
    import numpy as np
    import pandas as pd

    p = _col(rows, price)
    n = len(p)
    e = np.nan_to_num(np.asarray(entries, dtype=float)) if np.ndim(entries) else np.full(n, float(entries))
    e = np.sign(e)
    sz = np.abs(np.nan_to_num(np.asarray(size, dtype=float), nan=1.0)) if np.ndim(size) else np.full(n, abs(float(size)))
    session, minute = clock(rows[time], tz)
    new_day = np.r_[True, session[1:] != session[:-1]]
    lr = np.log(np.where(p > 0, p, np.nan))
    r = pd.Series(np.r_[np.nan, np.diff(lr)])
    move = (r.rolling(int(vol_window), min_periods=max(10, int(vol_window) // 4)).std() * np.sqrt(vol_window)).to_numpy()
    vwap = session_vwap(rows, price=price, time=time, tz=tz).to_numpy() if vwap_exit else None
    t_flat, t_open, t_last = _hhmm(flat_at), _hhmm(no_entry_before), _hhmm(no_entry_after)
    out = np.zeros(n)
    pos = entry = best = dist = stop = 0.0
    trades = 0
    prev_sig = armed = 0.0
    for i in range(n):
        if new_day[i]:
            trades, pos, prev_sig, armed = 0, 0.0, 0.0, 0.0
        x = p[i]
        sig = e[i]
        # A signal ARMS an entry when it changes (or on every row with retrigger); the entry fires
        # at the first row it is allowed and is then used up -- a stopped-out trade is not re-entered
        # until the signal changes again.
        if retrigger or sig != prev_sig:
            armed = sig
        prev_sig = sig
        if not np.isfinite(x):
            out[i] = pos
            continue
        if minute[i] >= t_flat:
            pos = 0.0
            out[i] = 0.0
            continue
        if pos != 0.0:
            s = 1.0 if pos > 0 else -1.0
            if (x - best) * s > 0:
                best = x
            if trail:
                stop = max(stop, best - dist) if s > 0 else min(stop, best + dist)
            if breakeven_at is not None and (best - entry) * s >= breakeven_at * dist:
                stop = max(stop, entry) if s > 0 else min(stop, entry)
            hit = (x <= stop) if s > 0 else (x >= stop)
            if vwap is not None and np.isfinite(vwap[i]):
                hit = hit or ((x < vwap[i]) if s > 0 else (x > vwap[i]))
            if hit:
                pos = 0.0
        if armed != 0 and pos != 0.0 and np.sign(pos) == armed:
            armed = 0.0                                   # already in that trade
        can_open = (armed != 0 and t_open <= minute[i] < t_last
                    and (not max_trades_per_day or trades < max_trades_per_day))
        if can_open and (pos == 0.0 or reverse):
            d = (stop_pct / 100.0 * x) if stop_pct else (move[i] * stop_mult * x if np.isfinite(move[i]) else np.nan)
            if np.isfinite(d) and d > 0:
                pos = armed * (sz[i] if np.isfinite(sz[i]) and sz[i] > 0 else 1.0)
                entry = best = x
                dist = d
                stop = x - d if armed > 0 else x + d
                trades += 1
                armed = 0.0
        out[i] = pos
    return pd.Series(out, index=pd.DatetimeIndex(pd.to_datetime(rows[time])), name="position")


def noise_area_breakout(rows, price: str = "Close", time: str = "t", *, lookback_days: int = 14,
                        band_mult: float = 1.0, check_every: int = 30, first_check: str = "10:00",
                        flat_at: str = "15:55", size=1.0, gate=None, tz: str = _TZ):
    """The intraday-momentum baseline of Zarattini, Aziz & Barbon (2024, 'Beat the Market: An
    Effective Intraday Momentum Strategy for S&P500 ETF (SPY)'): trade a breakout of the 'noise area'
    around the open, exit when the move gives back to that area or to the session VWAP. Returns
    positions (a float Series indexed by rows[time]).

    * For each minute of the day, sigma = the mean absolute move from the open at that minute over
      the previous `lookback_days` sessions (never today). Upper band = max(open, previous close) x
      (1 + band_mult x sigma), lower band = min(open, previous close) x (1 - band_mult x sigma).
    * Only at check times -- every `check_every` minutes from `first_check` -- the position is set:
      long when the price is above the upper band AND the VWAP, short when below the lower band AND
      the VWAP, flat otherwise. It is held between checks, and flat from `flat_at`.
    * `gate` (optional, one per row, truthy = may trade) restricts entries, e.g.
      gate=ft.gamma_regime(rows) < 0 to trade only when dealers are short gamma.

        pos = ft.noise_area_breakout(rows, size=ft.inverse_vol(rows["Close"], 360).clip(0.5, 3))
    """
    import numpy as np
    import pandas as pd

    p = _col(rows, price)
    n = len(p)
    session, minute = clock(rows[time], tz)
    starts = _session_starts(session)
    ends = np.r_[starts[1:], n]
    day_of = np.repeat(np.arange(len(starts)), ends - starts)
    open_ = p[starts][day_of]
    prev_close = np.r_[np.nan, p[ends[:-1] - 1]][day_of]
    # |move from the open| at each whole minute of each session: the last row in that minute.
    mins = np.floor(minute).astype(int)
    grid = np.full((len(starts), 24 * 60), np.nan)
    with np.errstate(all="ignore"):
        grid[day_of, mins] = np.abs(p / open_ - 1.0)          # later rows in a minute overwrite earlier ones
    grid = pd.DataFrame(grid).ffill(axis=1).to_numpy()
    # sigma for session d = mean over sessions d-lookback .. d-1 (shift: today never counts)
    sig_days = (pd.DataFrame(grid).rolling(int(lookback_days), min_periods=int(lookback_days)).mean()
                .shift(1).to_numpy())
    sigma = sig_days[day_of, mins] * float(band_mult)
    hi_ref = np.where(np.isfinite(prev_close), np.maximum(open_, prev_close), open_)
    lo_ref = np.where(np.isfinite(prev_close), np.minimum(open_, prev_close), open_)
    ub, lb = hi_ref * (1 + sigma), lo_ref * (1 - sigma)
    vwap = session_vwap(rows, price=price, time=time, tz=tz).to_numpy()
    step = int(check_every)
    first = _hhmm(first_check)
    # A check fires at the first row of each check minute (e.g. 10:00:00, 10:30:00, ...).
    new_min = np.r_[True, (mins[1:] != mins[:-1]) | (session[1:] != session[:-1])]
    check = new_min & (minute >= first) & (((mins - int(first)) % step) == 0) & (minute < _hhmm(flat_at))
    g = np.ones(n, bool) if gate is None else np.asarray(pd.Series(np.asarray(gate)).fillna(False), dtype=bool)
    sz = np.abs(np.nan_to_num(np.asarray(size, dtype=float), nan=1.0)) if np.ndim(size) else np.full(n, abs(float(size)))
    want = np.full(n, np.nan)
    ok = check & np.isfinite(sigma) & np.isfinite(p)
    longs = ok & (p > ub) & (p > vwap) & g
    shorts = ok & (p < lb) & (p < vwap) & g
    want[ok] = 0.0
    want[longs] = sz[longs]
    want[shorts] = -sz[shorts]
    # A trade that is open keeps its size until the next check changes its side.
    pos = pd.Series(want).groupby(day_of).ffill().fillna(0.0).to_numpy()
    same_side = np.sign(pos)
    run = np.cumsum(np.r_[True, (same_side[1:] != same_side[:-1]) | (day_of[1:] != day_of[:-1])])
    pos = np.array(pd.Series(pos).groupby(run).transform("first"), dtype=float)   # a writable copy
    pos[minute >= _hhmm(flat_at)] = 0.0
    return pd.Series(pos, index=pd.DatetimeIndex(pd.to_datetime(rows[time])), name="position")


def decision_points(rows, times=("09:45", "10:00", "10:30", "11:00", "11:30", "12:00", "13:00", "14:00", "15:00"),
                    time: str = "t", tz: str = _TZ):
    """Row positions (integers) of fixed decision times: for each session and each clock time,
    the first row at or after it. Use them to study what happens after a decision, or to act only
    at those times.

        pts = ft.decision_points(rows, times=["10:00", "10:30", "11:30", "13:00", "14:00"])
        rows.iloc[pts]
    """
    import numpy as np

    session, minute = clock(rows[time], tz)
    starts = _session_starts(session)
    ends = np.r_[starts[1:], len(minute)]
    out = []
    for a, b in zip(starts, ends):
        m = minute[a:b]
        for x in sorted(_hhmm(s) for s in times):
            j = int(np.searchsorted(m, x, side="left"))
            if j < b - a:
                out.append(a + j)
    return np.unique(np.asarray(out, dtype=np.int64))


def label_outcomes(rows, points, price: str = "Close", time: str = "t", horizons=(30, 60, 120),
                   barrier: tuple[float, float] | None = None, tz: str = _TZ):
    """RESEARCH ONLY -- this reads the FUTURE on purpose. What happened after each decision point,
    within its session: one row per point with

      t, session, minute, price,
      ret_close_bps        return to the session's last row,
      ret_<h>m_bps         return over the next h minutes (NaN when that passes the session end),
      mfe_bps / mae_bps    best and worst move up to the close, long side (a short's are mirrored),
      barrier              with barrier=(up_pct, down_pct): +1 if +up_pct came first, -1 if
                           -down_pct came first, 0 if neither before the close.

    Use it in run_python to find which features separate good from bad decisions (compare winners
    WITH losers, group statistics by session -- the points of one day are not independent). Never
    feed these columns to a strategy directly: that is look-ahead. ft.meta_filter uses them the
    only causal way, from sessions that have already closed.
    """
    import numpy as np
    import pandas as pd

    p = _col(rows, price)
    pts = np.asarray(points, dtype=np.int64)
    session, minute = clock(rows[time], tz)
    starts = _session_starts(session)
    ends = np.r_[starts[1:], len(p)]
    day_of = np.repeat(np.arange(len(starts)), ends - starts)
    t = pd.to_datetime(np.asarray(rows[time]))
    rec = {"t": t[pts], "session": session[pts], "minute": minute[pts], "price": p[pts]}
    last = ends[day_of[pts]] - 1
    with np.errstate(all="ignore"):
        rec["ret_close_bps"] = (p[last] / p[pts] - 1.0) * 1e4
        for h in horizons:
            tgt = np.full(len(pts), np.nan)
            for k, i in enumerate(pts):
                a, b = i, ends[day_of[i]]
                q = a + int(np.searchsorted(minute[a:b], minute[i] + float(h), side="left"))
                if q < b:
                    tgt[k] = p[q] / p[i] - 1.0
            rec[f"ret_{int(h)}m_bps"] = tgt * 1e4
        mfe, mae, bar = np.full(len(pts), np.nan), np.full(len(pts), np.nan), np.zeros(len(pts))
        for k, i in enumerate(pts):
            path = p[i:ends[day_of[i]]] / p[i] - 1.0
            path = path[np.isfinite(path)]
            if not len(path):
                continue
            mfe[k], mae[k] = path.max() * 1e4, path.min() * 1e4
            if barrier:
                up = np.flatnonzero(path >= barrier[0] / 100.0)
                dn = np.flatnonzero(path <= -barrier[1] / 100.0)
                fu = up[0] if len(up) else np.inf
                fd = dn[0] if len(dn) else np.inf
                bar[k] = 0 if fu == fd == np.inf else (1 if fu < fd else -1)
    rec["mfe_bps"], rec["mae_bps"] = mfe, mae
    if barrier:
        rec["barrier"] = bar
    return pd.DataFrame(rec)


def meta_filter(features, labels, sessions, *, min_train_sessions: int = 40, refit_every: int = 5,
                l2: float = 1.0, window_sessions: int | None = None):
    """Walk-forward meta-labelling: the probability that each decision is a good one, learned ONLY
    from decisions of sessions that have already closed. Causal, so a strategy may use it.

        pts = ft.decision_points(rows)
        lab = ft.label_outcomes(rows, pts)                     # the future -- used only for past sessions
        X = rows.iloc[pts][["GEX", "IntrVol", "Pressure_Total"]].to_numpy()
        side = np.sign(rows["Close"].iloc[pts] - ft.session_vwap(rows).iloc[pts]).to_numpy()
        good = (side * lab["ret_close_bps"] > 5).to_numpy()   # the primary signal's call was right
        prob = ft.meta_filter(X, good, lab["session"])
        take = prob > 0.55                                       # trade only those

    `features` (n x k), `labels` (n, 0/1 -- NaN rows are not learned from) and `sessions` (n, the
    session of each decision) are aligned. For session d the model -- an L2-regularised logistic
    regression on standardised features -- is fitted on sessions strictly before d (the last
    `window_sessions` of them, or all), refitted every `refit_every` sessions; NaN until
    `min_train_sessions` sessions are available. Missing feature values are filled with the
    training mean.
    """
    import numpy as np

    X = np.asarray(features, dtype=float)
    X = X.reshape(len(X), -1)
    y = np.asarray(labels, dtype=float)
    s = np.asarray(sessions)
    uniq = np.unique(s)
    pos_of = {v: i for i, v in enumerate(uniq)}
    si = np.array([pos_of[v] for v in s])
    out = np.full(len(X), np.nan)
    w = None
    fitted_at = -10**9
    mu = sd = None
    for d in range(len(uniq)):
        if d < int(min_train_sessions):
            continue
        if w is None or d - fitted_at >= int(refit_every):
            lo = 0 if not window_sessions else max(0, d - int(window_sessions))
            m = (si >= lo) & (si < d) & np.isfinite(y)
            if m.sum() < 10 or len(np.unique(y[m])) < 2:
                continue
            Xt = X[m]
            mu = np.nanmean(Xt, axis=0)
            sd = np.nanstd(Xt, axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            Z = np.nan_to_num((Xt - mu) / sd)
            Z = np.c_[np.ones(len(Z)), Z]
            yt = y[m]
            w = np.zeros(Z.shape[1])
            reg = np.r_[0.0, np.full(Z.shape[1] - 1, float(l2))]
            for _ in range(25):                                   # Newton / IRLS
                pr = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
                g = Z.T @ (pr - yt) + reg * w
                H = (Z * (pr * (1 - pr))[:, None]).T @ Z + np.diag(reg + 1e-9)
                step = np.linalg.solve(H, g)
                w -= step
                if np.abs(step).max() < 1e-6:
                    break
            fitted_at = d
        today = si == d
        if w is not None and today.any():
            Z = np.nan_to_num((X[today] - mu) / sd)
            out[today] = 1.0 / (1.0 + np.exp(-np.clip(np.c_[np.ones(len(Z)), Z] @ w, -30, 30)))
    return out


def admit(scores, sessions, per_day: int = 3, *, window_sessions: int = 40, min_sessions: int = 10,
          floor: float | None = None):
    """Causally keep the best `per_day` candidates of each session -- the version of "take the day's
    top 3" that does not need to know the rest of the day. Returns a boolean array (True = take it).

        prob = ft.meta_filter(X, good, lab["session"])               # a score per candidate entry
        take = ft.admit(prob, lab["session"], per_day=3, floor=0.5)

    `scores` and `sessions` are one per candidate, in time order. For each session the bar is set
    from the previous `window_sessions` sessions: the score that would have let through about
    `per_day` candidates a session there. Then the session's candidates are taken IN TIME ORDER
    while their score clears the bar (and `floor`, if given), until `per_day` are taken -- a later,
    better candidate cannot displace an earlier one, just as in live trading. Nothing is taken
    before `min_sessions` sessions of history; NaN scores are never taken.
    """
    import numpy as np

    sc = np.asarray(scores, dtype=float)
    s = np.asarray(sessions)
    out = np.zeros(len(sc), dtype=bool)
    uniq, si = np.unique(s, return_inverse=True)
    order = np.argsort(si, kind="stable")                               # time order within each session
    for d in range(len(uniq)):
        if d < int(min_sessions):
            continue
        lo = max(0, d - int(window_sessions))
        past = sc[(si >= lo) & (si < d)]
        past = past[np.isfinite(past)]
        want = int(per_day) * (d - lo)
        if not len(past):
            continue
        bar = np.sort(past)[::-1][want - 1] if len(past) > want else past.min()
        if floor is not None:
            bar = max(bar, float(floor))
        taken = 0
        for i in order[si[order] == d]:
            if taken >= per_day:
                break
            if np.isfinite(sc[i]) and sc[i] >= bar:
                out[i] = True
                taken += 1
    return out


def _merge(update: dict) -> None:
    if _CAPTURE is not None:
        # A member's own report()/report_returns() is not this run's result.
        return
    os.makedirs(_FT, exist_ok=True)
    try:
        with open(_RESULT, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        doc = {}
    doc.update(update)
    with open(_RESULT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)


def _position_times(index, who: str = "report_positions"):
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
        raise ValueError(f"{who}: the series must be indexed by the bar timestamp alone, "
                         f"not a MultiIndex ({', '.join(str(n) for n in index.names)}) -- "
                         "e.g. series.droplevel(...) or set_index(time_col)")
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    if isinstance(index, pd.DatetimeIndex):
        return pd.to_datetime(index)
    if pd.api.types.is_numeric_dtype(index) or pd.api.types.is_bool_dtype(index):
        kind = ("a RangeIndex (row numbers) -- did you call reset_index() or pass a list/array?"
                if isinstance(index, pd.RangeIndex) else f"an index of dtype {index.dtype} (row numbers or epoch values?)")
        # A library signal(df) returns its positions "indexed like df" -- row numbers, for a frame
        # from ft.load() -- so handing that straight to report_positions lands here: name `t=`.
        raise ValueError(
            f"{who}: the series must be indexed by the bar timestamp "
            "(e.g. df.set_index(time_col)['pos'] or pd.Series(pos.values, index=df[time_col])), "
            f"or pass the bar times beside the values: ft.{who}(pos, t=df[time_col]); "
            f"got {kind}")
    try:
        return pd.to_datetime(index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{who}: the index is not bar timestamps ({exc}); index the "
                         "series by the time column, e.g. pd.Series(pos.values, index=df[time_col])") from None


# ---------------------------------------------------------------------------------------------
# Task objectives: the rows come from a task server, the result is one action per row
# ---------------------------------------------------------------------------------------------
_TASK = "/task"


def task() -> dict:
    """The task this objective is scored on (task objectives only): its description, the target
    column, what an action means (`action`: kind, min, max, initial, description), the score,
    the columns with their roles and descriptions, and the holdout boundary."""
    try:
        with open(os.path.join(_TASK, "task.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except OSError:
        raise RuntimeError("ft.task(): this objective is not scored by a task server -- use ft.load() "
                           "for its datasets instead") from None


def rows(columns: list[str] | None = None):
    """The task's rows as a pandas DataFrame, one row per time step, sorted by the timestamp
    column `t` (task objectives only). Every column is a signal you may use; the target is named
    in ft.task()["target"]. Decide each row's action from that row and earlier rows ONLY: the
    harness re-runs your code on rows cut at several points and fails it if any earlier action
    changes. `columns` loads only those columns (plus `t`)."""
    import pandas as pd

    path = os.path.join(_TASK, "rows.parquet")
    if not os.path.exists(path):
        raise RuntimeError("ft.rows(): this objective is not scored by a task server -- use ft.load() instead")
    cols = None if columns is None else _unique(["t"] + [c for c in columns if c != "t"])
    df = pd.read_parquet(path, columns=cols)
    return df.sort_values("t", kind="stable").reset_index(drop=True)


def rows_pl(columns: list[str] | None = None):
    """The task's rows as a POLARS DataFrame sorted by `t` -- the same rows as ft.rows(), several
    times faster to load and compute on (700k+ rows). Every ft helper accepts polars frames and
    series as they are; report with ft.report_actions(values, t=rows["t"])."""
    import polars as pl

    path = os.path.join(_TASK, "rows.parquet")
    if not os.path.exists(path):
        raise RuntimeError("ft.rows_pl(): this objective is not scored by a task server -- use ft.load_pl() instead")
    cols = None if columns is None else _unique(["t"] + [c for c in columns if c != "t"])
    return pl.read_parquet(path, columns=cols).sort("t", maintain_order=True)


def _pandas_compat() -> None:
    """Let candidate code written for pandas < 3 keep running: fillna(method="ffill"/"bfill") was
    removed in pandas 3, and models keep writing it. Such a call is sent to .ffill()/.bfill()
    with the same limit/axis/inplace. A no-op on a pandas that still takes `method`."""
    import inspect

    import pandas as pd

    for cls in (pd.Series, pd.DataFrame):
        orig = cls.fillna
        if "method" in inspect.signature(orig).parameters:
            continue

        def fillna(self, value=None, *args, method=None, _orig=orig, **kw):
            if method is None:
                return _orig(self, value, *args, **kw)
            if value is not None:
                raise ValueError("fillna: pass either a value or method, not both")
            fill = {"ffill": self.ffill, "pad": self.ffill, "bfill": self.bfill, "backfill": self.bfill}.get(method)
            if fill is None:
                raise ValueError(f"fillna: unknown method {method!r} (use .ffill() or .bfill())")
            kw.pop("downcast", None)
            return fill(**kw)

        cls.fillna = fillna


def _to_pandas(x):
    """A polars DataFrame/Series as pandas; anything else unchanged."""
    mod = type(x).__module__
    if mod.startswith("polars") and hasattr(x, "to_pandas"):
        return x.to_pandas()
    return x


def _accepts_polars(fn):
    """Let a pandas-based helper take polars frames and series (converted on the way in)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        return fn(*[_to_pandas(a) for a in args], **{k: _to_pandas(v) for k, v in kw.items()})

    return wrapper


def report_actions(actions=None, t=None, *, values=None, positions=None) -> None:
    """Report the strategy's ACTIONS -- one per row (task objectives only).

    `actions` is the action decided at each row (using that row and earlier rows only), which
    takes effect from that row to the next: a pandas Series indexed by the rows' `t` values, or
    any values (polars Series, numpy array, list) with their times in `t`, or a polars/pandas
    DataFrame with the column `t` and one action column. What an action means and its bounds are
    in ft.task()["action"]; the task server scores them. Actions may be sparse -- a row without
    one (or with NaN) keeps the previous action; before the first one the action is the task's
    `initial` value.

        rows = ft.rows_pl()
        ft.report_actions(my_decision(rows), t=rows["t"])      # polars
        ft.report_actions(pd.Series(my_decision(rows), index=rows["t"]))   # pandas

    `values=` / `positions=` are accepted for `actions`. Values with no times at all (an array, a
    list, a series indexed by row number) are taken as one per task row, in ft.rows() order, when
    there are exactly as many of them as rows.
    """
    import numpy as np
    import pandas as pd

    given = [x for x in (actions, values, positions) if x is not None]
    if len(given) != 1:
        raise TypeError("report_actions: pass the actions once -- report_actions(values, t=rows['t'])")
    actions, t = _to_pandas(given[0]), _to_pandas(t)
    if t is None and not isinstance(actions, pd.DataFrame):
        t = _row_times_for(actions)
    if isinstance(actions, pd.DataFrame):
        if "t" not in actions.columns or len(actions.columns) != 2:
            raise ValueError("report_actions: a DataFrame needs the column `t` and exactly one action column")
        val = next(c for c in actions.columns if c != "t")
        actions = pd.Series(actions[val].to_numpy(), index=actions["t"])
    if t is not None:
        vals = actions.to_numpy() if isinstance(actions, pd.Series) else np.asarray(actions)
        if len(vals) != len(t):
            raise ValueError(f"report_actions: {len(vals)} actions but {len(t)} times in `t`")
        actions = pd.Series(vals, index=pd.DatetimeIndex(pd.to_datetime(t)))
    s = actions if isinstance(actions, pd.Series) else pd.Series(actions)
    if len(s) == 0:
        raise ValueError("report_actions got an empty series")
    df = pd.DataFrame({"t": _position_times(s.index, "report_actions"),
                       "pos": pd.to_numeric(pd.Series(s.values), errors="coerce").astype("float64")})
    if getattr(df["t"].dt, "tz", None) is not None:
        df["t"] = df["t"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.dropna(subset=["t", "pos"]).drop_duplicates("t", keep="last").sort_values("t")
    if df.empty:
        raise ValueError("report_actions: no action with both a timestamp and a number")
    lo, hi = df["t"].iloc[0], df["t"].iloc[-1]
    if lo < pd.Timestamp("1990-01-01") or hi > pd.Timestamp("2100-01-01"):
        raise ValueError(f"report_actions: actions are dated {lo} .. {hi}, not the rows' time range. Index the "
                         "series by the rows' t column: pd.Series(values, index=rows['t']).")
    os.makedirs(_FT, exist_ok=True)
    df.to_parquet(os.path.join(_FT, "actions.parquet"), index=False)


def _row_times_for(actions):
    """The task rows' `t`, when `actions` carries no times of its own (an array, a list, a series
    indexed by row number) and has exactly one value per row; None otherwise (the caller's own
    checks then say what is wrong)."""
    import numpy as np
    import pandas as pd

    if isinstance(actions, pd.Series):
        idx = actions.index
        if isinstance(idx, pd.MultiIndex) or not (pd.api.types.is_numeric_dtype(idx) or pd.api.types.is_bool_dtype(idx)):
            return None
        n = len(actions)
    else:
        n = len(np.asarray(actions).reshape(-1))
    path = os.path.join(_TASK, "rows.parquet")
    if not n or not os.path.exists(path):
        return None
    t = pd.read_parquet(path, columns=["t"])["t"].sort_values(kind="stable")
    return t.to_numpy() if len(t) == n else None


def report_positions(positions, t=None) -> None:
    """Report the strategy's POSITIONS -- the preferred way to report a trading strategy.

    `positions` is a pandas Series indexed by bar timestamp -- or any values (polars Series,
    numpy array) with the bar timestamps in `t`: the position (e.g. -1, 0, 0.5, 1) decided at the
    close of that bar using only data up to and including that bar. A series indexed by row
    numbers (e.g. after reset_index()) is refused -- it has no times. The harness holds each
    position until the next reported one and computes the returns itself, from the dataset's
    prices, with trading costs -- so you never compute or report returns yourself.

        ft.report_positions(pos, t=df["t"])                   # polars
    """
    import numpy as np
    import pandas as pd

    if t is not None:
        vals = positions.to_numpy() if hasattr(positions, "to_numpy") else np.asarray(positions)
        if len(vals) != len(t):
            raise ValueError(f"report_positions: {len(vals)} positions but {len(t)} times in `t`")
        positions = pd.Series(vals, index=pd.DatetimeIndex(pd.to_datetime(t)))
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
    if _CAPTURE is not None:
        # Running as a member of another script (candidate_positions): hand the positions over.
        _CAPTURE["positions"] = pd.Series(df["pos"].to_numpy(), index=pd.DatetimeIndex(df["t"]))
        return
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
    if _ROUTED and not os.path.exists(os.path.join(_FT, "regime.parquet")) and len(_ROUTED["labels"]) == len(s):
        # The positions came from ft.route: record which regime (and so which signal) each bar
        # was in, for the equity chart. Timestamps come from the positions' own index.
        _write_regime(pd.DataFrame({"t": _position_times(s.index), "label": _ROUTED["labels"]}),
                      _ROUTED["name"], _ROUTED["routes"])
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


# =======================================================================================
# Fast research: score many variants in one experiment, and test day-direction signals
# =======================================================================================
def _cost_bps(cost_bps):
    if cost_bps is not None:
        return float(cost_bps)
    try:
        return float(task()["valuation"]["cost_bps"])
    except Exception:  # noqa: BLE001 -- not a task objective, or no valuation: the usual 2 bps
        return 2.0


def _sharpe(daily):
    import numpy as np

    d = np.asarray(daily, dtype=float)
    if len(d) < 2 or not np.std(d, ddof=1) > 0:
        return float("nan")
    return float(np.mean(d) / np.std(d, ddof=1) * np.sqrt(252.0))


def quick_score(positions, rows, price: str = "Close", time: str = "t", *, cost_bps: float | None = None,
                delay: int = 1, tz: str = _TZ) -> dict:
    """An APPROXIMATE in-sample score of one position series, in a second -- for comparing ideas
    and parameter variants inside run_python before you spend a submission. Same rules as the
    task server: the position decided at a row is filled `delay` bars later, pays cost_bps per unit
    of position change (the task's own cost by default), and is closed at each session's last
    row. The server's number differs a little (its fills and daily accounting are exact): use
    this to RANK variants, and the submission for the real score.

    Returns {sharpe, sharpe_gross, sharpe_flipped, sharpe_h1, sharpe_h2, worst_half, bps_per_day,
    trades_per_day, long_share, short_share, active_days, days, floors_ok} -- h1/h2 are the first and
    second half of the sessions: an idea worth keeping is positive in BOTH (worst_half > 0)."""
    import numpy as np

    pos = np.nan_to_num(np.asarray(_to_pandas(positions), dtype=float), nan=0.0)
    p = _col(_to_pandas(rows), price)
    if len(pos) != len(p):
        raise ValueError(f"quick_score: {len(pos)} positions for {len(p)} rows -- one per row")
    session, _ = clock(_to_pandas(rows)[time], tz)
    starts = _session_starts(session)
    ends = np.r_[starts[1:], len(p)]
    cost = _cost_bps(cost_bps) / 1e4
    daily, gross_d, flip_d = [], [], []
    trades = longs = shorts = active = 0
    for a, b in zip(starts, ends):
        h = np.zeros(b - a)
        if b - a > delay:
            h[delay:] = pos[a:b - delay]
        h[-1] = 0.0                                       # closed at the session's last row
        r = np.zeros(b - a)
        with np.errstate(all="ignore"):
            r[:-1] = p[a + 1:b] / p[a:b - 1] - 1.0
        r = np.nan_to_num(r)
        turn = np.abs(np.diff(np.r_[0.0, h]))
        gross = float(np.sum(h * r))
        daily.append(gross - cost * float(turn.sum()))
        gross_d.append(gross)
        flip_d.append(-gross - cost * float(turn.sum()))
        prev = np.r_[0.0, h[:-1]]
        opened = (h != 0) & (np.sign(h) != np.sign(prev))
        trades += int(opened.sum())
        longs += int((opened & (h > 0)).sum())
        shorts += int((opened & (h < 0)).sum())
        active += int(np.any(h != 0))
    n = len(daily)
    half = n // 2
    tpd = trades / n if n else 0.0
    ls, ss = (longs / trades, shorts / trades) if trades else (0.0, 0.0)
    h1, h2 = _sharpe(daily[:half]), _sharpe(daily[half:])
    return {"sharpe": _sharpe(daily), "sharpe_gross": _sharpe(gross_d), "sharpe_flipped": _sharpe(flip_d),
            "sharpe_h1": h1, "sharpe_h2": h2,
            "worst_half": min(h1, h2) if math.isfinite(h1) and math.isfinite(h2) else float("nan"),
            "bps_per_day": float(np.mean(daily) * 1e4) if n else float("nan"), "trades_per_day": tpd,
            "long_share": ls, "short_share": ss, "active_days": active, "days": n,
            "floors_ok": bool(tpd >= 2.0 and min(ls, ss) >= 0.2)}


def sweep(make_positions, grid: dict, rows, *, max_variants: int = 48, time_budget_s: float = 200.0, **score_kw):
    """Score every combination of `grid` in ONE experiment: `make_positions(**params)` returns one
    position per row for those parameters; each variant is scored with ft.quick_score. Returns a
    DataFrame (polars when available) best first -- by `worst_half` (the weaker of the two halves of
    the in-sample sessions), so a variant that only works in one half does not come out on top.

        def make(z_in, stop):
            entries = ...                      # +1/-1 when your signal fires
            return ft.trend_exits(entries, rows, stop_mult=stop)
        table = ft.sweep(make, {"z_in": [1.0, 1.5, 2.0], "stop": [1.5, 3.0]}, rows)
        print(table.head(10))

    Pick a variant that is good in BOTH halves and whose neighbours are good too (a lone spike
    among its neighbours is noise), then submit it. `floors_ok` says whether it meets the trade
    floor and the 20% per side. At most `max_variants`; stops after `time_budget_s` and scores
    what it has."""
    import itertools
    import time as _time

    keys = list(grid)
    combos = list(itertools.product(*(list(grid[k]) for k in keys)))
    if len(combos) > max_variants:
        raise ValueError(f"sweep: {len(combos)} variants -- at most {max_variants}; use fewer values per parameter")
    t0, out = _time.time(), []
    for combo in combos:
        if _time.time() - t0 > time_budget_s:
            print(f"sweep: time budget reached after {len(out)} of {len(combos)} variants")
            break
        params = dict(zip(keys, combo))
        try:
            out.append({**params, **quick_score(make_positions(**params), rows, **score_kw), "error": ""})
        except Exception as exc:  # noqa: BLE001 -- one bad variant must not lose the others
            out.append({**params, "error": f"{type(exc).__name__}: {exc}"[:200]})
    return _table(out, "worst_half")


def _table(records: list[dict], by: str):
    import math as _m

    records = sorted(records, key=lambda r: -(r.get(by) if isinstance(r.get(by), float) and not _m.isnan(r.get(by))
                                              else -1e18))
    try:
        import polars as pl

        return pl.DataFrame(records, infer_schema_length=None)
    except ImportError:
        import pandas as pd

        return pd.DataFrame(records)


def direction_scan(rows, fields=None, times=("10:00", "10:30", "11:00", "11:30", "12:00"), exit: str = "15:55",
                   price: str = "Close", time: str = "t", *, cost_bps: float | None = None, min_sessions: int = 20,
                   tz: str = _TZ):
    """RESEARCH ONLY (reads the rest of each day on purpose): does a field, read at a decision time,
    tell the DIRECTION of the rest of the day? That is where this data's money is: the swarm's big
    winners were almost all trades pointing with the day's move, and costs sank everything that
    traded minute-scale noise.

    For each field and decision time, per session: the field's value at the first row at/after the
    time (as served -- already delayed), turned into a direction causally (its sign against the mean
    of the PREVIOUS sessions' values at that time; for the built-in price features, their own sign),
    and the return from the next bar to `exit` (default 15:55 New York). Built-in features:
    ret_since_open, gap (open vs the previous close), vwap_dist (price vs the session VWAP so far).

    Returns a DataFrame best first by `worst_half_bps`, one row per (field, time):
      days, ic (rank correlation of value and rest-of-day return), hit (share of days the direction
      was right), follow_bps (average bps a day trading that direction, after a round trip of costs),
      fade_bps (trading against it), follow_h1/h2 and fade_h1/h2 (each half of the sessions),
      best ('follow' or 'fade') and worst_half_bps (the best side's weaker half).

    Only a row positive in BOTH halves is a lead -- and with ~140 fields x 5 times, chance alone puts
    a few dozen rows in both halves at |t_stat| near 2 (on 09-30, 38 of 667 did): treat t_stat >= 3,
    a neighbouring time agreeing, and a reason it should work as the bar. To trade it: at that time, enter the direction
    (or its opposite for 'fade'), hold with a wide trailing stop (ft.trend_exits) to the clock exit;
    two or more decision times a day meet the trade floor. Build and check the strategy with
    ft.quick_score / ft.sweep before submitting."""
    import numpy as np

    df = _to_pandas(rows)
    p = _col(df, price)
    session, minute = clock(df[time], tz)
    starts = _session_starts(session)
    ends = np.r_[starts[1:], len(p)]
    cost = _cost_bps(cost_bps)
    tx = sorted(_hhmm(s) for s in times)
    ex = _hhmm(exit)
    vol = np.asarray(df["Volume"], dtype=float) if "Volume" in df.columns else None
    skip = {time, "Open", "High", "Low", "Close", price}
    if fields is None:
        fields = [c for c in df.columns if c not in skip and str(df[c].dtype) not in ("object", "string")
                  and not str(df[c].dtype).startswith(("datetime", "bool"))]
    cols = {f: np.asarray(df[f], dtype=float) for f in fields if f in df.columns}
    builtins = ["ret_since_open", "gap", "vwap_dist"]
    feats = {name: {x: [] for x in tx} for name in list(cols) + builtins}
    rets = {x: [] for x in tx}
    prev_close = np.nan
    for a, b in zip(starts, ends):
        m = minute[a:b]
        e = a + int(np.searchsorted(m, ex, side="left"))
        e = min(e, b - 1)
        for x in tx:
            i = a + int(np.searchsorted(m, x, side="left"))
            ok = i + 1 < e
            with np.errstate(all="ignore"):
                rets[x].append((p[e] / p[i + 1] - 1.0) * 1e4 if ok else np.nan)
                for f, v in cols.items():
                    feats[f][x].append(v[i] if ok else np.nan)
                feats["ret_since_open"][x].append(p[i] / p[a] - 1.0 if ok else np.nan)
                feats["gap"][x].append(p[a] / prev_close - 1.0 if ok else np.nan)
                if vol is not None and ok:
                    w = np.nan_to_num(vol[a:i + 1])
                    vw = np.sum(p[a:i + 1] * w) / np.sum(w) if np.sum(w) > 0 else np.nan
                    feats["vwap_dist"][x].append(p[i] / vw - 1.0)
                else:
                    feats["vwap_dist"][x].append(np.nan)
        prev_close = p[b - 1]
    out = []
    for f, by_t in feats.items():
        for x in tx:
            v, r = np.asarray(by_t[x], dtype=float), np.asarray(rets[x], dtype=float)
            if f in builtins:
                d = np.sign(v)
            else:                                   # against the previous sessions' mean: causal
                past = np.r_[np.nan, np.nancumsum(v)[:-1] / np.maximum(np.cumsum(np.isfinite(v))[:-1], 1)]
                past[:min_sessions] = np.nan
                d = np.sign(v - past)
            keep = np.isfinite(d) & np.isfinite(r) & (d != 0)
            if keep.sum() < 2 * min_sessions:
                continue
            dd, rr, vv = d[keep], r[keep], v[keep]
            pnl = dd * rr
            half = len(pnl) // 2
            rk = lambda z: np.argsort(np.argsort(z)).astype(float)  # noqa: E731
            ic = float(np.corrcoef(rk(vv), rk(rr))[0, 1]) if np.std(vv) > 0 else float("nan")
            fol = [float(np.mean(s)) - 2 * cost for s in (pnl, pnl[:half], pnl[half:])]
            fad = [float(np.mean(-s)) - 2 * cost for s in (pnl, pnl[:half], pnl[half:])]
            best = "follow" if min(fol[1:]) >= min(fad[1:]) else "fade"
            side = fol if best == "follow" else fad
            out.append({"field": f, "time": f"{int(x // 60):02d}:{int(x % 60):02d}", "days": int(keep.sum()),
                        "ic": ic, "hit": float(np.mean(pnl > 0)), "follow_bps": fol[0], "fade_bps": fad[0],
                        "follow_h1": fol[1], "follow_h2": fol[2], "fade_h1": fad[1], "fade_h2": fad[2],
                        "best": best, "worst_half_bps": min(side[1:]),
                        "t_stat": float(np.mean(pnl) / (np.std(pnl, ddof=1) / np.sqrt(len(pnl))))
                        if np.std(pnl) > 0 else float("nan")})
    return _table(out, "worst_half_bps")


# Every data helper takes polars frames and series too (ft.rows_pl(), ft.load_pl()): they are
# converted to pandas on the way in, so a polars script calls them as they are.
for _name in ("resample", "align", "route", "regime_grid", "regimes", "report_regime", "inverse_vol", "size",
              "clock", "session_vwap", "gamma_regime", "trend_exits", "noise_area_breakout", "decision_points",
              "label_outcomes", "meta_filter", "admit", "report_positions", "report_returns"):
    globals()[_name] = _accepts_polars(globals()[_name])
