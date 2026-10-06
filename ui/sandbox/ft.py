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


def _polars_compat() -> None:
    """Positional arguments written from pandas habit land in the wrong polars slot: the second
    argument of rolling_quantile is `interpolation` (rolling_quantile(0.9, 20) -> "'int' object is
    not an instance of 'str' while processing 'interpolation'", 10-01 12:20), of rolling_mean & co.
    `weights` (rolling_mean(60, 30), pandas' rolling(60, min_periods=30)). A whole number can mean
    neither, so it is taken as the window / min_samples it was meant as."""
    import functools

    try:
        import polars as pl
    except ImportError:
        return
    for cls in (pl.Expr, pl.Series):
        orig = getattr(cls, "rolling_quantile", None)
        if orig is not None and not getattr(orig, "_ft_compat", False):
            def rolling_quantile(self, quantile, interpolation="nearest", *args, _orig=orig, **kw):
                if isinstance(interpolation, int) and not isinstance(interpolation, bool):
                    if args or "window_size" in kw:
                        raise TypeError(f"rolling_quantile: interpolation={interpolation!r} -- pass the window "
                                        "as window_size=..., the method as interpolation='nearest'/'linear'/...")
                    interpolation, args = "nearest", (interpolation,)
                return _orig(self, quantile, interpolation, *args, **kw)

            functools.update_wrapper(rolling_quantile, orig)    # help() / signature show polars' own
            rolling_quantile._ft_compat = True
            cls.rolling_quantile = rolling_quantile
        for name in ("rolling_mean", "rolling_std", "rolling_var", "rolling_sum", "rolling_min", "rolling_max",
                     "rolling_median"):
            orig = getattr(cls, name, None)
            if orig is None or getattr(orig, "_ft_compat", False):
                continue

            def rolling(self, window_size, weights=None, *args, _orig=orig, **kw):
                if isinstance(weights, int) and not isinstance(weights, bool) and "min_samples" not in kw:
                    weights, kw = None, {**kw, "min_samples": weights}
                return _orig(self, window_size, weights, *args, **kw)

            functools.update_wrapper(rolling, orig)
            rolling._ft_compat = True
            setattr(cls, name, rolling)
    # df.select(pl.col('GEX').describe()) -- the pandas habit, 7 runs on 10-01 alone, and the hint did not
    # stop it. describe is a frame/Series method in polars; asked for in a select, it is given as meant:
    # the frame's describe() table over those columns.
    if not hasattr(pl.Expr, "describe"):
        pl.Expr.describe = lambda self, percentiles=(0.25, 0.5, 0.75), interpolation="nearest": \
            _Describe(self, percentiles, interpolation)
        orig = pl.DataFrame.select

        @functools.wraps(orig)
        def select(self, *exprs, _orig=orig, **named):
            flat = [e for x in exprs for e in (x if isinstance(x, (list, tuple)) else (x,))]
            if flat and not named and all(isinstance(e, _Describe) for e in flat):
                return _orig(self, [e.expr for e in flat]).describe(
                    percentiles=flat[0].percentiles, interpolation=flat[0].interpolation)
            if any(isinstance(e, _Describe) for e in flat):
                raise TypeError("pl.col(...).describe() works only on its own in df.select(...) -- "
                                "print(df.select('a', 'b').describe()) for the stats table, or "
                                "pl.col('a').mean() / .std() / .quantile(0.5) for single numbers")
            return _orig(self, *exprs, **named)

        pl.DataFrame.select = select
    # pl.col('minute').where(cond).forward_fill() -- pandas' where (keep the value where cond holds,
    # null elsewhere). polars' deprecated where is filter: the column shrinks and with_columns dies
    # with "can't broadcast Series 'minute' of length 55826 to length 496482" (Qwen3.6, 10-04 16:50).
    # A filtered column could only ever be used in an aggregate, where nulls are skipped the same way.
    orig = getattr(pl.Expr, "where", None)
    if orig is not None and not getattr(orig, "_ft_compat", False):
        def where(self, cond, other=None):
            return pl.when(cond).then(self).otherwise(other if isinstance(other, pl.Expr) else pl.lit(other))

        where._ft_compat = True
        pl.Expr.where = where


class _Describe:
    """pl.col(...).describe(): not an expression -- a request for df.select(...)'s describe table."""

    def __init__(self, expr, percentiles, interpolation):
        self.expr, self.percentiles, self.interpolation = expr, percentiles, interpolation

    def __repr__(self):
        return f"<{self.expr}.describe() -- use it as df.select(pl.col(...).describe())>"


# ---------------------------------------------------------------------------------------
# One library's habits on the other's objects. The work archive of 09-30/10-01 counted each of
# these again and again, every one after its hint was already in place: the models write what
# they remember, so what they mean is done, with a one-line note on stderr saying what was done.
# ---------------------------------------------------------------------------------------
_TIME_COLUMNS = ("t", "SlotUtc", "SlotEt")         # the datetime columns of every dataset and the rows
_SESSION_COLUMNS = ("session", "date", "day")      # names models give the session date and expect to exist
_FRAME_SYNONYMS = {"sort_by": "sort", "order_by": "sort"}   # an expression method's name used on a frame
_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume", "t", "SlotUtc")   # replaced only by mistake, as a rule
_NOTED: set[str] = set()


def _note(key: str, text: str) -> None:
    if key not in _NOTED:
        _NOTED.add(key)
        import sys

        print(f"[ft] {text}", file=sys.stderr)


_RECENT_FRAMES: list = []   # weak references to the polars frames the script made last, newest first


def _remember(df):
    import weakref

    try:
        ref = weakref.ref(df)
    except TypeError:
        return df
    _RECENT_FRAMES[:] = [ref] + [r for r in _RECENT_FRAMES if r() is not None and r() is not df][:7]
    return df


def _mixup_compat() -> None:
    import functools
    import inspect
    import re

    try:
        import pandas as pd
        import polars as pl
    except ImportError:
        return
    if getattr(pl.DataFrame, "_ft_mixups", False):
        return
    pl.DataFrame._ft_mixups = True

    # pl.col('t').str.to_datetime(...) / .str.strptime(...) / df['SlotUtc'].str.to_date() on a column that
    # IS a datetime -- "SchemaError: expected `String`, got `datetime[ns]`", 11 runs (bug #134).
    for cls in (pl.Expr, pl.Series):
        prop = cls.__dict__.get("str")
        if not isinstance(prop, property):
            continue

        def str_(self, _get=prop.fget):
            ns = _get(self)
            try:
                if isinstance(self, pl.Series):
                    temporal = self.dtype.is_temporal()
                else:
                    temporal = self.meta.is_column() and self.meta.output_name() in _TIME_COLUMNS
            except Exception:
                temporal = False
            return _DatetimeText(self, ns) if temporal else ns

        setattr(cls, "str", property(str_, doc=prop.__doc__))

    # A pandas method on a polars frame/Series (df.sort_values, df.copy(), s.corr(other), s.values, ...):
    # run it on .to_pandas() -- the result is pandas, which is what the rest of such a script expects.
    def pandas_fallback(kind, pd_cls):
        def __getattr__(self, name):
            last = self.__dict__.get("_ft_last") if name == "alias" else None
            if last is not None:
                # rows.with_columns(<expr>).alias('fwd30') -- the alias meant for the expression (10-01 14:07;
                # the unnamed expression had silently overwritten Close). Done as meant, on the frame before.
                base, expr = last
                _note("frame.alias", ".alias(...) was written after with_columns(...) instead of on the expression -- "
                                     "taken as with_columns(expr.alias(...)); write it that way")
                return lambda alias_name: base.with_columns(expr.alias(alias_name))
            if name in _FRAME_SYNONYMS and kind == "DataFrame":
                # df.sort_by('SlotUtc') -- the expression method's name for the frame's .sort (10-01 15:01)
                real = _FRAME_SYNONYMS[name]
                _note(f"frame.{name}", f".{name} is the expression method -- the frame's is .{real}; used that")
                return getattr(self, real)
            if name.startswith("_") or not hasattr(pd_cls, name):
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            if name == "copy":                      # a copy, still polars
                return lambda *a, **k: self.clone()
            _note(f"pd.{name}", f".{name} is a pandas method and this is a polars {kind} (ft.load_pl / ft.rows_pl): "
                                f"it ran on .to_pandas() and gave pandas. ft.load / ft.rows give pandas from the start")
            attr = getattr(self.to_pandas(), name)
            if not callable(attr):
                return attr
            return lambda *a, **k: attr(*[_to_pandas(v) for v in a], **{n: _to_pandas(v) for n, v in k.items()})

        return __getattr__

    pl.DataFrame.__getattr__ = pandas_fallback("DataFrame", pd.DataFrame)
    pl.Series.__getattr__ = pandas_fallback("Series", pd.Series)

    # s.to_numpy().corr(other) / .abs() / .nunique() / .values -- a pandas method on the numpy array a
    # Series turned into (10-01 14:44, and five other names before). The array is an ndarray in every way;
    # only a name numpy lacks falls back to the pandas Series method.
    for cls in (pl.Series, pd.Series):
        orig = cls.to_numpy

        def to_numpy(self, *a, _orig=orig, **k):
            # s.to_numpy(dtype=float) on a polars Series -- pandas' keyword; polars has none (10-01 20:27)
            dtype = k.pop("dtype", None) if "dtype" not in inspect.signature(_orig).parameters else None
            out = _orig(self, *a, **k)
            if dtype is not None:
                out = _np.asarray(out, dtype=dtype)
            if type(out) is not _np.ndarray or out.ndim != 1:
                return out
            return out.view(_TimeArray if out.dtype.kind == "M" else _Array)

        functools.update_wrapper(to_numpy, orig)
        cls.to_numpy = to_numpy
    # The same through pandas' .values: df['SkewRR_Value'].astype(float).values.rolling(2000) and
    # rows['t'].values.astype('int64').values (10-01 replay, 3 submissions -- AttributeError on 'numpy.ndarray').
    values = pd.Series.__dict__.get("values")
    if isinstance(values, property):
        def series_values(self, _get=values.fget):
            out = _get(self)
            return out.view(_Array) if type(out) is _np.ndarray and out.ndim == 1 else out

        pd.Series.values = property(series_values, doc=values.__doc__)
    # pos[pos != 0].index.date.nunique() -- a DatetimeIndex's dates are a plain object array (10-01 replay, 2 runs).
    dates = pd.DatetimeIndex.__dict__.get("date")
    if isinstance(dates, property):
        def index_dates(self, _get=dates.fget):
            out = _get(self)
            return out.view(_Array) if type(out) is _np.ndarray and out.ndim == 1 else out

        pd.DatetimeIndex.date = property(index_dates, doc=dates.__doc__)
    # q1, q2 = s.quantile([0.33, 0.66]).to_list() -- polars gives several quantiles as a plain list (10-01 replay,
    # 3 runs: "'list' object has no attribute 'to_list'"); it takes .to_list() / .tolist() now.
    orig_quantile = pl.Series.quantile

    @functools.wraps(orig_quantile)
    def series_quantile(self, *a, _orig=orig_quantile, **k):
        out = _orig(self, *a, **k)
        return _Columns(out) if type(out) is list else out

    pl.Series.quantile = series_quantile

    # df.columns.tolist() on a polars frame, whose columns are a plain list (10-01 14:17).
    cols = pl.DataFrame.__dict__["columns"]
    pl.DataFrame.columns = property(lambda self, _get=cols.fget: _Columns(_get(self)), cols.fset, cols.fdel, cols.__doc__)

    # The other way round: a polars method on a pandas frame/Series -- bars = ft.resample(...) (pandas)
    # then bars.sort('SlotUtc') (10-01 14:16). Run on pl.from_pandas(...), the result is polars.
    def polars_fallback(kind, pl_cls, pd_cls):
        orig = pd_cls.__getattr__

        def __getattr__(self, name, _orig=orig):
            try:
                return _orig(self, name)
            except AttributeError:
                if name.startswith("_") or not hasattr(pl_cls, name) or hasattr(pd_cls, name):
                    raise
            _note(f"pl.{name}", f".{name} is a polars method and this is a pandas {kind} (ft.load / ft.rows and every "
                                f"ft helper that returns a frame give pandas): it ran on pl.from_pandas(...) and gave polars")
            index = not isinstance(self.index, pd.RangeIndex)
            conv = pl.from_pandas(self, include_index=index) if kind == "DataFrame" else pl.from_pandas(self)
            attr = getattr(conv, name)
            if not callable(attr):
                return attr
            return lambda *a, **k: attr(*a, **k)

        functools.update_wrapper(__getattr__, orig)
        pd_cls.__getattr__ = __getattr__

    polars_fallback("DataFrame", pl.DataFrame, pd.DataFrame)
    polars_fallback("Series", pl.Series, pd.Series)

    # bars['Imb_OINet_D0'] after an .agg({...}) that did not keep it: pandas says only KeyError: 'Imb_OINet_D0'
    # (10-01 15:05). Name the frame's nearest columns, as the polars side now does.
    orig_gi = pd.DataFrame.__getitem__

    @functools.wraps(orig_gi)
    def __getitem__(self, key, _orig=orig_gi):
        try:
            return _orig(self, key)
        except KeyError:
            if not isinstance(key, str):
                raise
            import difflib

            cols = [str(c) for c in self.columns]
            near = difflib.get_close_matches(key, cols, n=3, cutoff=0.5)
            raise KeyError(f"{key!r} -- the frame has no such column"
                           + (f"; did you mean {', '.join(map(repr, near))}?" if near else "")
                           + f" It has: {', '.join(cols[:40])}{', ...' if len(cols) > 40 else ''}") from None

    pd.DataFrame.__getitem__ = __getitem__

    # pl.col('x').mean().over('session') / group_by('date') on rows that have no such column ("unable to
    # find column \"session\"", 10-01 14:07; "day" twice before): the session date is added as asked for.
    orig_wc = pl.DataFrame.with_columns

    def flat(exprs, named):
        return [e for x in exprs for e in (x if isinstance(x, (list, tuple)) else (x,))] + list(named.values())

    def with_session(df, items):
        want = set()
        for e in items:
            if isinstance(e, str):
                want.add(e)
            elif isinstance(e, pl.Expr):
                try:
                    want.update(e.meta.root_names())
                except Exception:
                    pass
        missing = [n for n in _SESSION_COLUMNS if n in want and n not in df.columns]
        time = next((c for c in _TIME_COLUMNS if c in df.columns and df.schema[c] == pl.Datetime), None)
        if not missing or time is None:
            return df
        t = pl.col(time)
        local = (t.dt.convert_time_zone(_TZ) if getattr(df.schema[time], "time_zone", None)
                 else t.dt.replace_time_zone("UTC").dt.convert_time_zone(_TZ)).dt.date()
        _note(f"session.{'.'.join(missing)}", f"the frame had no {', '.join(repr(m) for m in missing)} column -- added as the "
                                              f"New York session date of {time!r} (what ft.clock gives)")
        return orig_wc(df, *[local.alias(m) for m in missing])

    def as_column(v, name=None):
        if isinstance(v, (pd.Series, pd.Index)) or (isinstance(v, np.ndarray) and v.ndim == 1):
            s = pl.Series(name or getattr(v, "name", None) or "", np.asarray(v))
            return s.alias(name) if name else s
        if isinstance(v, (list, tuple)):
            return type(v)(as_column(x) for x in v)
        return v

    import numpy as np

    @functools.wraps(orig_wc)
    def with_columns(self, *exprs, **named):
        # rows.with_columns(vwap=ft.session_vwap(rows)) -- a pandas Series (or a numpy array) as a column:
        # "cannot create expression literal for value of type Series" (10-01 16:37). Made a polars column.
        exprs = tuple(as_column(e) for e in exprs)
        named = {k: as_column(v, k) for k, v in named.items()}
        items = flat(exprs, named)
        base = with_session(self, items)
        try:
            out = orig_wc(base, *exprs, **named)
        except pl.exceptions.ColumnNotFoundError as exc:
            # with_columns(a.alias('x'), (pl.col('x') != ...).alias('y')) -- one call's expressions all see the
            # frame as it was, so 'x' does not exist yet (10-01 14:31, twice). Made one after another, as written.
            m = re.search(r'unable to find column "([^"]+)"', str(exc))
            # names the call CREATES -- not pl.col('x') * 2, which keeps the name of the column it reads
            made = [e.meta.output_name() for e in flat(exprs, {}) if isinstance(e, pl.Expr)
                    and e.meta.output_name() not in e.meta.root_names()] + list(named)
            if not m or m.group(1) not in made:
                raise
            _note("with_columns.sequential", f"one with_columns(...) used {m.group(1)!r}, which the same call creates -- "
                                             "polars evaluates them side by side, so they were run one after another; "
                                             "chain with_columns calls for that")
            out = base
            for e in flat(exprs, {}):
                out = orig_wc(out, e)
            if named:
                out = orig_wc(out, **named)
        if len(items) == 1 and not named and isinstance(items[0], pl.Expr):
            out._ft_last = (base, items[0])
        _remember(out)
        # with_columns((pl.col('Close').shift(-h) - pl.col('Close')) / pl.col('Close')) -- no .alias, so the result
        # REPLACES Close and every later use reads returns as prices (10-01 15:25, in a loop). Allowed, but said.
        for e in flat(exprs, {}):
            if isinstance(e, pl.Expr) and not e.meta.is_column():
                try:
                    name = e.meta.output_name()
                except Exception:
                    continue
                if name in _PRICE_COLUMNS and name in base.columns:
                    _note(f"replaced.{name}", f"with_columns(...) REPLACED the {name!r} column with a computed value (the "
                                              f"expression has no .alias) -- every later use of {name!r} reads that; "
                                              f"add .alias('new_name') if you meant a new column")
        return out

    pl.DataFrame.with_columns = with_columns
    for name in ("select", "filter", "group_by", "sort"):
        orig = getattr(pl.DataFrame, name)

        def call(self, *args, _orig=orig, **kw):
            return _orig(with_session(self, flat(args, kw)), *args, **kw)

        functools.update_wrapper(call, orig)
        setattr(pl.DataFrame, name, call)

    # A column the frame lacks: polars often says only '"ret_60f" not found' -- no list, no near name
    # (10-01 15:02: the script had made ret_60bps). Say which of the frame's columns are closest.
    def names_on_miss(fn):
        @functools.wraps(fn)
        def call(self, *args, _fn=fn, **kw):
            try:
                return _fn(self, *args, **kw)
            except pl.exceptions.ColumnNotFoundError as exc:
                import difflib

                m = re.search(r'"([^"]+)"', str(exc))
                cols = list(dict.__iter__(self.schema)) if hasattr(self, "schema") else []
                near = difflib.get_close_matches(m.group(1), cols, n=3, cutoff=0.5) if m else []
                if not m or "did you mean" in str(exc) or "The frame has:" in str(exc):  # added already (nested calls)
                    raise
                more = (f" -- did you mean {', '.join(map(repr, near))}?" if near else "") + \
                       f" The frame has: {', '.join(cols[:40])}{', ...' if len(cols) > 40 else ''}"
                raise pl.exceptions.ColumnNotFoundError(f"{str(exc).splitlines()[0]}{more}") from None

        return call

    for name in ("with_columns", "select", "filter", "group_by", "sort", "drop_nulls", "drop", "unique", "join",
                 "pivot", "unpivot", "rename", "fill_null", "get_column", "__getitem__"):
        if hasattr(pl.DataFrame, name):
            setattr(pl.DataFrame, name, names_on_miss(getattr(pl.DataFrame, name)))

    # rows[pl.col('Close_30fwd').is_not_null()] -- pandas' boolean-mask indexing on a polars frame ("cannot select
    # columns using key of type 'Expr'", 10-01 replay). A boolean expression in [] is the filter it means; so is a
    # boolean Series with one value per ROW (polars reads a boolean Series as a mask over the columns, and refuses
    # one of any other length -- that case alone is taken as rows).
    orig_getitem = pl.DataFrame.__getitem__

    @functools.wraps(orig_getitem)
    def frame_getitem(self, key, _orig=orig_getitem):
        # df.with_columns(pl.when(...).then(1).otherwise(0))['entry'] -- the name asked for here is the .alias the
        # expression never got, so polars called it 'literal' and 'entry' was "not found" (10-05 21:3x).
        last = self.__dict__.get("_ft_last") if isinstance(key, str) and key not in self.columns else None
        if last is not None:
            try:
                made = last[1].meta.output_name()
                unnamed = last[1].meta.undo_aliases().meta.output_name() == made
            except Exception:
                unnamed = False
            if unnamed and made in self.columns:
                _note("frame.unaliased", f"with_columns(...)[{key!r}]: the expression had no .alias, so polars named it "
                                         f"{made!r} -- taken as {key!r}; write with_columns(expr.alias({key!r}))")
                return self.get_column(made).alias(key)
        mask = None
        if isinstance(key, pl.Expr):
            try:
                mask = self.select(key.alias("_ft_mask")).to_series()
            except Exception:
                mask = None
        elif isinstance(key, pl.Series) and len(key) != self.width:
            mask = key
        if mask is not None and mask.dtype == pl.Boolean and len(mask) == self.height:
            _note("frame[mask]", "frame[<boolean mask>] on a polars frame -- taken as frame.filter(mask), which is "
                                 "how polars writes it")
            return self.filter(mask)
        return _orig(self, key)

    pl.DataFrame.__getitem__ = frame_getitem

    # df.select(pl.col('x').quantile(0.1), pl.col('x').quantile(0.9)) / .select(pl.col('len').mean(), pl.col('len').min())
    # -- "DuplicateError: projections contained duplicate output name" (10-01 replay, 3 runs): two numbers asked for
    # from one column both keep its name. Each is named after what it computes (x_quantile_0.1, len_mean, len_min);
    # a plain pl.col('x') among them keeps 'x'.
    def output_name(e):
        try:
            return e.meta.output_name() if isinstance(e, pl.Expr) else None
        except Exception:
            return None

    def named_by_alias(e, name):                 # .alias('GEX') twice is a choice -- left to fail
        try:
            return e.meta.undo_aliases().meta.output_name() != name
        except Exception:
            return True

    def distinct_names(items):
        names = [output_name(e) for e in items]
        taken = {n for n in names if n is not None}
        out, renamed = [], []
        for e, name in zip(items, names):
            if name is not None and names.count(name) > 1 and not e.meta.is_column() and not named_by_alias(e, name):
                m = re.search(r"\.(\w+)\((?:\[dyn \w+: ([^\]]*)\])?[^()]*\)$", str(e))
                base = (f"{name}_{m.group(1)}" + (f"_{m.group(2)}" if m.group(2) else "")) if m else f"{name}_{len(out)}"
                new, k = base, 2
                while new in taken:
                    new, k = f"{base}_{k}", k + 1
                taken.add(new)
                e = e.alias(new)
                renamed.append(new)
            out.append(e)
        return out, renamed

    def dedupe(fn):
        @functools.wraps(fn)
        def call(self, *exprs, _fn=fn, **named):
            try:
                return _fn(self, *exprs, **named)
            except pl.exceptions.DuplicateError:
                fixed, renamed = distinct_names(flat(exprs, {}))
                if not renamed:
                    raise
                _note("select.duplicate", f"several expressions kept the same column name -- they were named "
                                          f"{', '.join(map(repr, renamed))}; give each its own .alias(...)")
                return _fn(self, *fixed, **named)

        return call

    pl.DataFrame.select = dedupe(pl.DataFrame.select)
    group_by_cls = type(pl.DataFrame({"a": [1]}).group_by("a"))
    group_by_cls.agg = dedupe(group_by_cls.agg)

    # df['x'] = values on a polars frame -- "DataFrame object does not support `Series` assignment by index".
    orig_set = pl.DataFrame.__setitem__

    @functools.wraps(orig_set)
    def __setitem__(self, key, value, _orig=orig_set):
        if not isinstance(key, str):
            return _orig(self, key, value)
        if isinstance(value, pl.Expr):
            col = value.alias(key)
        elif isinstance(value, pl.Series):
            col = value.alias(key)
        elif hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            import numpy as np

            col = pl.Series(key, np.asarray(_to_pandas(value)))
        else:
            col = pl.lit(value).alias(key)
        self._df = self.with_columns(col)._df

    pl.DataFrame.__setitem__ = __setitem__

    # s.rolling(60).apply(lambda x: (x - x.mean()) / x.std()) -- the function returns the whole window's z, and
    # pandas wants one number per window ("must be real number, not Series", 10-01 15:27). The window's LAST value
    # is the current row's -- what such a function means, and causal.
    from pandas.core.window.rolling import Rolling

    orig_apply = Rolling.apply

    @functools.wraps(orig_apply)
    def rolling_apply(self, func, *a, _orig=orig_apply, **k):
        if k.get("engine") == "numba":
            return _orig(self, func, *a, **k)

        def one(x, *fa, **fk):
            v = func(x, *fa, **fk)
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                _note("rolling.apply.last", "the rolling.apply function returned the whole window, not one number -- "
                                            "its LAST value (the current row's) was used")
                v = np.asarray(v, dtype=float)
                return v[-1] if len(v) else np.nan
            return v

        return _orig(self, one, *a, **k)

    import numpy as np

    Rolling.apply = rolling_apply

    # df.groupby('date').apply(f) -- pandas 3 hands f each group WITHOUT the grouping columns (pandas 2 kept them),
    # so the result has no 'date' and the next line dies: "['date'] not in index" (10-01 17:21). Models write pandas
    # 2: the group gets its key columns back (from g.name), as it did there.
    from pandas.core.groupby.generic import DataFrameGroupBy

    orig_gapply = DataFrameGroupBy.apply

    @functools.wraps(orig_gapply)
    def group_apply(self, func, *args, _orig=orig_gapply, **kw):
        keys = self.keys if isinstance(self.keys, list) else [self.keys]
        if not callable(func) or "include_groups" in kw or not all(isinstance(k, str) for k in keys):
            return _orig(self, func, *args, **kw)

        def with_keys(g, *a, **k):
            missing = [c for c in keys if c not in g.columns]
            name = getattr(g, "name", None)              # read before .copy(), which drops it
            if missing and name is not None:
                g = g.copy()
                g.name = name
                vals = name if isinstance(name, tuple) else (name,)
                for c, v in zip(keys, vals):
                    if c in missing:
                        g[c] = v
            return func(g, *a, **k)

        return _orig(self, with_keys, *args, **kw)

    DataFrameGroupBy.apply = group_apply

    # bars.loc[mask, 'r'] with mask = (pd.qcut(bars['z'].dropna(), 10, ...) == k) -- a True/False Series over FEWER
    # rows than the frame. pandas 2 said "Unalignable boolean Series"; pandas 3 dies on a bare AssertionError
    # (10-01 21:50). Meant: the rows where it is True -- rows it does not cover count as False.
    from pandas.core.indexing import _LocIndexer

    orig_loc_get = _LocIndexer.__getitem__

    def aligned(mask, index):
        if (isinstance(mask, pd.Series) and mask.dtype == bool and not mask.index.equals(index)
                and mask.index.isin(index).all()):
            _note("loc.mask", "a True/False mask over fewer rows than the frame (e.g. built on .dropna()) -- rows it "
                              "does not cover were taken as False")
            return mask.reindex(index, fill_value=False)
        return mask

    @functools.wraps(orig_loc_get)
    def loc_get(self, key, _orig=orig_loc_get):
        obj = self.obj
        if isinstance(key, tuple) and key:
            key = (aligned(key[0], obj.index),) + key[1:]
        else:
            key = aligned(key, obj.index)
        return _orig(self, key)

    _LocIndexer.__getitem__ = loc_get

    # A pandas frame asked for .to_pandas() (it already is one): itself.
    for cls in (pd.DataFrame, pd.Series):
        if not hasattr(cls, "to_pandas"):
            cls.to_pandas = lambda self, *a, **k: self

    # pandas' names in polars calls: clip(lower=, upper=); pl.col('x').rolling(60, min_periods=30).mean().
    for cls in (pl.Expr, pl.Series):
        orig = cls.clip

        # the bounds by every name models use: pandas lower=/upper=, numpy a_min=/a_max=, min=/max= (10-01 19:25)
        def clip(self, lower_bound=None, upper_bound=None, *, lower=None, upper=None, a_min=None, a_max=None,
                 min=None, max=None, _orig=orig, **kw):
            lo = next((v for v in (lower_bound, lower, a_min, min) if v is not None), None)
            hi = next((v for v in (upper_bound, upper, a_max, max) if v is not None), None)
            return _orig(self, lo, hi, **kw)

        functools.update_wrapper(clip, orig)
        cls.clip = clip
    orig_rolling = pl.Expr.rolling

    @functools.wraps(orig_rolling)
    def rolling(self, *args, _orig=orig_rolling, **kw):
        window = kw.pop("window", args[0] if args else None)
        if isinstance(window, int) and not isinstance(window, bool) and "period" not in kw:
            return _RollingWindow(self, window, kw.pop("min_periods", kw.pop("min_samples", None)),
                                  bool(kw.pop("center", False)))
        return _orig(self, *args, **kw)

    pl.Expr.rolling = rolling

    # rows['x'].filter(cond.to_numpy()) where cond had nulls (rolling warm-up): numpy makes an OBJECT array of
    # True/False/None and polars refuses it -- "Expected a boolean mask" (10-01 22:08). Null is False.
    def as_mask(m):
        if isinstance(m, np.ndarray) and m.dtype != bool and m.ndim == 1:
            vals = pd.Series(m, dtype="object").map(lambda v: bool(v) if v is not None and v == v else False)
            _note("filter.mask", "a numpy mask with missing values (None/NaN) was used to filter -- missing taken as False")
            return vals.to_numpy(dtype=bool)
        return m

    orig_sfilter = pl.Series.filter

    @functools.wraps(orig_sfilter)
    def series_filter(self, predicate, _orig=orig_sfilter):
        return _orig(self, as_mask(predicate))

    pl.Series.filter = series_filter
    orig_dfilter = pl.DataFrame.filter

    @functools.wraps(orig_dfilter)
    def frame_filter(self, *predicates, _orig=orig_dfilter, **kw):
        return _orig(self, *[as_mask(p) for p in predicates], **kw)

    pl.DataFrame.filter = frame_filter

    # f"{rows.select(pl.col('r').mean()):.3f}" -- a 1x1 frame (or a 1-value Series) formatted as the number it
    # holds: "unsupported format string passed to DataFrame.__format__" (10-01 22:18).
    for cls, one in ((pl.DataFrame, lambda x: x.shape == (1, 1)), (pl.Series, lambda x: len(x) == 1)):
        orig_fmt = cls.__format__

        def fmt(self, spec, _orig=orig_fmt, _one=one):
            if spec and _one(self):
                return format(self.item(), spec)
            return _orig(self, spec)

        cls.__format__ = fmt

        # ... and as that number in `if n_long else 0`, int(...), float(...): "the truth value of a DataFrame is
        # ambiguous" (10-01 23:20). Anything bigger keeps polars' own refusal.
        for dunder, conv in (("__bool__", bool), ("__float__", float), ("__int__", int)):
            orig_d = getattr(cls, dunder, None)

            def as_number(self, _orig=orig_d, _one=one, _conv=conv, _name=dunder):
                if _one(self):
                    return _conv(self.item())
                if _orig is None:
                    raise TypeError(f"{type(self).__name__} has {len(self)} values -- {_name[2:-2]}() needs one")
                return _orig(self)

            setattr(cls, dunder, as_number)

    # subset['ret30'].mean() * 10000 on an EMPTY selection: polars gives None (pandas NaN), and the next line dies --
    # "unsupported operand type(s) for *: 'NoneType' and 'int'", "format string passed to NoneType" (10-02 03:52,
    # 04:03). A numeric Series' statistics of nothing are NaN, as in pandas: they print and carry through maths.
    for name in ("mean", "median", "std", "var", "min", "max", "quantile"):
        orig_stat = getattr(pl.Series, name, None)
        if orig_stat is None:
            continue

        def stat(self, *a, _orig=orig_stat, **k):
            out = _orig(self, *a, **k)
            if out is None and self.dtype.is_numeric():
                return float("nan")
            return out

        functools.update_wrapper(stat, orig_stat)
        setattr(pl.Series, name, stat)

    # trades.sort('unit', reverse=True) -- Python's sorted() spelling (10-01 16:29); sort(ascending=False) is pandas'.
    # polars calls it descending=.
    for cls in (pl.DataFrame, pl.Series):
        orig = cls.sort

        def sort(self, *args, _orig=orig, **kw):
            if "reverse" in kw and "descending" not in kw:
                kw["descending"] = kw.pop("reverse")
            if "ascending" in kw and "descending" not in kw:
                asc = kw.pop("ascending")
                kw["descending"] = [not a for a in asc] if isinstance(asc, (list, tuple)) else not asc
            return _orig(self, *args, **kw)

        functools.update_wrapper(sort, orig)
        cls.sort = sort

    # pandas' offset aliases in polars' durations: '15min' / '15T' / '1H' -> '15m' / '15m' / '1h'
    # ("unit: 'min' not supported", group_by_dynamic, 3 runs).
    # Spelled out too: dt.offset_by('5 hours') ("expected a valid unit to follow integer in the duration string
    # '5 hours'", 10-01 replay).
    def duration(v):
        if not isinstance(v, str):
            return v
        units = {"min": "m", "T": "m", "H": "h", "D": "d", "S": "s", "week": "w", "day": "d", "hour": "h", "hr": "h",
                 "minute": "m", "second": "s", "sec": "s"}
        return re.sub(r"(\d+)\s*(weeks?|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|min|T|H|D|S)(?![A-Za-z])",
                      lambda m: m.group(1) + units[m.group(2).rstrip("s") if len(m.group(2)) > 2 else m.group(2)], v)

    def durations(fn, names):
        @functools.wraps(fn)
        def call(self, *args, _fn=fn, **kw):
            return _fn(self, *[duration(a) for a in args], **{k: duration(v) if k in names else v for k, v in kw.items()})

        return call

    # pl.col('t').dt.hour() is Int8, so dt.hour() * 60 + dt.minute() WRAPS: 14:30 became 102, not 870 (10-01
    # 14:38 -- a minute filter then dropped every row; every minute-of-day gate written so was garbage, silently).
    # The parts come back as Int32, which holds any arithmetic on them.
    dt_ns = type(pl.col("x").dt)
    for name in ("hour", "minute", "second", "day", "month", "weekday", "ordinal_day", "week", "quarter",
                 "millisecond", "microsecond"):
        orig = getattr(dt_ns, name, None)
        if orig is None:
            continue

        def part(self, *a, _orig=orig, **k):
            return _orig(self, *a, **k).cast(pl.Int32)

        functools.update_wrapper(part, orig)
        setattr(dt_ns, name, part)

    # pandas' .dt habits on polars datetimes (10-01 replay): .dt.floor('1min') is polars' .dt.truncate('1m'), and
    # .dt.cast(pl.Date) the column's own .cast(pl.Date). The parts written as pandas' PROPERTIES -- .dt.date.alias('day'),
    # pl.col('t').dt.hour * 60 ("'function' object has no attribute 'alias'", 2 runs) -- are taken as the calls.
    from polars._utils.wrap import wrap_expr, wrap_s

    for ns in (dt_ns, type(pl.Series([], dtype=pl.Datetime).dt)):
        def floor(self, every, *a, **k):
            _note("dt.floor", ".dt.floor(...) is pandas -- polars calls it .dt.truncate(...); used that")
            return self.truncate(duration(every), *a, **k)

        def cast(self, dtype, *a, **k):
            base = wrap_expr(self._pyexpr) if hasattr(self, "_pyexpr") else wrap_s(self._s)
            return base.cast(dtype, *a, **k)

        if not hasattr(ns, "floor"):
            ns.floor = floor
        if not hasattr(ns, "cast"):
            ns.cast = cast
        for name in ("date", "time", "year", "month", "day", "hour", "minute", "second", "weekday", "week",
                     "quarter", "ordinal_day"):
            fn = ns.__dict__.get(name)
            if callable(fn) and not isinstance(fn, _PartAttr):
                setattr(ns, name, _PartAttr(fn, name))

    # pl.col('a').corr(pl.col('b')) -- polars has pl.corr(a, b) only ("'Expr' object has no attribute 'corr'",
    # 10-01 15:31, three times in one script). Same for .cov.
    if not hasattr(pl.Expr, "corr"):
        pl.Expr.corr = lambda self, other, method="pearson", **kw: pl.corr(self, other, method=method, **kw)
    if not hasattr(pl.Expr, "cov"):
        pl.Expr.cov = lambda self, other, **kw: pl.cov(self, other, **kw)

    # group_by('date').agg(pl.col('Close').nth(30)) -- pandas' groupby().nth(n) (10-01 replay, 2 runs): polars calls it
    # .get(n). A group too short for it gives null (pandas leaves such a group out).
    if not hasattr(pl.Expr, "nth"):
        def nth(self, n):
            _note("expr.nth", ".nth(n) is pandas -- polars calls it .get(n); used .slice(n, 1).first()")
            return self.slice(n, 1).first()

        pl.Expr.nth = nth

    # pl.cut('ny_min', breaks=[540, 570, ...]) -- written as a polars FUNCTION, after pandas' pd.cut (10-01 replay, 2 runs);
    # polars has it as the method pl.col('ny_min').cut(breaks), with the same (a, b] intervals.
    if not hasattr(pl, "cut"):
        def cut(x, breaks, **kw):
            if isinstance(breaks, int) and not isinstance(breaks, bool):
                raise TypeError("pl.cut(x, N): polars cuts at the break POINTS you give -- pl.col(x).cut([b1, b2, ...]); "
                                "pl.col(x).qcut(N) for N equal-count bins")
            _note("pl.cut", "pl.cut(x, breaks) is written pl.col(x).cut(breaks) in polars; used that")
            if isinstance(x, str):
                x = pl.col(x)
            elif not isinstance(x, (pl.Expr, pl.Series)):
                x = pl.Series(_np.asarray(_to_pandas(x)))
            return x.cut(breaks, **kw)

        pl.cut = cut

    # pl.min(pl.col('inv_vol') * 2.0, 3.0) -- Python's min(a, b), a cap per row (10-05 21:30). polars' pl.min takes
    # column NAMES only and is their aggregate minimum, so an expression died in pl.col ("invalid input for `col`").
    # Names alone stay polars' own; one expression is its .min(); two or more values are the row-wise min.
    for name in ("min", "max"):
        orig = getattr(pl, name)
        if getattr(orig, "_ft_rowwise", False):
            continue

        def minmax(*args, _orig=orig, _name=name):
            if all(isinstance(a, str) for a in args):
                return _orig(*args)
            if len(args) == 1 and isinstance(args[0], pl.Expr):
                return getattr(args[0], _name)()
            _note(f"pl.{_name}", f"pl.{_name}(a, b) of values is the row-wise {_name} -- polars writes it "
                                 f"pl.{_name}_horizontal(a, b) (pl.{_name}('x') is a column's one {_name}imum); used that")
            return getattr(pl, f"{_name}_horizontal")(*args)

        functools.update_wrapper(minmax, orig)
        minmax._ft_rowwise = True
        setattr(pl, name, minmax)

    # size = (1.0 / pl.col('IntrVol') * 2.0).clip(...); size.to_numpy() -- an expression asked for its values
    # ("'Expr' object has no attribute 'to_numpy'", 10-05 21:3x). They are its values on the newest frame the
    # script made that has every column it reads; with no such frame it stays polars' own AttributeError.
    def on_newest_frame(name):
        def call(self, *a, **k):
            try:
                need = set(self.meta.root_names())
            except Exception:
                need = None
            for ref in _RECENT_FRAMES if need is not None else ():
                df = ref()
                if df is not None and need <= set(df.columns):
                    _note(f"expr.{name}", f"an expression has no .{name} -- it is a recipe, not data; it was run on the "
                                          f"newest frame with its columns ({df.height} rows). Write "
                                          f"df.select(expr).to_series().{name}() to say which frame")
                    return getattr(df.select(self.alias("_ft_expr")).to_series(), name)(*a, **k)
            raise AttributeError(f"'Expr' object has no attribute '{name}'")

        call.__name__ = name
        return call

    for name in ("to_numpy", "to_list"):
        if not hasattr(pl.Expr, name):
            setattr(pl.Expr, name, on_newest_frame(name))

    # pl.col(long_bw).mean() with long_bw already an expression ((pl.col('a') <= x) & ...) -- "invalid input
    # for `col`" (10-01 16:36). pl.col of an expression is that expression.
    if not isinstance(pl.col, _ColOrExpr):
        pl.col = _ColOrExpr(pl.col, pl.Expr)

    every = ("every", "period", "offset")
    for cls in (pl.DataFrame, pl.LazyFrame):
        cls.group_by_dynamic = durations(cls.group_by_dynamic, every)
        if hasattr(cls, "upsample"):
            cls.upsample = durations(cls.upsample, every)
    for name in ("truncate", "round", "offset_by"):
        ns = pl.Expr.dt.fget(pl.col("x")).__class__
        if hasattr(ns, name):
            setattr(ns, name, durations(getattr(ns, name), ("every", "by")))


import numpy as _np  # noqa: E402  (the sandbox always has numpy; ft's own functions import it lazily)


class _Array(_np.ndarray):
    """A Series' .to_numpy(): a plain ndarray, except that a pandas Series method numpy does not have
    (corr, abs, nunique, rolling, shift, ...) runs on pd.Series(this) instead of failing."""


    # Names libraries probe to tell a mapping / Series / frame from an array. Answered, they turned the array
    # into one: pandas' is_dict_like saw .keys, so sub['b'] = s.to_numpy() on a filtered frame was aligned by
    # LABEL and came out NaN (ft.resample's stamps all NaT) -- an array must stay an array to them.
    _DUCK = frozenset(("keys", "items", "index", "name", "columns", "axes", "attrs", "flags", "array", "dtypes",
                       "empty", "iloc", "loc", "iat", "at", "to_frame"))

    def __getattr__(self, name):
        import pandas as pd

        if name.startswith("_") or name in _Array._DUCK or not hasattr(pd.Series, name):
            raise AttributeError(f"'numpy.ndarray' object has no attribute '{name}'")
        _note(f"np.{name}", f".{name} is a pandas method and this is a numpy array (.to_numpy() / .values): it ran on "
                            "pd.Series(array). Keep the Series for pandas methods, or use numpy (np.corrcoef, np.abs, ...)")
        attr = getattr(pd.Series(_np.asarray(self)), name)
        if not callable(attr):
            return attr

        def call(*a, **k):
            a = [pd.Series(_np.asarray(v)) if isinstance(v, _np.ndarray) else _to_pandas(v) for v in a]
            return attr(*a, **k)

        return call


class _TimeArray(_Array):
    """A datetime column's .to_numpy(): [ts.hour for ts in rows['t'].to_numpy()] -- numpy's datetime64 has no
    .hour (10-01 16:39). Looping over it gives pandas Timestamps (.hour, .minute, .date(); equal to the
    datetime64). Indexing stays numpy's: np.unique and friends index elements and call isnan on them."""

    def __iter__(self):
        if self.dtype.kind != "M":                   # numpy keeps the subclass through np.unique's indices etc.
            return super().__iter__()
        import pandas as pd

        return (pd.Timestamp(x) for x in _np.asarray(self))


class _ColOrExpr:
    """pl.col that passes an expression through unchanged; everything else (names, dtypes, pl.col.Close)
    is polars' own pl.col."""

    def __init__(self, col, expr_cls):
        self._col, self._expr = col, expr_cls

    def __call__(self, *names, **kw):
        if len(names) == 1 and not kw and isinstance(names[0], self._expr):
            _note("col.expr", "pl.col(...) was given an expression, not a column name -- used the expression as it is")
            return names[0]
        return self._col(*names, **kw)

    def __getattr__(self, name):
        return getattr(self._col, name)


class _Columns(list):
    """A plain list -- a polars frame's column names, Series.quantile([...]) -- that also takes the pandas /
    polars calls models write on it (.tolist(), .to_list(), .values)."""

    def tolist(self):
        return list(self)

    to_list = tolist

    @property
    def values(self):
        import numpy as np

        return np.asarray(self, dtype=object)


class _DatetimeText:
    """`.str` of a column that already IS a datetime: parsing gives it back as the type asked for;
    any other string method works on its text ('2024-01-02 14:30:00.000000000')."""

    def __init__(self, x, ns):
        self._x, self._ns = x, ns

    def _parsed(self, what, out):
        _note(f"str.{what}", f".str.{what}(...) on a column that is already a datetime: it is used as it is "
                             "(no parsing needed -- .dt.date() / .dt.hour() / ft.clock work on it directly)")
        return out

    def to_datetime(self, format=None, *, time_unit=None, time_zone=None, **kw):
        return self._parsed("to_datetime", self._as_datetime(time_unit, time_zone))

    def strptime(self, dtype, format=None, **kw):
        import polars as pl

        if dtype == pl.Date or dtype == pl.Time:
            return self._parsed("strptime", self._x.cast(dtype))
        return self._parsed("strptime", self._as_datetime(getattr(dtype, "time_unit", None),
                                                          getattr(dtype, "time_zone", None)))

    def to_date(self, format=None, **kw):
        import polars as pl

        return self._parsed("to_date", self._x.cast(pl.Date))

    def to_time(self, format=None, **kw):
        import polars as pl

        return self._parsed("to_time", self._x.cast(pl.Time))

    def _as_datetime(self, unit, tz):
        import polars as pl

        out = self._x
        if isinstance(out, pl.Series) and out.dtype == pl.Date:
            out = out.cast(pl.Datetime("us"))
        if unit:
            out = out.dt.cast_time_unit(unit)
        if tz:
            out = out.dt.replace_time_zone(tz)
        return out

    def __getattr__(self, name):
        import polars as pl

        return getattr(self._x.cast(pl.String).str, name)


class _RollingWindow:
    """pl.col('x').rolling(60, min_periods=30) written the pandas way: .mean() / .std() / .sum() / ...
    are polars' rolling_mean / rolling_std / ... over that window."""

    _STATS = ("mean", "std", "var", "sum", "min", "max", "median")

    def __init__(self, expr, window, min_periods, center):
        self._expr, self._kw = expr, {"window_size": window, "min_samples": min_periods, "center": center}

    def quantile(self, quantile, interpolation="nearest"):
        return self._expr.rolling_quantile(quantile, interpolation, **self._kw)

    def apply(self, fn, *args, **kw):
        return self._expr.rolling_map(fn, **self._kw)

    def __getattr__(self, name):
        if name in self._STATS:
            return lambda *a, **k: getattr(self._expr, f"rolling_{name}")(**self._kw, **k)
        raise AttributeError(f"pl.col(...).rolling(n) supports .{', .'.join(self._STATS)}, .quantile(q), .apply(fn) "
                             f"-- not .{name}")


class _PartAttr:
    """A polars .dt part method (.dt.date, .dt.hour, ...) that also works written as pandas' property."""

    def __init__(self, fn, name):
        self._fn, self._name = fn, name
        self.__doc__ = getattr(fn, "__doc__", None)

    def __get__(self, ns, owner=None):
        if ns is None:
            return self._fn
        bound = self._fn.__get__(ns, owner)
        if not hasattr(ns, "_pyexpr"):
            return _Part(bound, self._name)          # a Series' .dt: computed only when used
        # an expression's: a real pl.Expr (so pl.col('t').dt.hour * 60 and pl.lit(1) + ...dt.hour work), callable
        out = object.__new__(_part_expr_class())
        out._pyexpr, out._ft_call, out._ft_name = bound()._pyexpr, bound, self._name
        return out


_PART_EXPR = []


def _part_expr_class():
    """pl.Expr that is also the .dt method it came from: called, it is that call; used as it is, it is
    the call's expression (with a note)."""
    if not _PART_EXPR:
        import polars as pl

        def told(self):
            name = object.__getattribute__(self, "_ft_name")
            _note(f"dt.{name}", f".dt.{name} without () is pandas' property -- in polars it is the method "
                                f".dt.{name}(); used that")

        class PartExpr(pl.Expr):
            def __call__(self, *a, **k):
                return self._ft_call(*a, **k)

            def __getattribute__(self, name):
                if not name.startswith("_"):
                    told(self)
                return pl.Expr.__getattribute__(self, name)

        def op(name):
            def call(self, *a):
                told(self)
                return getattr(pl.Expr, name)(self, *a)

            return call

        for o in ("add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "and", "or", "xor", "eq", "ne", "lt",
                  "le", "gt", "ge", "radd", "rsub", "rmul", "rtruediv", "rfloordiv", "rmod", "rand", "ror", "neg",
                  "invert", "abs"):
            if hasattr(pl.Expr, f"__{o}__"):
                setattr(PartExpr, f"__{o}__", op(f"__{o}__"))
        _PART_EXPR.append(PartExpr)
    return _PART_EXPR[0]


class _Part:
    """`.dt.hour` without the call: called, it is polars' .dt.hour(); used in any other way (.alias, * 60,
    == 15, ...), it stands for what that call returns."""

    def __init__(self, bound, name):
        self._bound, self._name = bound, name

    def __call__(self, *a, **k):
        return self._bound(*a, **k)

    def _value(self):
        _note(f"dt.{self._name}", f".dt.{self._name} without () is pandas' property -- in polars it is the method "
                                  f".dt.{self._name}(); used that")
        return self._bound()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._value(), name)

    def __array__(self, *a, **k):
        return _np.asarray(self._value(), *a, **k)

    def __len__(self):
        return len(self._value())

    def __iter__(self):
        return iter(self._value())

    def __repr__(self):
        return repr(self._value())

    __hash__ = None


def _part_op(op):
    return lambda self, *a: getattr(self._value(), op)(*a)


for _op in ("add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "and", "or", "xor", "eq", "ne", "lt", "le",
            "gt", "ge", "radd", "rsub", "rmul", "rtruediv", "rfloordiv", "rmod", "rand", "ror", "neg", "invert", "abs"):
    setattr(_Part, f"__{_op}__", _part_op(f"__{_op}__"))


def _common_names() -> None:
    """np / pd / pl without the import line (\"name 'pd' is not defined\", 6 runs): the harness
    makes the three usual aliases builtins, so a script or library module that forgot the import runs."""
    import builtins
    import importlib

    for alias, module in (("np", "numpy"), ("pd", "pandas"), ("pl", "polars")):
        if not hasattr(builtins, alias):
            try:
                setattr(builtins, alias, importlib.import_module(module))
            except ImportError:
                pass


import types as _types  # noqa: E402


class _CallableModule(_types.ModuleType):
    """A library module called like its function -- `symmetric_dabs_iv_signal(df)` after `from lib import
    symmetric_dabs_iv_signal` ("'module' object is not callable", 10-02 01:49): the call goes to the module's
    entry point (signal / regime / detect / positions / run / main)."""

    _ENTRY = ("signal", "regime", "detect", "positions", "run", "main")

    def __call__(self, *args, **kw):
        fn = next((getattr(self, n) for n in self._ENTRY if callable(getattr(self, n, None))), None)
        if fn is None:
            raise TypeError(f"'module' object is not callable -- {self.__name__} has none of "
                            f"{', '.join(self._ENTRY)}(); call one of its functions: {self.__name__.split('.')[-1]}.<function>(...)")
        _note(f"callmod.{self.__name__}", f"{self.__name__.split('.')[-1]}(...) called the module -- ran its "
                                          f"{fn.__name__}(...); write {self.__name__.split('.')[-1]}.{fn.__name__}(...)")
        return fn(*args, **kw)


class _LibCallable:
    """Makes every `lib.<module>` import a _CallableModule (first on sys.meta_path; finds nothing itself)."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("lib."):
            return None
        import importlib.machinery

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                module.__class__ = _CallableModule

            spec.loader.exec_module = exec_module
        return spec


class _ImportHelp:
    """Last on sys.meta_path, so asked only after every real import failed: a dataset or a tool
    imported as a Python module (`from trade_book_trades_19f971fff6 import trades`, 8 runs; `from
    chronos__chronos_forecast import ...`, 10; `from deci_plot import deci_plot`) fails with what it is
    and how to reach it, not a bare ModuleNotFoundError."""

    _TOOLS = {"deci_plot", "field_scan", "regime_map", "regime_lab", "query_data", "describe_data", "list_data",
              "forecast", "forecast_feature", "run_python", "library_save", "library_list", "submit_candidate",
              "team_board", "trade_review", "get_candidate", "research_search", "research_get", "ask_model"}

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in datasets():
            raise ModuleNotFoundError(
                f"No module named {top!r} -- {top!r} is a DATASET, not a Python module: "
                f"df = ft.load({top!r}) (pandas) or ft.load_pl({top!r}) (polars)", name=fullname)
        if "__" in top or top in self._TOOLS:
            raise ModuleNotFoundError(
                f"No module named {top!r} -- {top!r} is a TOOL you call (outside run_python), not a Python module. "
                "Inside a script: ft.load / ft.rows for data, ft.forecast(...) for forecasts, "
                "`from lib import <module>` for the code library", name=fullname)
        return None


_pandas_compat()
_polars_compat()
_mixup_compat()
if os.path.isdir(_FT):                              # in the sandbox only, not where ft is imported for tests
    _common_names()
    import sys as _sys

    if not any(isinstance(f, _ImportHelp) for f in _sys.meta_path):
        _sys.meta_path.append(_ImportHelp())
    if not any(isinstance(f, _LibCallable) for f in _sys.meta_path):
        _sys.meta_path.insert(0, _LibCallable())


def __getattr__(name: str):
    """`from ft import rows_pl, ft` (10-01 replay, 2 submissions: "cannot import name 'ft' from 'ft'"): ft
    imported from itself is the module."""
    if name == "ft":
        import sys

        me = next((m for m in list(sys.modules.values()) if getattr(m, "__dict__", None) is globals()), None)
        if me is not None:
            return me
    raise AttributeError(f"module 'ft' has no attribute {name!r}")


def datasets() -> list[str]:
    """The view names you can pass to load()."""
    return [c["view"] for c in _CATALOG]


def _unique(columns):
    """`columns` without repeats, in order. Models list a column twice in long column lists
    (['IV_AtmD0', ..., 'IV_AtmD0']); polars refuses that with a DuplicateError deep in its planner
    and the run is lost for nothing."""
    return None if columns is None else list(dict.fromkeys(columns))


def _find(name: str) -> dict:
    """The catalog entry for a view name or path. A guessed name (fc_imb_oinet_d0_forecast_1, or
    fc_fc_... with the prefix doubled -- bugs #402/#406) is answered with the nearest real names
    first: the full list is long enough that the monitor and the models both lose its tail."""
    import difflib

    for c in _CATALOG:
        if name in (c["view"], c["path"]):
            return c
    names = datasets()
    if isinstance(name, str) and name.startswith("fc_fc_") and name[3:] in names:
        return _find(name[3:])
    near = difflib.get_close_matches(str(name), names, n=3, cutoff=0.6)
    hint = f" did you mean {' or '.join(repr(n) for n in near)}?" if near else ""
    raise KeyError(f"no dataset {name!r};{hint} available: {', '.join(names) or '(none)'}")


def _task_view(name, loader: str) -> bool:
    """ft.load('mcp_tasks_gex_gex_intraday_src_2') in a task run: that view is the project's copy of the
    task's in-sample rows, for the analysis tools (query_data, decile plots) -- a scored run or a library
    smoke test mounts no project data, so it was "no dataset ...; available: (none)" (bug #428). The task's
    rows are what it holds, so they are served, as ft.rows() / ft.rows_pl() would."""
    if not (isinstance(name, str) and name.startswith("mcp_tasks_")) \
            or any(name in (c["view"], c["path"]) for c in _CATALOG) \
            or not os.path.exists(os.path.join(_TASK, "rows.parquet")):
        return False
    _note(f"taskview.{loader}", f"{name!r} is the analysis tools' copy of this task's rows -- a scored run has "
                                f"the rows themselves: ft.{loader}(columns=[...]) gave them; call that")
    return True


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


def _no_name(who: str, columns):
    """ft.load() / ft.load_pl() without a dataset name: in a task objective the rows are the one thing to load
    (`rows = ft.load()` in a library test, 10-01 replay: "load() missing 1 required positional argument: 'name'")."""
    if os.path.exists(os.path.join(_TASK, "rows.parquet")):
        _note(f"{who}.rows", f"ft.{who}() without a dataset name -- this objective's data are its task rows: gave "
                             f"ft.{'rows' if who == 'load' else 'rows_pl'}(), which is what to call")
        return (rows if who == "load" else rows_pl)(columns)
    raise TypeError(f"ft.{who}() needs the dataset's name: ft.{who}('<view>') -- one of: {', '.join(datasets()) or '(none)'}")


def load(name: str | None = None, columns: list[str] | None = None, prefix: str | None = None):
    """Load a dataset as a pandas DataFrame. `columns` limits what is read (parquet only).

    `prefix` renames every column except the time column ``t``: forecast features all share
    column names (fc_median, fc_q10, ...), so merging two of them leaves pandas' fc_median_x /
    fc_median_y and ``df["fc_median"]`` fails. ``ft.load("fc_a", prefix="a_")`` gives a_fc_median.
    """
    import pandas as pd

    if name is None:
        return _no_name("load", columns)
    columns = _unique(columns)
    if _task_view(name, "rows"):
        df = rows(columns)
        return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c != "t"}) if prefix else df
    item = _find(name)
    _note_used(item["view"])
    p = path(name)
    fmt = item.get("format", "")
    if fmt == "parquet":
        columns, renames = _time_alias(p, columns)
        df = pd.read_parquet(p, columns=columns).rename(columns=renames)
    elif fmt in ("csv", "tsv"):
        df = pd.read_csv(p, sep="\t" if fmt == "tsv" else ",", usecols=columns)
    elif fmt in ("jsonl", "ndjson"):
        df = pd.read_json(p, lines=True)
    else:
        df = pd.read_json(p)
    if prefix:
        df = df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c != "t"})
    return df


def load_pl(name: str | None = None, columns: list[str] | None = None, prefix: str | None = None):
    """Load a dataset as a POLARS DataFrame -- several times faster than load() on the 10s bar
    data (700k+ rows). Same arguments as load(); convert with .to_pandas() if you need pandas.

        df = ft.load_pl("sql_exports_dbo_gexbar10s", columns=["ts", "Close", "GEX"])
    """
    import polars as pl

    if name is None:
        return _no_name("load_pl", columns)
    columns = _unique(columns)
    if _task_view(name, "rows_pl"):
        df = rows_pl(columns)
        return df.rename({c: f"{prefix}{c}" for c in df.columns if c != "t"}) if prefix else df
    item = _find(name)
    _note_used(item["view"])
    p = path(name)
    fmt = item.get("format", "")
    if fmt == "parquet":
        src = os.path.join(p, "*.parquet") if os.path.isdir(p) else p
        df = pl.scan_parquet(src)
        columns, renames = _time_alias(p, columns)
        df = (df.select(columns) if columns else df).collect().rename(renames)
    elif fmt in ("csv", "tsv"):
        df = pl.read_csv(p, separator="\t" if fmt == "tsv" else ",", columns=columns, try_parse_dates=True)
    elif fmt in ("jsonl", "ndjson"):
        df = pl.read_ndjson(p)
    else:
        df = pl.read_json(p)
    if prefix:
        df = df.rename({c: f"{prefix}{c}" for c in df.columns if c != "t"})
    return _remember(df)


def _time_alias(p, columns):
    """(columns to read, {read name: asked name}) for a parquet dataset: a time column asked for under
    another dataset's name -- columns=['SlotUtc', 'fc_median'] on a forecast, whose time is `t` (10-01
    14:44) -- reads the dataset's own time column and gives it the name asked for."""
    if not columns:
        return columns, {}
    import polars as pl

    src = os.path.join(p, "*.parquet") if os.path.isdir(p) else p
    have = set(pl.scan_parquet(src).collect_schema().names())
    renames = {}
    for i, c in enumerate(columns):
        if c in _TIME_COLUMNS and c not in have:
            real = next((t for t in _TIME_COLUMNS if t in have and t not in columns), None)
            if real:
                columns = columns[:i] + [real] + columns[i + 1:]
                renames[real] = c
                _note(f"time.{c}", f"this dataset has no {c!r} -- its time column is {real!r}, loaded under the name {c!r}")
        elif c not in have:
            real = _same_column(c, have)
            if real and real not in columns:
                columns = columns[:i] + [real] + columns[i + 1:]
                renames[real] = c
                _note(f"col.{c}", f"this dataset has no {c!r} -- loaded {real!r} under the name {c!r}")
    return columns, renames


def _same_column(name, have):
    """The one real column a missing name means, or None: the same name in other letter case
    ('Gexflip_Neg' -> GexFlip_Neg), else the one column that is the name under a group prefix
    ('TotalAbsGex' -> Pinning_TotalAbsGex, 10-01 19:36). Two candidates -> None: no guessing."""
    low = name.lower()
    same = [h for h in have if h.lower() == low]
    if len(same) == 1:
        return same[0]
    tail = [h for h in have if h.lower().endswith("_" + low)]
    return tail[0] if len(tail) == 1 else None


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
        t = pd.to_datetime(_times(df, time))
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
    return _Clock((session.view(_Array), minute.view(_Array)))


class _Clock(tuple):
    """ft.clock's (session, minute): unpacks as a pair, and clock["session"] / clock.minute work
    too -- models read it as a dict (`clock['session']`, 10-01 12:16: "tuple indices must be
    integers or slices, not str")."""

    session = property(lambda self: self[0])
    minute = property(lambda self: self[1])

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in ("session", "minute"):
                raise KeyError(f"ft.clock gives (session, minute); no {key!r}")
            return getattr(self, key)
        return tuple.__getitem__(self, key)


def _times(rows, time="t"):
    """rows[time] -- or, when the frame has no such column, its time under another name: a frame
    loaded from a dataset has SlotUtc, not the rows' t (KeyError: 't' from inside quick_score /
    trend_exits, 6 runs), or a pandas DatetimeIndex."""
    import pandas as pd

    cols = list(rows.columns)
    if time in cols:
        return rows[time]
    if time == "t":
        for alt in _TIME_COLUMNS[1:]:
            if alt in cols:
                return rows[alt]
        index = getattr(rows, "index", None)
        if isinstance(index, pd.DatetimeIndex):
            return pd.Series(index, index=index)
    raise KeyError(f"no time column {time!r} in the rows (columns: {', '.join(map(str, cols[:12]))}"
                   f"{', ...' if len(cols) > 12 else ''}) -- pass time='<its name>'")


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
    session, _ = clock(_times(rows, time), tz)
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
    session, minute = clock(_times(rows, time), tz)
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
    return pd.Series(out, index=pd.DatetimeIndex(pd.to_datetime(_times(rows, time))), name="position")


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
    session, minute = clock(_times(rows, time), tz)
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
    return pd.Series(pos, index=pd.DatetimeIndex(pd.to_datetime(_times(rows, time))), name="position")


def decision_points(rows, times=("09:45", "10:00", "10:30", "11:00", "11:30", "12:00", "13:00", "14:00", "15:00"),
                    time: str = "t", tz: str = _TZ):
    """Row positions (integers) of fixed decision times: for each session and each clock time,
    the first row at or after it. Use them to study what happens after a decision, or to act only
    at those times.

        pts = ft.decision_points(rows, times=["10:00", "10:30", "11:30", "13:00", "14:00"])
        rows.iloc[pts]
    """
    import numpy as np

    session, minute = clock(_times(rows, time), tz)
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
    session, minute = clock(_times(rows, time), tz)
    starts = _session_starts(session)
    ends = np.r_[starts[1:], len(p)]
    day_of = np.repeat(np.arange(len(starts)), ends - starts)
    t = pd.to_datetime(np.asarray(_times(rows, time)))
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
    if isinstance(columns, str):
        if columns in datasets():
            return _dataset_not_rows("load", columns)
        columns = [columns]
    cols, derived = _task_columns(path, columns)
    df = pd.read_parquet(path, columns=cols).sort_values("t", kind="stable").reset_index(drop=True)
    return _derive(df, columns, derived)


def rows_pl(columns: list[str] | None = None):
    """The task's rows as a POLARS DataFrame sorted by `t` -- the same rows as ft.rows(), several
    times faster to load and compute on (700k+ rows). Every ft helper accepts polars frames and
    series as they are; report with ft.report_actions(values, t=rows["t"])."""
    import polars as pl

    path = os.path.join(_TASK, "rows.parquet")
    if not os.path.exists(path):
        raise RuntimeError("ft.rows_pl(): this objective is not scored by a task server -- use ft.load_pl() instead")
    if isinstance(columns, str):
        if columns in datasets():
            return _dataset_not_rows("load_pl", columns)
        columns = [columns]
    cols, derived = _task_columns(path, columns)
    return _remember(_derive(pl.read_parquet(path, columns=cols).sort("t", maintain_order=True), columns, derived))


def _dataset_not_rows(loader: str, name: str):
    """ft.rows_pl('trade_book_trades_19f971fff6') -- a DATASET named where the rows' columns go (10-01 replay,
    2 submissions; the name was read as 27 one-letter columns): it is loaded, as ft.load / ft.load_pl would."""
    _note(f"rows.{name}", f"ft.{'rows' if loader == 'load' else 'rows_pl'}({name!r}): {name!r} is a dataset, not the "
                          f"rows -- loaded it with ft.{loader}({name!r}), which is what to call")
    return globals()[loader](name)


# Columns models ask ft.rows(columns=[...]) for that the rows do not have but ft can make (Qwen asked for
# "VWAP" twice on 10-01 14:08): name -> (columns it needs, how to make it from the frame, what it is).
_DERIVED = {"VWAP": (("Close", "Volume"), lambda df: session_vwap(df), "ft.session_vwap(rows)")}


def _derivation(name: str, have: set):
    """(columns it needs, how to make it, what it is) for a column ft can make, else None: VWAP, and the
    features the TRADE REVIEW reports entries by (app/trade_book.py snapshots) -- agents read them off the
    brief and ask the rows for them (10-01 00:42-03:59, 6 runs: 'GexFlip_Pos_vs_price_bps',
    'minutes_into_session'). Made on the rows as the review defines them, per session, causally; `price`
    is the task's target (Close)."""
    import re

    if name in _DERIVED:
        return _DERIVED[name]
    try:
        price = task().get("target") or "Close"
    except Exception:  # noqa: BLE001 -- no task.json: the rows' Close
        price = "Close"
    price = price if price in have else "Close"
    m = re.fullmatch(r"(\w+)_vs_price_bps", name)
    if m and m.group(1) in have:
        c = m.group(1)
        return ((c, price), lambda df: (_col(df, c) / _col(df, price) - 1.0) * 1e4,
                f"({c} / {price} - 1) * 1e4")
    if name == "minutes_into_session":
        return ((), lambda df: _session_clock(df)[1], "minutes since the session's first row")
    m = re.fullmatch(r"price_chg_(\d+)m_bps", name)
    if m:
        k = int(m.group(1))
        return ((price,), lambda df: _price_change(df, price, k), f"({price} / {price} {k} minutes earlier in the "
                                                                   "session - 1) * 1e4")
    if name == "price_since_open_bps":
        return ((price,), lambda df: _price_change(df, price, None), f"({price} / the session's first {price} - 1) * 1e4")
    if name == "price_in_day_range":
        return ((price,), lambda df: _day_range(df, price), f"where {price} sits in the session's range so far (0..1)")
    return None


def _session_clock(df):
    """(index of each row's session start, minutes since it, times as int64 ns) -- rows sorted by t."""
    import numpy as np

    t = np.asarray(_to_pandas(df["t"]), dtype="datetime64[ns]").astype("int64")
    session, _ = clock(t.astype("datetime64[ns]"))
    starts = _session_starts(np.asarray(session))
    first = starts[np.searchsorted(starts, np.arange(len(t)), side="right") - 1]
    return first, (t - t[first]) / 60e9, t


def _price_change(df, price, minutes):
    import numpy as np

    p = _col(df, price)
    first, _, t = _session_clock(df)
    j = first if minutes is None else np.searchsorted(t, t - int(minutes * 60e9), side="right") - 1
    ok = j >= first
    out = np.full(len(p), np.nan)
    with np.errstate(all="ignore"):
        out[ok] = (p[ok] / p[j[ok]] - 1.0) * 1e4
    return out


def _day_range(df, price):
    import numpy as np
    import pandas as pd

    p = pd.Series(_col(df, price))
    first, _, _ = _session_clock(df)
    hi, lo = p.groupby(first).cummax().to_numpy(), p.groupby(first).cummin().to_numpy()
    with np.errstate(all="ignore"):
        return np.where(hi > lo, (p.to_numpy() - lo) / (hi - lo), np.nan)


def _task_columns(path, columns):
    """(columns to read, derived names to add) for ft.rows / ft.rows_pl(columns=...). A name the rows
    lack fails here, with the nearest real names, instead of deep in a query plan listing all 146."""
    if columns is None:
        return None, []
    import difflib

    import pyarrow.parquet as pq

    have = set(pq.read_schema(path).names)
    want = _unique(["t"] + [c for c in columns if c != "t"])
    # 'Gexflip_Neg' for GexFlip_Neg (10-01 16:46): a name that differs from exactly one column only in letter case
    # IS that column -- read it and give it the name asked for.
    lower = {}
    for h in have:
        lower.setdefault(h.lower(), []).append(h)
    make = {}
    for c in want:
        if c in have:
            continue
        real = _same_column(c, have)
        make[c] = (([real], lambda df, _h=real: df[_h], f"the column {real!r}")
                   if real else _derivation(c, have))
    derived = [c for c, how in make.items() if how is not None and set(how[0]) <= have]
    unknown = [c for c in make if c not in derived]
    if unknown:
        near = {c: difflib.get_close_matches(c, sorted(have), n=3, cutoff=0.6) for c in unknown}
        raise KeyError("ft.rows(columns=...): the rows have no " + "; ".join(
            f"{c!r}" + (f" (did you mean {', '.join(map(repr, n))}?)" if n else "") for c, n in near.items())
            + " -- ft.task()['columns'] lists every column")
    need = [c for d in derived for c in make[d][0] if c not in want]
    return [c for c in want if c in have] + _unique(need), [(d, make[d]) for d in derived]


def _derive(df, columns, derived):
    if not derived:
        return df
    import numpy as np

    polars = type(df).__module__.startswith("polars")
    for name, (_, make, what) in derived:
        values = np.asarray(make(df), dtype=float)
        if polars:
            import polars as pl

            df = df.with_columns(pl.Series(name, values))
        else:
            df[name] = values
        _note(f"derived.{name}", f"{name!r} is not a column of the rows -- computed it ({name}: {what})")
    keep = _unique(["t"] + [c for c in columns if c != "t"])
    return df.select(keep) if polars else df[keep]


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


# Parameter names models reach for from other backtest libraries, and the ft parameter that does
# that job. Only suggested -- never mapped silently, the meanings are close but not the same.
_PARAM_ALIASES = {
    "atr_lookback": "vol_window", "atr_window": "vol_window", "atr_period": "vol_window",
    "lookback": ("vol_window", "lookback_days"),
    "vol_lookback": "vol_window", "atr_mult": "stop_mult", "trail_mult": "stop_mult", "atr_multiplier": "stop_mult",
    "stop_atr": "stop_mult", "trailing_stop": "trail", "entry_start": "no_entry_before",
    "end_time": "flat_at", "exit_time": "flat_at", "eod_exit": "flat_at", "flat_time": "flat_at",
    "last_entry": "no_entry_after", "entry_end": "no_entry_after", "max_trades": "max_trades_per_day",
    "size_func": "size", "sizing": "size", "sizes": "size", "base": "size", "retrigger_on": "retrigger",
    "close": "price", "price_col": "price", "close_col": "price", "time_col": "time", "timestamp": "time",
    # the first moment a helper may act: trend_exits calls it no_entry_before, noise_area_breakout first_check
    # (start_bar / start_minute on noise_area_breakout, 2026-10-01)
    "start_time": ("no_entry_before", "first_check"), "start_bar": ("no_entry_before", "first_check"),
    "start_minute": ("no_entry_before", "first_check"), "first_entry": ("no_entry_before", "first_check"),
    "check_minutes": "check_every", "check_interval": "check_every", "lookback_sessions": "lookback_days",
    "band_width": "band_mult", "band_k": "band_mult",
}


# Names that mean exactly what the ft parameter does: trend_exits' stop IS an ATR-style distance
# (stop_mult x the typical move over vol_window rows), and the team's lessons say "2-4x ATR trailing
# stop" -- so atr_lookback / atr_mult came back again and again (11 runs on 2026-09-30/10-01, after the
# refusal named vol_window). These are taken as said, with a note; any other unknown name is refused.
_SAME_AS = {"atr_lookback": "vol_window", "atr_window": "vol_window", "atr_period": "vol_window",
            "atr_length": "vol_window", "atr_mult": "stop_mult", "atr_multiplier": "stop_mult",
            # noise_area_breakout's window is counted in sessions (2026-10-04: lookback=14 refused)
            "lookback": "lookback_days", "lookback_sessions": "lookback_days"}


def _same_as(fn, kw: dict) -> dict:
    import inspect
    import sys

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kw
    out = dict(kw)
    for k, real in _SAME_AS.items():
        if k in out and k not in params and real in params and real not in out:
            out[real] = out.pop(k)
            print(f"[ft] {fn.__name__}: {k}= is called {real}= here -- used {real}={out[real]!r}", file=sys.stderr)
    return out


def _bad_call(fn, exc: TypeError, kw: dict) -> TypeError | None:
    """A TypeError naming what `fn` does take, for a call it cannot bind; None if it binds.
    "trend_exits() got an unexpected keyword argument 'atr_lookback'" sent agents to a
    run_python just to read the signature -- or to submit the same guess again."""
    import difflib
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = [p for p in sig.parameters if p not in ("self",)]
    lines = [f"ft.{fn.__name__}: {exc}."]
    for k in kw:
        if k in params:
            continue
        alias = _PARAM_ALIASES.get(k, ())
        near = [a for a in ((alias,) if isinstance(alias, str) else alias) if a in params]             or difflib.get_close_matches(k, params, n=2)
        if near:
            lines.append(f"  '{k}' is not a parameter -- did you mean {' or '.join(repr(n) for n in near)}?")
    lines.append(f"  ft.{fn.__name__}{sig}")
    lines.append(f"  (print(ft.{fn.__name__}.__doc__) explains each parameter)")
    return TypeError("\n".join(lines))


def _accepts_polars(fn):
    """Let a pandas-based helper take polars frames and series (converted on the way in)."""
    import functools
    import inspect

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        kw = _same_as(fn, kw)
        try:
            inspect.signature(fn).bind(*args, **kw)
        except TypeError as exc:
            bad = _bad_call(fn, exc, kw)
            if bad is not None:
                raise bad from None
        except ValueError:
            pass
        args, kw = _eval_exprs(fn, args, kw)
        return fn(*[_to_pandas(a) for a in args], **{k: _to_pandas(v) for k, v in kw.items()})

    return wrapper


def _eval_exprs(fn, args, kw):
    """A polars EXPRESSION where a helper wants one value per row -- trend_exits(pl.when(...).then(1)
    .otherwise(0), rows) -- evaluated on the frame passed beside it. It died in float() ("float()
    argument must be ... not 'Expr'", 2026-09-30 and 10-01), though what the agent meant was clear."""
    def is_expr(x):
        return type(x).__module__.startswith("polars") and type(x).__name__ == "Expr"

    if not any(is_expr(a) for a in (*args, *kw.values())):
        return args, kw
    import polars as pl

    frame = next((a for a in (*args, *kw.values()) if isinstance(a, pl.DataFrame)), None)
    if frame is None:
        pdf = next((a for a in (*args, *kw.values()) if type(a).__name__ == "DataFrame"), None)
        frame = pl.from_pandas(pdf) if pdf is not None else None
    if frame is None:
        raise TypeError(f"ft.{fn.__name__}: got a polars expression (pl.col(...) / pl.when(...)) but no frame to "
                        "evaluate it on -- pass the rows too, or evaluate it first: rows.select(expr).to_series()")

    def ev(x):
        return frame.select(x.alias("_ft_expr")).to_series() if is_expr(x) else x

    return [ev(a) for a in args], {k: ev(v) for k, v in kw.items()}


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
    """Extra numbers worth showing next to the score (trade count, turnover, ...).

    A whole series passed as positions= or returns= is what report_positions / report_returns
    take, and goes to them: ft.report(positions=pos) used to keep str(pos)[:200] as an "extra
    number", report no positions, and fail the submission with "no positions reported" (#6)."""
    for k, fn in (("positions", report_positions), ("returns", report_returns)):
        v = numbers.get(k)
        if v is not None and not isinstance(v, (str, bytes)) and hasattr(v, "__len__"):
            print(f"[ft] report({k}=...) is a series: reporting it with ft.report_{k}()")
            fn(v)                                # (the polars-accepting wrappers, by now)
            numbers.pop(k)
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


ACTIVE_SHARE_FLOOR = 0.3


def quick_score(positions, rows, price: str = "Close", time: str = "t", *, cost_bps: float | None = None,
                delay: int = 1, tz: str = _TZ) -> dict:
    """An APPROXIMATE in-sample score of one position series, in a second -- for comparing ideas
    and parameter variants inside run_python before you spend a submission. Same rules as the
    task server: the position decided at a row is filled `delay` bars later, pays cost_bps per unit
    of position change (the task's own cost by default), and is closed at each session's last
    row. The server's number differs a little (its fills and daily accounting are exact): use
    this to RANK variants, and the submission for the real score.

    Returns {sharpe, sharpe_gross, sharpe_flipped, sharpe_h1, sharpe_h2, worst_half, bps_per_day,
    trades_per_day, long_share, short_share, active_days, days, active_share, floors_ok} -- h1/h2 are
    the first and second half of the sessions: an idea worth keeping is positive in BOTH (worst_half > 0).
    active_share is the share of sessions with a position: under ACTIVE_SHARE_FLOOR the unseen holdout
    (a few months) gets too few active days to be scored at all, and the candidate is not ranked."""
    import numpy as np

    pos = np.nan_to_num(np.asarray(_to_pandas(positions), dtype=float), nan=0.0)
    p = _col(_to_pandas(rows), price)
    if len(pos) != len(p):
        raise ValueError(f"quick_score: {len(pos)} positions for {len(p)} rows -- one per row")
    if len(p) == 0:
        raise ValueError("quick_score: the rows are EMPTY -- a filter earlier in the script removed every row "
                         "(check the time-of-day / session filters: print(rows.height) after each one)")
    session, _ = clock(_times(_to_pandas(rows), time), tz)
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
    return _Score({"sharpe": _sharpe(daily), "sharpe_gross": _sharpe(gross_d), "sharpe_flipped": _sharpe(flip_d),
            "sharpe_h1": h1, "sharpe_h2": h2,
            "worst_half": min(h1, h2) if math.isfinite(h1) and math.isfinite(h2) else float("nan"),
            "bps_per_day": float(np.mean(daily) * 1e4) if n else float("nan"), "trades_per_day": tpd,
            "long_share": ls, "short_share": ss, "active_days": active, "days": n,
            "active_share": active / n if n else 0.0,
            "trades": trades, "long_trades": longs, "short_trades": shorts,
            "floors_ok": bool(tpd >= 2.0 and min(ls, ss) >= 0.2 and n and active / n >= ACTIVE_SHARE_FLOOR)})


class _Score(dict):
    """ft.quick_score's stats -- a dict, that also stands for its Sharpe where a NUMBER is written:
    f"{score:.4f}" (Qwen, 10-01 14:16: "unsupported format string passed to dict.__format__"),
    float(score), score > best, max(scores)."""

    def __missing__(self, key):
        # score['in_sample']['sharpe'] -- the shape of a SCORED candidate's metrics (get_candidate), 10-01 14:19.
        # A quick score is in-sample already: it is its own in_sample.
        if key in ("in_sample", "is", "metrics", "score"):
            return self
        # names models guess for the fields (score['half1'], 10-02 05:06)
        alias = {"half1": "sharpe_h1", "h1": "sharpe_h1", "first_half": "sharpe_h1", "sharpe_first_half": "sharpe_h1",
                 "half2": "sharpe_h2", "h2": "sharpe_h2", "second_half": "sharpe_h2", "sharpe_second_half": "sharpe_h2",
                 "sharpe_ratio": "sharpe", "net_sharpe": "sharpe", "gross_sharpe": "sharpe_gross",
                 "n_trades": "trades", "num_trades": "trades", "trade_count": "trades",
                 "n_long": "long_trades", "n_short": "short_trades"}.get(key) if isinstance(key, str) else None
        if alias in self:
            return self[alias]
        # score[0] -- read as a tuple (10-02 00:54): the values in order, sharpe first
        if isinstance(key, int) and not isinstance(key, bool) and -len(self) <= key < len(self):
            return list(self.values())[key]
        raise KeyError(f"{key!r} -- ft.quick_score gives in-sample stats only: {', '.join(self)}")

    def __format__(self, spec):
        return format(self["sharpe"], spec) if spec else dict.__repr__(self)

    def __float__(self):
        return float(self["sharpe"])

    def _num(self, other):
        return float(other) if isinstance(other, (int, float, _Score)) else NotImplemented

    def __lt__(self, other):
        o = self._num(other)
        return o if o is NotImplemented else self["sharpe"] < o

    def __le__(self, other):
        o = self._num(other)
        return o if o is NotImplemented else self["sharpe"] <= o

    def __gt__(self, other):
        o = self._num(other)
        return o if o is NotImplemented else self["sharpe"] > o

    def __ge__(self, other):
        o = self._num(other)
        return o if o is NotImplemented else self["sharpe"] >= o


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
    import pandas as pd

    df = _to_pandas(rows)
    p = _col(df, price)
    session, minute = clock(_times(df, time), tz)
    starts = _session_starts(session)
    ends = np.r_[starts[1:], len(p)]
    cost = _cost_bps(cost_bps)
    tx = sorted(_hhmm(s) for s in times)
    ex = _hhmm(exit)
    vol = np.asarray(df["Volume"], dtype=float) if "Volume" in df.columns else None
    skip = {time, "Open", "High", "Low", "Close", price}
    if fields is None:
        # numbers only: pandas 3 names its text dtype "str", which slipped past a check for "object"/"string" and
        # died on float('2023-08-01') (a polars date-as-text column, 10-01 replay)
        fields = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])
                  and not pd.api.types.is_bool_dtype(df[c])]
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
