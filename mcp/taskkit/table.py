"""TableTask: a task defined by a JSON file instead of code (optional helper, polars).

    {
      "name": "bike_rentals",
      "title": "...", "description": "What the problem is, for the agents.", "brief": "Extra rules.",
      "sources": [
        {"path": "data/bike_rentals.csv", "time_column": "time"},
        {"path": "data/weather.parquet", "time_column": "ts", "columns": ["wind"], "prefix": "w_"}
      ],
      "target": "rentals",
      "holdout_from": "2024-11-01",
      "action": {"kind": "value", "description": "..."},
      "evaluator": {"kind": "forecast", "horizon": 1},
      "shift_rows": {"rows": 2, "except": ["Open", "High", "Low", "Close", "Volume"]},
      "ahead_columns": ["temp_next_hour_forecast"],
      "column_notes": {"rentals": "bikes rented this hour"}
    }

Sources: the FIRST source defines the rows (one per distinct timestamp; duplicates are averaged,
text keeps the last). Every other source is joined AS OF each row's time -- its latest row at or
before that time -- so a slower or irregular table never leaks a value from the future. `columns`
limits a source, `prefix` renames its columns. Paths are relative to the JSON file; globs of
parquet, csv/tsv and ndjson are read by polars.

`shift_rows` delays columns that are filed before they were really known: every column but
`except`, or only `columns` (`*` at the end matches a prefix), take their value from `rows` rows
earlier. `ahead_columns` names columns legitimately known in advance (published forecasts).

Evaluators: "trading" (actions are positions in `target`) and "forecast" (actions predict `target`
`horizon` rows ahead) -- see taskkit.evaluators.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import evaluators
from .task import KEY, Task, cache_dir


def _scan(path: str) -> pl.LazyFrame:
    ext = os.path.splitext(path.replace("*", "x"))[1].lower()
    if ext == ".parquet":
        return pl.scan_parquet(path)
    if ext in (".csv", ".tsv"):
        return pl.scan_csv(path, separator="\t" if ext == ".tsv" else ",", try_parse_dates=True)
    if ext in (".ndjson", ".jsonl"):
        return pl.scan_ndjson(path)
    raise ValueError(f"cannot read {path!r}: use parquet, csv, tsv or ndjson")


def _match(c: str, pats: list[str]) -> bool:
    return any(c == p or (p.endswith("*") and c.startswith(p[:-1])) for p in pats)


class TableTask(Task):
    def __init__(self, config: dict[str, Any], base_dir: Path):
        self.config = config
        self.base_dir = base_dir
        self.name = str(config["name"])
        self.title = config.get("title") or self.name
        self.description = config.get("description") or ""
        self.brief = config.get("brief") or ""
        self.target = config["target"]
        self.holdout_from = config.get("holdout_from") or ""
        self.mid_cut = config.get("mid_cut")
        ev = dict(config.get("evaluator") or {"kind": "trading"})
        self.evaluator = ev
        kind = ev.get("kind", "trading")
        if kind == "trading":
            # Every trading score reads higher-is-better (max_drawdown is negative: nearer 0 is better).
            self.score_name, self.higher_is_better = ev.get("score", "sharpe"), True
            default_action = {"kind": "position", "min": -1.0, "max": 1.0,
                              "description": f"Units of {self.target} held from this row to the next; negative = short, 0 = flat."}
        elif kind == "forecast":
            self.score_name, self.higher_is_better = "skill", True
            default_action = {"kind": "value", "min": None, "max": None,
                              "description": f"Your prediction, made at this row, of {self.target} "
                                             f"{ev.get('horizon', 1)} row(s) ahead."}
        else:
            raise ValueError(f"task {self.name}: unknown evaluator kind {kind!r} (trading | forecast)")
        self.action = {"initial": 0.0, **default_action, **(config.get("action") or {})}
        self.column_notes = dict(config.get("column_notes") or {})
        self.ahead_columns = list(config.get("ahead_columns") or [])
        self.target_options = list(config.get("target_options") or [])
        self.display_tz = config.get("display_tz") or "UTC"
        summary = ({"trading": f"daily returns of positions in the target, scored by {ev.get('score', 'sharpe')}, net of "
                               f"{ev.get('cost_bps', 1.0)} bps per unit traded",
                    "forecast": f"skill of forecasts {ev.get('horizon', 1)} row(s) ahead against 'no change' "
                                "(1 - squared error / naive squared error)"}[kind])
        self.valuation_info = {"summary": summary, **{k: v for k, v in ev.items() if k != "kind"}, "evaluator": kind}
        self.sources = list(config.get("sources") or [])
        if not self.sources:
            raise ValueError(f"task {self.name}: no sources")

    def _paths(self) -> list[str]:
        return [s["path"] if os.path.isabs(s["path"]) else str((self.base_dir / s["path"]).resolve())
                for s in self.sources]

    def version_parts(self) -> list[Any]:
        parts: list[Any] = [json.dumps(self.config, sort_keys=True)]
        for p in self._paths():
            parts.append([(f, os.path.getsize(f), int(os.path.getmtime(f))) for f in sorted(glob.glob(p))])
        return parts

    def load_rows(self) -> pl.DataFrame:
        cached = cache_dir() / f"table-{self.name}-{self.version()}.parquet"
        if cached.is_file():
            return pl.read_parquet(cached)
        frames = []
        for s, path in zip(self.sources, self._paths()):
            if not glob.glob(path):
                raise FileNotFoundError(f"task {self.name}: source {path!r} matches no files")
            lf = _scan(path)
            schema = lf.collect_schema()
            tc = s["time_column"]
            keep = [c for c in (s.get("columns") or list(schema)) if c != tc]
            missing = [c for c in keep + [tc] if c not in schema]
            if missing:
                raise ValueError(f"task {self.name}: source {path!r} has no column(s) {missing}")
            t = pl.col(tc)
            t = t.str.to_datetime() if schema[tc] == pl.Utf8 else t
            prefix = s.get("prefix") or ""
            aggs = [(pl.col(c).mean() if schema[c].is_numeric() else pl.col(c).last()).alias(prefix + c) for c in keep]
            frames.append(lf.with_columns(t.cast(pl.Datetime("ns")).alias(KEY)).drop_nulls(KEY)
                          .group_by(KEY).agg(aggs).sort(KEY))
        df = frames[0]
        for f in frames[1:]:
            df = df.join_asof(f, on=KEY, strategy="backward")
        df = df.collect()
        shift = self.config.get("shift_rows")
        if shift and int(shift.get("rows") or 0) > 0:
            n = int(shift["rows"])
            if shift.get("columns"):
                cols = [c for c in df.columns if c != KEY and _match(c, list(shift["columns"]))]
            else:
                keep = list(shift.get("except") or []) + [self.target]
                cols = [c for c in df.columns if c != KEY and not _match(c, keep)]
            df = df.with_columns([pl.col(c).shift(n) for c in cols])
        if self.target not in df.columns:
            raise ValueError(f"task {self.name}: target {self.target!r} is not a column of the rows")
        tmp = cached.with_name(cached.name + ".tmp")
        df.write_parquet(tmp)
        tmp.replace(cached)
        return df

    def evaluate(self, rows: pl.DataFrame, actions: np.ndarray) -> dict[str, Any]:
        ev = {k: v for k, v in self.evaluator.items() if k != "kind"}
        if self.evaluator.get("kind", "trading") == "trading":
            return evaluators.trading(rows, actions, price=self.target, holdout_ns=self.holdout_ns(), **ev)
        return evaluators.forecast(rows, actions, target=self.target, holdout_ns=self.holdout_ns(), **ev)


def load_table_tasks(folders: list[Path]) -> tuple[list[TableTask], list[str]]:
    """Every *.json task file in `folders`; files that do not parse are reported, not fatal."""
    tasks, errors, seen = [], [], set()
    for folder in folders:
        for f in sorted(Path(folder).glob("*.json")):
            try:
                cfg = json.loads(f.read_text(encoding="utf-8"))
                if cfg.get("disabled"):
                    continue
                t = TableTask(cfg, f.parent)
                if t.name in seen:
                    errors.append(f"{f}: a task named {t.name!r} already exists -- skipped")
                    continue
                seen.add(t.name)
                tasks.append(t)
            except Exception as exc:  # noqa: BLE001 -- one bad file must not hide the rest
                errors.append(f"{f}: {type(exc).__name__}: {exc}")
    return tasks, errors
