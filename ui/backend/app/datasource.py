"""SQL over a project's data folder: parquet, CSV and JSON files, queried in place.

Agents point at a project's data directory -- which can be a whole data drive -- and ask
questions of it in SQL. DuckDB reads parquet and CSV directly, so there is no import step
and a 20 GB parquet set is queried without being loaded.

**Confinement is enforced by DuckDB, not by inspecting the query.** Every query runs on a
fresh in-memory connection configured with:

* ``allowed_directories = [data_dir]`` plus ``enable_external_access = false`` -- files
  outside the project's folder cannot be opened at all, whatever the SQL says;
* ``file_search_path = data_dir`` -- so ``SELECT * FROM 'prices/2024.parquet'`` works with
  a path relative to the project, which is how a model naturally writes it;
* ``lock_configuration = true`` -- the query cannot SET its way back out.

On top of that the statement must parse as a single read-only SELECT (sqlglot), because an
allowed directory is writable too and ``COPY ... TO`` would otherwise overwrite the user's
own data. The connection is in-memory, so nothing it creates outlives the query.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

MAX_ROWS = 500
MAX_CELL_CHARS = 2_000
QUERY_TIMEOUT_S = 60
# Files an agent can query, and the DuckDB reader for each.
_READERS = {
    ".parquet": "read_parquet",
    ".csv": "read_csv_auto",
    ".tsv": "read_csv_auto",
    ".json": "read_json_auto",
    ".jsonl": "read_json_auto",
    ".ndjson": "read_json_auto",
}
# Statement types that change state. A SELECT is the only thing an agent may run here.
_WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.Copy, exp.Command, exp.Attach, exp.Detach, exp.Pragma, exp.Set, exp.Use,
    exp.TruncateTable,
)


class DataError(ValueError):
    """A request that should become a 400."""


def _view_name(rel: Path) -> str:
    """'prices/2024 daily.parquet' -> 'prices_2024_daily' -- a table name an agent can type."""
    stem = str(rel.with_suffix("")).replace("\\", "/")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_").lower()
    return name if name and not name[0].isdigit() else f"t_{name}"


def catalog(data_dir: str, limit: int = 400) -> list[dict]:
    """Queryable files under the data directory, each with the view name it is exposed as."""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    out: list[dict] = []
    # A folder holding a `.export.json` manifest (a SQL table exported to parquet parts) is ONE
    # dataset, exposed as one view over all its parts -- not part_00000, part_00001, ... which
    # would make an agent union them by hand and silently miss any it did not list.
    datasets: set[Path] = set()
    for manifest in sorted(root.rglob(".export.json")):
        folder = manifest.parent
        rel = folder.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue  # an in-progress export's temp folder
        try:
            info = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        datasets.add(folder)
        out.append({
            "path": f"{rel.as_posix()}/*.parquet",
            "view": _view_name(rel),
            "bytes": sum(p.stat().st_size for p in folder.glob("*.parquet")),
            "format": "parquet",
            "dataset": True,
            "rows": info.get("rows"),
            "source": info.get("source"),
        })
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in _READERS or not path.is_file():
            continue
        # Skip dot-directories (caches, .git) -- not data anyone meant to share.
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.parent in datasets:
            continue  # already covered by its dataset's single view
        out.append(
            {
                "path": rel.as_posix(),
                "view": _view_name(rel),
                "bytes": path.stat().st_size,
                "format": path.suffix.lower().lstrip("."),
            }
        )
        if len(out) >= limit:
            break
    return out


def _absolutize(stmt: exp.Expression, root: Path) -> None:
    """Rewrite relative file literals ('prices/2024.parquet') to absolute paths in the folder.

    DuckDB checks a literal path against allowed_directories BEFORE resolving it through
    file_search_path, so a relative path -- the natural way to name a project file -- was
    refused as if it were an escape. Resolving here keeps that natural form working, and the
    containment check below means a '../' literal is rejected rather than rewritten.
    DuckDB's own guard still applies to the result.
    """
    def resolve(value: str) -> str:
        target = (root / value).resolve()
        if target != root and not target.is_relative_to(root):
            raise DataError(f"path escapes the project data folder: {value!r}")
        return target.as_posix()

    # `FROM 'prices/2024.parquet'` parses as a TABLE named 'prices/2024.parquet', not a string
    # literal, so it needs its own rewrite: into an explicit reader call on the resolved path.
    for table in list(stmt.find_all(exp.Table)):
        name = table.name
        suffix = Path(name).suffix.lower()
        if suffix not in _READERS or "://" in name:
            continue
        path = name if (Path(name).is_absolute() or re.match(r"^[A-Za-z]:", name)) else resolve(name)
        call = exp.Anonymous(this=_READERS[suffix], expressions=[exp.Literal.string(path)])
        alias = table.args.get("alias")
        table.replace(exp.Alias(this=call, alias=alias.this) if alias else call)

    for lit in list(stmt.find_all(exp.Literal)):
        if not lit.is_string:
            continue
        value = lit.this
        if Path(value).suffix.lower() not in _READERS:
            continue
        if "://" in value or Path(value).is_absolute() or re.match(r"^[A-Za-z]:", value):
            continue  # absolute or remote: leave for DuckDB to refuse
        target = (root / value).resolve()
        if target != root and not target.is_relative_to(root):
            raise DataError(f"path escapes the project data folder: {value!r}")
        lit.replace(exp.Literal.string(target.as_posix()))


def _check_select(sql: str) -> exp.Expression:
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise DataError(f"could not parse the query: {exc}") from None
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise DataError("send exactly one SELECT statement")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.With, exp.Subquery)):
        raise DataError("only SELECT queries are allowed on project data")
    for node in stmt.walk():
        if isinstance(node, _WRITE_NODES):
            raise DataError(f"{type(node).__name__} is not allowed -- project data is read-only")
        if isinstance(node, exp.Into):
            raise DataError("SELECT ... INTO is not allowed -- project data is read-only")
    return stmt


def _connect(data_dir: str) -> duckdb.DuckDBPyConnection:
    root = str(Path(data_dir).resolve())
    con = duckdb.connect(":memory:")
    con.execute("SET threads = 4")
    # Views first (they need to name absolute paths), then shut the filesystem down.
    for item in catalog(root):
        reader = _READERS[Path(item["path"]).suffix.lower()]
        full = str(Path(root) / item["path"]).replace("'", "''")
        con.execute(f'CREATE OR REPLACE VIEW "{item["view"]}" AS SELECT * FROM {reader}(\'{full}\')')
    con.execute("SET allowed_directories = ?", [[root]])
    con.execute("SET file_search_path = ?", [root])
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def _cell(v: Any) -> Any:
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = str(v)
    return s if len(s) <= MAX_CELL_CHARS else s[:MAX_CELL_CHARS] + "..."


def query(data_dir: str, sql: str, max_rows: int = MAX_ROWS) -> dict:
    """Run one read-only SELECT against the project's files."""
    if not sql or not sql.strip():
        raise DataError("empty query")
    stmt = _check_select(sql)
    _absolutize(stmt, Path(data_dir).resolve())
    con = _connect(data_dir)
    t0 = time.time()
    try:
        cur = con.execute(stmt.sql(dialect="duckdb"))
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchmany(max_rows + 1)
    except duckdb.Error as exc:
        raise DataError(str(exc).splitlines()[0]) from None
    finally:
        con.close()
    truncated = len(rows) > max_rows
    return {
        "columns": cols,
        "rows": [[_cell(v) for v in r] for r in rows[:max_rows]],
        "row_count": min(len(rows), max_rows),
        "truncated": truncated,
        "seconds": round(time.time() - t0, 3),
    }


def describe(data_dir: str, view: str) -> dict:
    """Columns and a few sample rows of one file (by its view name or relative path)."""
    items = catalog(data_dir)
    item = next((i for i in items if i["view"] == view or i["path"] == view), None)
    if item is None:
        raise DataError(f"no data file {view!r}; call list_data to see what exists")
    con = _connect(data_dir)
    try:
        cols = con.execute(f'DESCRIBE "{item["view"]}"').fetchall()
        count = con.execute(f'SELECT count(*) FROM "{item["view"]}"').fetchone()[0]
        sample = con.execute(f'SELECT * FROM "{item["view"]}" LIMIT 5').fetchall()
    except duckdb.Error as exc:
        raise DataError(str(exc).splitlines()[0]) from None
    finally:
        con.close()
    return {
        **item,
        "row_count": count,
        "columns": [{"name": c[0], "type": c[1]} for c in cols],
        "sample": [[_cell(v) for v in r] for r in sample],
    }
