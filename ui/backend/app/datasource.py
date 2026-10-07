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
from typing import Any, Callable

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


_NO_TABLE = re.compile(r"Table with name (\S+) does not exist")


def sql_error(exc: Exception, views: list[str]) -> str:
    """A DuckDB error as the one line an agent reads -- with the part that lets it fix the query.

    Only the first line used to be kept, which for a mistyped name is "Table with name
    sql_exports_dbo_gex_bar10s does not exist!" and nothing else: DuckDB's own "Did you mean"
    and "Candidate bindings" lines were dropped, so the agent guessed again (bug #103). A
    missing table is answered from `views` (the names this query could have used) rather than
    DuckDB's guess, which offers its system tables ("pg_tables") when nothing is close."""
    import difflib

    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    msg = lines[0] if lines else "the query failed"
    m = _NO_TABLE.search(msg)
    if m:
        wanted = m.group(1).strip('"').lower()
        close = difflib.get_close_matches(wanted, [v.lower() for v in views], n=1, cutoff=0.6)
        name = next((v for v in views if close and v.lower() == close[0]), None)
        shown = ", ".join(views[:40]) + (f", ... ({len(views)} in all)" if len(views) > 40 else "")
        return (msg + (f' Did you mean "{name}"?' if name else "")
                + (f" Tables you can query: {shown}" if views else " There is no table to query here."))
    extra = next((ln for ln in lines[1:4] if ln.startswith(("Did you mean", "Candidate bindings"))), "")
    return f"{msg} {extra[:400]}".strip()


_NO_COLUMN = re.compile(r'Referenced column "([^"]+)" not found|does not have a column named "([^"]+)"')
# The catalog tables that list the views, each by its `table_name`.
_CATALOG_TABLES = frozenset({"tables", "views"})
# What an agent calls that `table_name` when it lists the views: list_data shows each one under
# the key "view", so `SELECT view FROM information_schema.tables WHERE table_name LIKE 'fc_%'`
# was sent three times and refused with DuckDB's only candidate, "is_insertable_into" (bug #405).
_NAME_ALIASES = frozenset({"view", "views", "view_name", "viewname", "name", "table", "tablename",
                           "dataset", "dataset_name", "relname"})
_HINT_COLUMNS = 150


def _missing_column(exc: Exception) -> str | None:
    m = _NO_COLUMN.search(str(exc))
    return (m.group(1) or m.group(2)) if m else None


def _reads_catalog(stmt: exp.Expression) -> bool:
    return any(t.db.lower() == "information_schema" for t in stmt.find_all(exp.Table))


def _catalog_fix(stmt: exp.Expression, exc: Exception) -> tuple[exp.Expression, str] | None:
    """A query listing the views (information_schema.tables / .views) that named the view-name
    column by an obvious other name ("view", "name") -> the same query on `table_name`, and the
    note telling the agent so. Anything else -> None: the error stands."""
    missing = _missing_column(exc)
    if not missing or missing.lower() not in _NAME_ALIASES:
        return None
    if not any(t.db.lower() == "information_schema" and t.name.lower() in _CATALOG_TABLES
               for t in stmt.find_all(exp.Table)):
        return None
    fixed = stmt.copy()
    cols = [c for c in fixed.find_all(exp.Column) if c.name.lower() == missing.lower()]
    if not cols:
        return None
    for c in cols:
        c.set("this", exp.to_identifier("table_name"))
    return fixed, (f'information_schema has no column "{missing}": a view\'s name is in table_name, so the '
                   f'query was run with table_name in its place. information_schema.tables lists every view '
                   f'you can query (data files and fc_* feature views alike).')


def column_hint(con: duckdb.DuckDBPyConnection, stmt: exp.Expression, exc: Exception, views: list[str]) -> str:
    """For a column that does not exist: the real columns of each table the query reads.

    DuckDB names at most five "candidate bindings", picked by spelling -- for `view` read from
    information_schema.tables its one candidate was "is_insertable_into" -- so the agent guessed
    again. The columns themselves end the guessing. A query of information_schema is an agent
    looking for the tables, so it is told those too."""
    import difflib

    missing = _missing_column(exc)
    out: list[str] = []
    if missing:
        by_lower = {v.lower(): v for v in views}
        ctes = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
        seen: set[str] = set()
        for t in stmt.find_all(exp.Table):
            name, db = t.name, t.db
            if db.lower() == "information_schema":
                target = label = f"information_schema.{name.lower()}"
            elif not db and name.lower() in by_lower and name.lower() not in ctes:
                label = by_lower[name.lower()]
                target = '"' + label.replace('"', '""') + '"'
            else:
                continue
            if label in seen:
                continue
            seen.add(label)
            try:
                cols = [r[0] for r in con.execute(f"DESCRIBE {target}").fetchall()]
            except duckdb.Error:
                continue
            close = difflib.get_close_matches(missing.lower(), [c.lower() for c in cols], n=1, cutoff=0.6)
            mean = next((c for c in cols if close and c.lower() == close[0]), None)
            shown = ", ".join(cols[:_HINT_COLUMNS]) + (f", ... ({len(cols)} in all)" if len(cols) > _HINT_COLUMNS else "")
            out.append((f'Did you mean "{mean}"? ' if mean else "") + f"Columns of {label}: {shown}.")
            if len(seen) >= 4:
                break
    if _reads_catalog(stmt) and views:
        shown = ", ".join(views[:60]) + (f", ... ({len(views)} in all)" if len(views) > 60 else "")
        out.append(f"Tables you can query: {shown}.")
    return " ".join(out)


def run_select(con: duckdb.DuckDBPyConnection, stmt: exp.Expression, max_rows: int,
               views: list[str] | Callable[[], list[str]]) -> tuple[list[str], list[tuple], str | None]:
    """Run a checked SELECT on a prepared connection -> (columns, up to max_rows + 1 rows, note).

    An obvious intent that DuckDB refuses on a name (listing the views by a `view` column) is
    run as meant, with a note saying what was changed; any other failure is a DataError that
    carries what fixes the query (the tables there are, the columns a table has)."""
    try:
        cur = con.execute(stmt.sql(dialect="duckdb"))
        return [d[0] for d in (cur.description or [])], cur.fetchmany(max_rows + 1), None
    except duckdb.Error as exc:
        fix = _catalog_fix(stmt, exc)
        if fix is not None:
            try:
                cur = con.execute(fix[0].sql(dialect="duckdb"))
                return [d[0] for d in (cur.description or [])], cur.fetchmany(max_rows + 1), fix[1]
            except duckdb.Error:
                pass  # the rewrite did not help: report the query as the agent wrote it
        names = views() if callable(views) else views  # only a failure needs them
        hint = column_hint(con, stmt, exc, names)
        raise DataError(f"{sql_error(exc, names)} {hint}".strip()) from None


def query(data_dir: str, sql: str, max_rows: int = MAX_ROWS) -> dict:
    """Run one read-only SELECT against the project's files."""
    if not sql or not sql.strip():
        raise DataError("empty query")
    stmt = _check_select(sql)
    _absolutize(stmt, Path(data_dir).resolve())
    con = _connect(data_dir)
    t0 = time.time()
    try:
        cols, rows, note = run_select(con, stmt, max_rows, lambda: [i["view"] for i in catalog(data_dir)])
    finally:
        con.close()
    truncated = len(rows) > max_rows
    return {
        "columns": cols,
        "rows": [[_cell(v) for v in r] for r in rows[:max_rows]],
        "row_count": min(len(rows), max_rows),
        "truncated": truncated,
        "seconds": round(time.time() - t0, 3),
        **({"note": note} if note else {}),
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
