"""A project's SQL Server connection: read-only, and only the tables the user chose.

The operator's own Windows login is typically sysadmin on a local SQL Server. Letting the
swarm connect as that login would hand model-written SQL the power to drop or rewrite any
database on the box, so it never does. Instead there are two identities:

* **Setup** (listing databases and tables, provisioning) runs as the operator's Windows
  login, and only when they click for it on the Projects page.
* **Queries** run as a dedicated SQL login per project, ``freetoken_<slug>``, which setup
  creates with a random password and grants ``SELECT`` on exactly the chosen tables --
  nothing else, no roles. Read-only is therefore enforced **by SQL Server**, not by this code
  parsing the query: a hostile statement is refused by the server even if it got past us.

Defence in depth on top of that, each query is parsed as T-SQL (sqlglot) and must be one
SELECT touching only allowed tables; it runs inside a transaction that is always rolled
back, with a timeout and a row cap.

The reader's password lives in ``ui/backend/auth/secrets.json`` (owner-only ACL, git-ignored,
the same place as the token-signing key). It is never returned to the browser.
"""

from __future__ import annotations

import json
import re
import secrets as pysecrets
import string
import threading
import time
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from .auth import AUTH_DIR, _restrict_permissions

DRIVER = "ODBC Driver 17 for SQL Server"
SECRETS_FILE = AUTH_DIR / "secrets.json"
MAX_ROWS = 1000
MAX_CELL_CHARS = 2_000
QUERY_TIMEOUT_S = 60

_lock = threading.Lock()
# Server and database names go into a connection string; refuse anything that could break
# out of it. Real names on a local box never need these characters.
_UNSAFE_CS = re.compile(r"[;{}=\r\n]")
_WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.Command, exp.TruncateTable, exp.Use, exp.Set, exp.Grant, exp.Revoke,
)


class SqlError(ValueError):
    """A request that should become a 400."""


# ------------------------------------------------------------------------------------------
# secrets
# ------------------------------------------------------------------------------------------
def _read_secrets() -> dict:
    try:
        return json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_secret(key: str, value: str | None) -> None:
    with _lock:
        data = _read_secrets()
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        SECRETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _restrict_permissions(SECRETS_FILE)


def _password_for(login: str) -> str | None:
    return _read_secrets().get(f"sql:{login}")


def _new_password() -> str:
    # Satisfies Windows password policy (CHECK_POLICY = ON): upper, lower, digit, symbol.
    alphabet = string.ascii_letters + string.digits + "!#$%*+-.:?@^_~"
    while True:
        pw = "".join(pysecrets.choice(alphabet) for _ in range(32))
        if (any(c.islower() for c in pw) and any(c.isupper() for c in pw)
                and any(c.isdigit() for c in pw) and any(not c.isalnum() for c in pw)):
            return pw


# ------------------------------------------------------------------------------------------
# connections
# ------------------------------------------------------------------------------------------
def _check_name(value: str, what: str) -> str:
    value = (value or "").strip()
    if not value or _UNSAFE_CS.search(value):
        raise SqlError(f"invalid {what}: {value!r}")
    return value


def _admin(server: str, database: str | None = None):
    """Windows-authenticated connection -- setup only, never used for agent queries."""
    import pyodbc

    cs = (
        f"DRIVER={{{DRIVER}}};SERVER={_check_name(server, 'server')};"
        "Trusted_Connection=yes;TrustServerCertificate=yes;"
    )
    if database:
        cs += f"DATABASE={_check_name(database, 'database')};"
    try:
        return pyodbc.connect(cs, timeout=10, autocommit=True)
    except pyodbc.Error as exc:
        raise SqlError(f"cannot connect to {server}: {exc.args[-1] if exc.args else exc}") from None


def _reader(cfg: dict):
    """The project's restricted SQL login. Every agent query goes through this."""
    import pyodbc

    login = cfg.get("login")
    pw = _password_for(login) if login else None
    if not login or not pw:
        raise SqlError("this project's SQL access is not set up -- use Set up on the Projects page")
    cs = (
        f"DRIVER={{{DRIVER}}};SERVER={_check_name(cfg['server'], 'server')};"
        f"DATABASE={_check_name(cfg['database'], 'database')};"
        f"UID={login};PWD={{{pw.replace('}', '}}')}}};TrustServerCertificate=yes;"
    )
    try:
        con = pyodbc.connect(cs, timeout=10, autocommit=False)
    except pyodbc.Error as exc:
        raise SqlError(f"reader login failed: {exc.args[-1] if exc.args else exc}") from None
    con.timeout = QUERY_TIMEOUT_S
    return con


# ------------------------------------------------------------------------------------------
# setup (Windows login, operator-initiated)
# ------------------------------------------------------------------------------------------
def list_databases(server: str) -> list[str]:
    con = _admin(server)
    try:
        rows = con.execute(
            "SELECT name FROM sys.databases WHERE database_id > 4 AND state_desc = 'ONLINE' "
            "ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def list_tables(server: str, database: str) -> list[dict]:
    con = _admin(server, database)
    try:
        rows = con.execute(
            """
            SELECT s.name, t.name, SUM(p.rows)
            FROM sys.tables t
            JOIN sys.schemas s ON s.schema_id = t.schema_id
            LEFT JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id IN (0, 1)
            GROUP BY s.name, t.name
            UNION ALL
            SELECT s.name, v.name, NULL
            FROM sys.views v JOIN sys.schemas s ON s.schema_id = v.schema_id
            ORDER BY 1, 2
            """
        ).fetchall()
    finally:
        con.close()
    return [{"table": f"{r[0]}.{r[1]}", "rows": int(r[2]) if r[2] is not None else None} for r in rows]


def _login_name(slug: str) -> str:
    return "freetoken_" + re.sub(r"[^a-z0-9_]", "_", slug.lower())[:100]


def provision(slug: str, server: str, database: str, tables: list[str]) -> dict:
    """Create/refresh the project's reader login: SELECT on exactly `tables`, nothing else.

    Idempotent. Re-running with a different table list revokes what was dropped. Runs as the
    operator's Windows login -- the one privileged step, and only on request.
    """
    tables = sorted({_normalize_table(t) for t in tables})
    if not tables:
        raise SqlError("choose at least one table")
    available = {t["table"].lower(): t["table"] for t in list_tables(server, database)}
    missing = [t for t in tables if t.lower() not in available]
    if missing:
        raise SqlError(f"not found in {database}: {', '.join(missing)}")
    tables = [available[t.lower()] for t in tables]

    login = _login_name(slug)
    pw = _password_for(login) or _new_password()

    admin = _admin(server)
    try:
        exists = admin.execute("SELECT 1 FROM sys.server_principals WHERE name = ?", login).fetchone()
        stmt = "ALTER LOGIN {l} WITH PASSWORD = {p}" if exists else (
            "CREATE LOGIN {l} WITH PASSWORD = {p}, CHECK_POLICY = ON, DEFAULT_DATABASE = {d}"
        )
        # Names and the password are quoted by SQL Server itself (QUOTENAME), never spliced.
        admin.execute(
            "DECLARE @s nvarchar(max) = REPLACE(REPLACE(REPLACE(?, '{l}', QUOTENAME(?)), "
            "'{p}', QUOTENAME(?, '''')), '{d}', QUOTENAME(?)); EXEC sp_executesql @s;",
            stmt, login, pw, database,
        )
    finally:
        admin.close()
    _write_secret(f"sql:{login}", pw)

    db = _admin(server, database)
    try:
        if not db.execute("SELECT 1 FROM sys.database_principals WHERE name = ?", login).fetchone():
            db.execute(
                "DECLARE @s nvarchar(max) = N'CREATE USER ' + QUOTENAME(?) + N' FOR LOGIN ' + "
                "QUOTENAME(?); EXEC sp_executesql @s;",
                login, login,
            )
        # Drop every existing grant, then grant exactly the chosen set -- so the list on the
        # Projects page is the whole truth, not an addition to whatever was there before.
        granted = db.execute(
            """
            SELECT s.name + '.' + o.name
            FROM sys.database_permissions p
            JOIN sys.database_principals u ON u.principal_id = p.grantee_principal_id
            JOIN sys.objects o ON o.object_id = p.major_id
            JOIN sys.schemas s ON s.schema_id = o.schema_id
            WHERE u.name = ? AND p.class = 1
            """,
            login,
        ).fetchall()
        for (obj,) in granted:
            schema, name = obj.split(".", 1)
            db.execute(
                "DECLARE @s nvarchar(max) = N'REVOKE ALL ON OBJECT::' + QUOTENAME(?) + N'.' + "
                "QUOTENAME(?) + N' FROM ' + QUOTENAME(?); EXEC sp_executesql @s;",
                schema, name, login,
            )
        for t in tables:
            schema, name = t.split(".", 1)
            db.execute(
                "DECLARE @s nvarchar(max) = N'GRANT SELECT ON OBJECT::' + QUOTENAME(?) + N'.' + "
                "QUOTENAME(?) + N' TO ' + QUOTENAME(?); EXEC sp_executesql @s;",
                schema, name, login,
            )
    finally:
        db.close()

    cfg = {"server": server, "database": database, "tables": tables, "login": login}
    return {"config": cfg, "verification": verify(cfg)}


def remove(slug: str, server: str) -> None:
    """Drop the project's reader login (its database users are orphaned harmlessly)."""
    login = _login_name(slug)
    admin = _admin(server)
    try:
        # ODBC pools connections, so a reader session can outlive its close() -- and SQL Server
        # refuses to drop a login that is "currently logged in". End those sessions first.
        for (spid,) in admin.execute(
            "SELECT session_id FROM sys.dm_exec_sessions WHERE login_name = ?", login
        ).fetchall():
            admin.execute(f"KILL {int(spid)}")
        if admin.execute("SELECT 1 FROM sys.server_principals WHERE name = ?", login).fetchone():
            admin.execute(
                "DECLARE @s nvarchar(max) = N'DROP LOGIN ' + QUOTENAME(?); EXEC sp_executesql @s;",
                login,
            )
    finally:
        admin.close()
    _write_secret(f"sql:{login}", None)


# ------------------------------------------------------------------------------------------
# verification: ask SQL Server, as the reader, what it could actually do
# ------------------------------------------------------------------------------------------
def verify(cfg: dict) -> dict:
    con = _reader(cfg)
    try:
        cur = con.cursor()
        cur.execute("SELECT IS_SRVROLEMEMBER('sysadmin'), HAS_PERMS_BY_NAME(DB_NAME(), 'DATABASE', 'CREATE TABLE')")
        sysadmin, create = cur.fetchone()
        per_table = []
        for t in cfg["tables"]:
            cur.execute(
                "SELECT HAS_PERMS_BY_NAME(?, 'OBJECT', 'SELECT'), HAS_PERMS_BY_NAME(?, 'OBJECT', 'INSERT'), "
                "HAS_PERMS_BY_NAME(?, 'OBJECT', 'UPDATE'), HAS_PERMS_BY_NAME(?, 'OBJECT', 'DELETE')",
                t, t, t, t,
            )
            sel, ins, upd, dele = cur.fetchone()
            per_table.append({"table": t, "select": bool(sel), "write": bool(ins or upd or dele)})
        con.rollback()
    finally:
        con.close()
    read_only = not sysadmin and not create and not any(p["write"] for p in per_table)
    return {
        "read_only": read_only,
        "can_read_all": all(p["select"] for p in per_table),
        "sysadmin": bool(sysadmin),
        "can_create_tables": bool(create),
        "tables": per_table,
    }


# ------------------------------------------------------------------------------------------
# agent queries
# ------------------------------------------------------------------------------------------
def _normalize_table(t: str) -> str:
    parts = [p.strip().strip("[]") for p in t.split(".") if p.strip()]
    if len(parts) == 1:
        parts = ["dbo", parts[0]]
    if len(parts) != 2:
        raise SqlError(f"table must be schema.name: {t!r}")
    return f"{parts[0]}.{parts[1]}"


def _check_query(sql: str, cfg: dict) -> None:
    try:
        statements = [s for s in sqlglot.parse(sql, read="tsql") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise SqlError(f"could not parse the query: {exc}") from None
    if len(statements) != 1:
        raise SqlError("send exactly one SELECT statement")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.With, exp.Subquery)):
        raise SqlError("only SELECT queries are allowed")
    ctes = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
    allowed = {t.lower() for t in cfg["tables"]}
    for node in stmt.walk():
        if isinstance(node, _WRITE_NODES) or isinstance(node, exp.Into):
            raise SqlError(f"{type(node).__name__} is not allowed -- this connection is read-only")
        if isinstance(node, exp.Table):
            name = node.name.lower()
            if not node.args.get("db") and name in ctes:
                continue
            if node.args.get("catalog"):
                cat = node.catalog.lower()
                if cat != cfg["database"].lower():
                    raise SqlError(f"only database {cfg['database']} is available")
            qualified = f"{(node.db or 'dbo')}.{node.name}".lower()
            if qualified not in allowed:
                raise SqlError(
                    f"table {node.db + '.' if node.db else ''}{node.name} is not available to this "
                    f"project. Allowed: {', '.join(cfg['tables'])}"
                )


def _cell(v: Any) -> Any:
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = str(v)
    return s if len(s) <= MAX_CELL_CHARS else s[:MAX_CELL_CHARS] + "..."


def query(cfg: dict, sql: str, max_rows: int = MAX_ROWS) -> dict:
    if not sql or not sql.strip():
        raise SqlError("empty query")
    _check_query(sql, cfg)
    import pyodbc

    con = _reader(cfg)
    t0 = time.time()
    try:
        cur = con.cursor()
        cur.execute(sql)
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchmany(max_rows + 1)
    except pyodbc.Error as exc:
        raise SqlError(str(exc.args[-1] if exc.args else exc)) from None
    finally:
        # Always roll back: nothing an agent runs is ever committed, whatever got through.
        try:
            con.rollback()
        finally:
            con.close()
    return {
        "columns": cols,
        "rows": [[_cell(v) for v in r] for r in rows[:max_rows]],
        "row_count": min(len(rows), max_rows),
        "truncated": len(rows) > max_rows,
        "seconds": round(time.time() - t0, 3),
    }


def describe(cfg: dict, table: str) -> dict:
    t = _normalize_table(table)
    if t.lower() not in {x.lower() for x in cfg["tables"]}:
        raise SqlError(f"table {t} is not available to this project")
    schema, name = t.split(".", 1)
    con = _reader(cfg)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION",
            schema, name,
        )
        cols = [{"name": r[0], "type": r[1], "nullable": r[2] == "YES"} for r in cur.fetchall()]
        cur.execute(f"SELECT TOP 5 * FROM [{schema.replace(']', ']]')}].[{name.replace(']', ']]')}]")
        sample = [[_cell(v) for v in r] for r in cur.fetchall()]
        con.rollback()
    finally:
        con.close()
    return {"table": t, "columns": cols, "sample": sample}


# ------------------------------------------------------------------------------------------
# export: one allowed table -> parquet files in the project's data folder
# ------------------------------------------------------------------------------------------
# Streams through the READER login, so only an allowed table can be exported and nothing is
# written to the server. Rows arrive in batches and go out as parquet row groups, rolling to
# a new file every `rows_per_file`, so a table far bigger than RAM exports in bounded memory.
EXPORT_BATCH = 50_000
EXPORT_MANIFEST = ".export.json"  # dot-file: marks the folder as ONE dataset for the catalog


def _arrow_type(sql_type: str, precision, scale):
    """SQL Server column type -> arrow type, from the server's own metadata.

    Declared up front rather than inferred per batch: inference would call a column that is
    all NULL in the first batch `null`, then fail when a later batch has values.
    """
    import pyarrow as pa

    t = sql_type.lower()
    return {
        "bigint": pa.int64(), "int": pa.int32(), "smallint": pa.int16(),
        "tinyint": pa.int16(),  # tinyint is unsigned 0-255; int8 would overflow
        "bit": pa.bool_(), "float": pa.float64(), "real": pa.float32(),
        "date": pa.date32(), "time": pa.time64("us"),
        "datetime": pa.timestamp("us"), "datetime2": pa.timestamp("us"),
        "smalldatetime": pa.timestamp("us"), "datetimeoffset": pa.timestamp("us", tz="UTC"),
        "money": pa.decimal128(19, 4), "smallmoney": pa.decimal128(10, 4),
        "binary": pa.binary(), "varbinary": pa.binary(), "image": pa.binary(),
        "timestamp": pa.binary(), "rowversion": pa.binary(),
    }.get(t) or (
        pa.decimal128(int(precision or 18), int(scale or 0)) if t in ("decimal", "numeric") else pa.string()
    )


def export_table(cfg: dict, table: str, dest, where: str | None = None,
                 rows_per_file: int = 1_000_000, progress=None) -> dict:
    """Export `table` (optionally filtered) to `dest/part-NNNNN.parquet`. Returns a summary.

    Written to a temporary sibling folder and swapped in at the end, so a failed or
    interrupted export never leaves a half-written dataset where agents would read it.
    """
    import datetime as dt
    import shutil
    import uuid
    from decimal import Decimal
    from pathlib import Path as _P

    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyodbc

    t = _normalize_table(table)
    if t.lower() not in {x.lower() for x in cfg["tables"]}:
        raise SqlError(f"table {t} is not available to this project")
    schema_name, name = t.split(".", 1)
    ident = f"[{schema_name.replace(']', ']]')}].[{name.replace(']', ']]')}]"
    select = f"SELECT * FROM {ident}"
    if where and where.strip():
        select += f" WHERE {where.strip()}"
    # The filter is user text, so the WHOLE statement goes through the same read-only check
    # as an agent query: one SELECT, allowed tables only.
    _check_query(select, cfg)

    dest = _P(dest)
    tmp = dest.parent / f".{dest.name}.tmp-{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=False)

    con = _reader(cfg)
    con.timeout = 0  # an export legitimately runs long; the reader login still bounds WHAT
    started = time.time()
    written = files = 0
    writer = None
    file_rows = 0
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT COLUMN_NAME, DATA_TYPE, NUMERIC_PRECISION, NUMERIC_SCALE "
            "FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
            "ORDER BY ORDINAL_POSITION",
            schema_name, name,
        )
        cols = cur.fetchall()
        if not cols:
            raise SqlError(f"cannot read the columns of {t}")
        schema = pa.schema([pa.field(c[0], _arrow_type(c[1], c[2], c[3])) for c in cols])
        stringy = [pa.types.is_string(f.type) for f in schema]

        total = None
        try:
            cur.execute(f"SELECT COUNT_BIG(*) FROM {ident}" + (f" WHERE {where.strip()}" if where and where.strip() else ""))
            total = int(cur.fetchone()[0])
        except pyodbc.Error:
            pass  # progress then shows rows only
        if progress:
            progress(0, total, 0)

        cur.execute(select)
        while True:
            batch = cur.fetchmany(EXPORT_BATCH)
            if not batch:
                break
            columns = list(zip(*batch))
            arrays = []
            for i, field in enumerate(schema):
                vals = list(columns[i])
                if stringy[i]:
                    # uniqueidentifier, xml, sql_variant... arrive as non-str objects.
                    vals = [None if v is None else str(v) for v in vals]
                elif pa.types.is_timestamp(field.type) and field.type.tz:
                    vals = [v.astimezone(dt.timezone.utc) if isinstance(v, dt.datetime) and v.tzinfo else v for v in vals]
                elif pa.types.is_decimal(field.type):
                    vals = [None if v is None else Decimal(v) for v in vals]
                arrays.append(pa.array(vals, type=field.type))
            chunk = pa.Table.from_arrays(arrays, schema=schema)

            offset = 0
            while offset < chunk.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(tmp / f"part-{files:05d}.parquet", schema, compression="zstd")
                    files += 1
                    file_rows = 0
                take = min(chunk.num_rows - offset, rows_per_file - file_rows)
                writer.write_table(chunk.slice(offset, take))
                offset += take
                file_rows += take
                written += take
                if file_rows >= rows_per_file:
                    writer.close()
                    writer = None
            if progress:
                progress(written, total, files)
        if writer is not None:
            writer.close()
            writer = None
        if files == 0:  # empty result: still leave a readable (empty) dataset
            pq.write_table(schema.empty_table(), tmp / "part-00000.parquet")
            files = 1
        manifest = {
            "source": f"{cfg['server']}/{cfg['database']}/{t}",
            "where": where or None,
            "rows": written,
            "files": files,
            "columns": [{"name": f.name, "type": str(f.type)} for f in schema],
            "exported_at": time.time(),
            "seconds": round(time.time() - started, 1),
        }
        (tmp / EXPORT_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except pyodbc.Error as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SqlError(str(exc.args[-1] if exc.args else exc)) from None
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        if writer is not None:
            writer.close()
        try:
            con.rollback()
        finally:
            con.close()

    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return {**manifest, "path": str(dest)}
