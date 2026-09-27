# Task servers (data/action MCPs)

A **task server** is an MCP server that owns a problem the FreeSwarm swarm can work on:

1. **It serves data**: time-aligned rows of signals, one per time step, with one column as the
   **target** (a price, a power price, a quantity to forecast).
2. **It takes and manages actions**: a strategy decides one number per row. The server turns
   those numbers into whatever they drive (a position with its trades, a battery's charge, a
   forecast) under its rules, such as bounds, costs and forced exits.
3. **It values the result**: a **curve** (daily returns, daily profit, …) and a **valuation** per
   segment (the in-sample period agents learn on, and the holdout they never see), with a `score`
   that ranks candidates.

The swarm writes strategy code against a task. The FreeSwarm harness runs that code in its
sandbox, proves it never uses the future, and ranks it by the server's score. How a server does
any of this inside (polars, numpy, pandas, SQL, a simulator, another language) is its own business.
**The interface below is the only thing the harness depends on.**

```
mcp/
  README.md            this file: the interface, and how to build, run and secure a server
  taskkit/             OPTIONAL helpers for Python servers (MCP wiring, OAuth, a polars Task base)
  gex/                 the GEX server: SPY 10-second GEX bars (git-ignored, project-specific)
  test/battery/        example: run a home battery on a power grid (custom simulation + valuation)
  test/tables/         example: tasks from JSON files, no code (bike rentals forecast)
  tests/               tests for all of the above (python -m pytest mcp/tests -q)
```

| server | what it serves | action | valuation | transport |
| --- | --- | --- | --- | --- |
| `gex` | SPY 10-s bars + options positioning (146 columns) | position, or buy/sell/hold orders; bounded, flat at each close | daily returns; Sharpe, Sortino, Calmar, …, segment matching, long/short balance | HTTP :8200 + OAuth |
| `battery-demo` | hourly grid prices, weather, day-ahead forecasts | battery power for the next hour (charge/sell) | daily profit; share of perfect-foresight profit | stdio (or HTTP + OAuth) |
| `test-tables` | any table described by a JSON file | position or forecast | trading or forecast skill | stdio (or HTTP + OAuth) |

---

## The interface

An MCP server that exposes these tools with these arguments and **JSON object** results is a
task server. Files are exchanged **by path**, so the server runs on the same machine as the
control plane (or shares its disk).

### Agent-facing tools (in-sample only; the swarm may call these)

| tool | arguments | result |
| --- | --- | --- |
| `task_list` | none | `{"tasks": [{name, title, target, score:{name, higher_is_better}, rows, first, holdout_from, action}], "errors": [...]}` |
| `task_describe` | `task, target, value_function` | `{name, title, description, brief, key:"t", target, target_options:[...], shape:{rows, columns, first, last, step_s, ...}, action:{kind, min, max, initial, description, ...}, valuation:{summary, ...}, value_functions:[{name, title, description, higher_is_better}], value_function, score:{name, higher_is_better}, display_tz, rows, in_sample_rows, first, last_in_sample, holdout_from, cuts:[iso...], columns:[{name, dtype, role, description}], version, ahead_columns}` |
| `task_sample_rows` | `task, limit, offset, columns` | `{"rows": [records], offset, in_sample_rows}`, **never holdout rows** |
| `task_column_stats` | `task` | `{"columns": {name: {count, missing, mean, std, min, 25%, 50%, 75%, max}}, in_sample_rows}` |
| `task_query` | `task, sql, limit` | a read-only `SELECT` over the in-sample rows as the table `rows`: `{columns, rows, returned}` |

### Harness-only tools (FreeSwarm hides and refuses `harness_*` on every route agents use)

| tool | arguments | result |
| --- | --- | --- |
| `harness_export_rows` | `task, path, until, target, value_function` | writes rows with `t < until` (all rows without it) as parquet to `path`, plus `task.json` (the description **without** `cuts`) beside it; `{path, rows, until, version}` |
| `harness_evaluate` | `task, actions_path, target, value_function` | manages and values the actions (see below), or `{"problem": "why they cannot be scored"}` |
| `harness_actions` | `task, actions_path, start, end, limit, target` | the **drill-down** of a window (a day of the curve): `{"bars": {"kind": "ohlc"\|"line", "columns": [...], "rows": [[t, ...]], "tz"}, "state": [[t, value]], "state_kind": "...", "events": [records]}`: the target as candles or a line, the managed state (a position, a charge) and what the actions did (trades, a charge schedule, …) |
| `harness_leak_scan` | `task, top, target` | `{"columns": [{column, change_vs_current_move, change_vs_next_move, suspect}], "suspects": [...], "declared_ahead": [...]}` |

**`target`** (optional everywhere): value the task on another of its `target_options`, the
column the operator chose for the project (for example GEX on `Open` instead of `Close`, or on any of
its 145 numeric fields). A server refuses anything else and decides how to value what it accepts
(GEX: returns for a positive series, P&L in the target's units for a signed one). Without it the
task's default target applies.

**`value_function`** (optional): rank on another of the task's `value_functions`, the one the
operator chose for the project. Each value function has a name, a title, a description and a
direction, and names the valuation statistic that becomes each segment's `score`. The server
still reports every statistic; the choice decides which one ranks. The first listed is the
default. GEX offers Sharpe, smooth Sharpe, Sortino, Calmar, total return and segment matching;
the battery offers share of perfect foresight, total profit and the Sharpe of daily profit.

**`shape`, `valuation`, `display_tz`** in `task_describe` are for people. The console shows them
when a project is connected: the data's size and span, what the value function measures and
with which settings, and the time zone the drill-down is drawn in.

### Files

- **Rows** (written by `harness_export_rows`): parquet, a timestamp column `t` (UTC, naive,
  sorted, one row per time step) and any other columns. Every value in a row must be **known at
  that row's time**.
- **Actions** (read by `harness_evaluate` / `harness_actions`): parquet with `t` (timestamp) and
  `pos` (number); the strategy writes it with `ft.report_actions(...)`. Align it **as of**: a row's
  action is the latest one at or before its `t` (in an "order" mode, an order applies once, at the
  first row at or after its `t`). Before the first action, `action.initial`. NaN means "no action".

### The valuation (`harness_evaluate`)

```json
{
  "segments": {
    "in_sample": {"score": 1.21, "note": "...only when score is null", "...": "any valuation numbers"},
    "holdout":   {"score": 0.84, "...": "..."}
  },
  "curve": [["2024-01-02", 0.0013], ["2024-01-03", -0.0004]],
  "curve_kind": "returns",
  "diagnostics": {"in_sample": {"...": "free-form"}, "holdout": {"...": "free-form"}},
  "notes": "In-sample: ... (told to the agent -- in-sample numbers only)",
  "unranked": null,
  "score_name": "sharpe",
  "higher_is_better": true
}
```

- `curve_kind` is `"returns"` (compounded into equity) or `"additive"` (summed, e.g. profit).
- A `null` score means "not rankable" and `note` says why. `unranked` (a reason string) keeps a
  candidate off the leaderboard, for example a one-sided trading strategy.
- The harness ranks on the **weaker** of the in-sample and holdout scores (robust ranking), or on
  the holdout alone. **Agents only ever see in-sample numbers and `notes`**, so never put holdout
  figures in `notes`.

### Rules the harness relies on

1. **The action decided at row t takes effect from row t to row t+1**, and is valued on what
   happens after it.
2. The holdout starts at **midnight UTC** of `holdout_from`. Days are UTC calendar days.
3. Exports cut **strictly before** `until`, and `task.json` in the export omits the cuts.
4. Agent-facing tools never return a holdout row or number.
5. A failure is an exception or `{"error": "..."}`. The harness reports it as the server's.

---

## Building a server

**Any language:** implement the nine tools above over MCP (stdio or streamable HTTP).

**Python, with the optional `taskkit` helpers:**

```python
from taskkit.server import serve, TasksProvider        # the MCP wiring only; imports no data library

class MyProvider:                                       # implement the tools yourself (mcp/gex does)
    def task_list(self): ...
    def task_describe(self, task): ...
    def task_sample_rows(self, task, limit, offset, columns): ...
    def task_column_stats(self, task): ...
    def task_query(self, task, sql, limit): ...
    def harness_export_rows(self, task, path, until): ...
    def harness_evaluate(self, task, actions_path): ...
    def harness_actions(self, task, actions_path, start, end, limit): ...
    def harness_leak_scan(self, task, top): ...

serve(MyProvider(), name="my-tasks", default_port=8203, oauth_dir=HERE / ".oauth")
```

or subclass the polars `taskkit.task.Task` and let it implement the data half (rows, exports,
samples, SQL, stats, leak scan). Then you only write `load_rows()`, `evaluate()` and
optionally `action_log()`, and serve it with `TasksProvider([MyTask()])` (mcp/test/battery
does). For a table with no custom logic, write a JSON file for `mcp/test/tables` (format in
`taskkit/table.py`).

| helper | what it is |
| --- | --- |
| `taskkit/server.py` | the tool → method mapping, `serve()` (stdio / `--http`), `TasksProvider` |
| `taskkit/oauth.py` | OAuth 2.1 for HTTP servers, and `generate()` behind every `make_oauth_secrets.py` |
| `taskkit/task.py` | `Task`: a polars base class for the data half of the interface |
| `taskkit/table.py` | `TableTask`: a task from a JSON file (as-of joins, delayed columns) |
| `taskkit/evaluators.py` | ready valuations: trading positions, forecasts (numpy) |
| `taskkit/metrics.py` | Sharpe & co., swing capture (numpy) |

---

## Running and connecting securely

**Registering** a server: in the console under **Connectors → Register an MCP server**, give a
name and either a **URL** (`http(s)://…/mcp`, where OAuth is detected from the server's
discovery metadata and an optional client id and secret are stored owner-only) or a **local
path** (a Python server script, or a folder holding `server.py`, run over stdio; you must confirm
you trust the code, because the control plane will run it). The kind is auto-detected from the
tools the server offers, or chosen when it can't be probed before sign-in. **Remove** unregisters
it, except while a project uses it as its data/action MCP. A server's own `make_oauth_secrets.py`
registers it too. Or edit the file directly.

Every data/action MCP is registered in `ui/backend/mcp_servers.json` with **`"kind": "task"`**.
That is what separates it from the ordinary tool connectors (`"kind": "tool"`, the default):
only `task` servers can be a project's data/action MCP or score an objective, and the Connectors
page lists them in their own section.

**stdio** (the default): the control plane launches the server as a child process per call.
There's no network surface and nothing to sign in to. Register it with a command:

```json
{"name": "battery-demo", "kind": "task", "transport": "stdio", "command": "C:\\...\\.venv\\Scripts\\python.exe",
 "args": ["C:\\...\\mcp\\test\\battery\\server.py"], "enabled": true}
```

**HTTP** (`python server.py --http --port N`) **requires OAuth 2.1.** A server refuses to start
without its secrets; `--no-auth` is accepted on a loopback host only, for testing.

1. **Make the secrets.** Run `python make_oauth_secrets.py` in the server's folder. It writes:
   - `.oauth/server.json`: the one pre-registered client, the approval passphrase's scrypt hash
     and the token signing key (owner-only, git-ignored).
   - `ui/backend/auth/mcp_clients/<name>.secret`: the client secret for the control plane
     (owner-only).
   - the `<name>` entry in `ui/backend/mcp_servers.json`, with `kind: "task"`, `http`,
     `oauth: true`, the client id and `client_secret_file`.

   It prints the **approval passphrase** (also saved to `.oauth/approval_passphrase.txt`; store it
   and delete the file). Use `--rotate` to replace everything, `--help` for the options.
2. **Start the server** over HTTP. The GEX server is started by `start-services.cmd`
   (`ui\run-mcp-gex.bat`, port 8200); the examples run with `python server.py --http`.
3. **Connect from the console:** **Connectors** → the server → **Connect**. The server's own
   approval page opens; approve with the passphrase. The control plane stores the tokens
   owner-only and refreshes them itself, silently, even across restarts. **Disconnect**
   forgets them.

What makes it secure:

- A confidential client with a 256-bit secret and one fixed loopback redirect URI. Dynamic
  client registration is off.
- Authorization code with PKCE (S256), approved only with the operator's passphrase. Five wrong
  tries lock approvals for ten minutes.
- Access tokens are HMAC-signed, last one hour, and are bound to the server's resource URL
  (checked on every request) and the `tasks` scope. Refresh tokens are stored only as hashes, rotated on every use and
  revocable (`/revoke`).
- Every `/mcp` request needs a valid bearer token.
- Plain `http` is allowed on loopback only. To reach a server from another machine, put it
  behind TLS (a reverse proxy) and pass `--public-url https://...` and `--host`.

---

## Tests

```bat
.venv\Scripts\python.exe -m pytest mcp\tests -q
```

- `test_interface.py` runs the same interface checks against **every** server: every tool
  present, in-sample-only reads, exports cut before the cut, actions → valuation, refusing
  misdated actions.
- `test_oauth.py` drives the full OAuth flow against a live HTTP server, including forged,
  replayed and missing credentials.
- `test_gex.py` and `test_taskkit.py` test each server's internals.
- The GEX tests skip when `mcp/gex` is absent.
