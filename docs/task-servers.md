# Task objectives: putting any problem in front of the swarm

A **task objective** is an objective scored by a **task server**: an MCP server that serves the
data, manages the actions a strategy takes, and values the result. The swarm writes strategy
code; FreeSwarm runs it in the sandbox, proves it never uses the future, and ranks it by the
server's score. Nothing about this is finance-specific. Any problem you can phrase as "given the
rows so far, decide a number now; here is how good that decision turned out" is a task.

The servers themselves, the interface they share, the examples and how to secure them are in
**[`mcp/README.md`](../mcp/README.md)**. This page covers the FreeSwarm side.

```
                       ┌───────────── task server (MCP) ─────────────┐
                       │ rows · target · action management · valuation │
                       └──▲──────────────────────┬───────────────────▲─┘
       harness_export_rows │ rows (all, or < cut)  │ task_* tools       │ harness_evaluate
                           │                       ▼ (in-sample only)   │ (actions file)
   ┌───────────────────────┴───┐        ┌──────────────────┐   ┌──────┴────────────┐
   │ control plane (harness)   │───────►│ agents (swarm)   │   │ sandbox (docker)  │
   │ mounts the rows at /task, │ brief  │ explore, write a │──►│ ft.rows()          │
   │ runs, values, look-ahead  │◄───────│ strategy         │   │ ft.report_actions()│
   └───────────────────────────┘ submit └──────────────────┘   └───────────────────┘
```

## Setting one up

1. **Register and connect the server.** Any MCP can be registered from **Connectors → Register an
   MCP server** by URL or local path (see [`mcp/README.md`](../mcp/README.md)). `mcp/gex` is started
   by `start-services.cmd` and uses
   OAuth: run its `make_oauth_secrets.py` once, then in the console go to **Connectors** → `gex` →
   **Connect** and approve with the passphrase. The examples in `mcp/test` run over stdio with
   nothing to connect.
2. **Give the project its data/action MCP.** On **Projects** → the project → **Data/action MCP**,
   pick the server (the GEX project uses `gex`). Only servers registered with `"kind": "task"` are
   offered; ordinary connectors stay in the Connectors list below. The pill shows whether it is
   `ready`, `needs sign-in` or not registered. The project's agents can then always query that
   server (in-sample only), and its task objectives are scored by it: a new objective defaults
   to it, and naming another server is refused.

   Once it is ready, the panel shows what the server offers, task by task:
   - **Data:** rows × columns, bar spacing, the span, the holdout, and any delayed columns.
   - **Target:** a picker over the server's `target_options` (GEX: all 145 numeric fields, prices
     first), with how the server will value it. This is a project setting; new objectives of the
     task are valued on it.
   - **Value function:** a picker over the server's value functions (e.g. Sharpe, smooth Sharpe,
     Calmar, segment matching), with a description of what each rewards. This is a project
     setting too: it ranks new objectives of the task, the agents' brief names it, and the harness
     still takes the weaker of in-sample and holdout.
   - **Schema:** every column with its type, role and description, filterable.
   - **Actions:** what an action means and its bounds.
   - **Value function:** what the score measures and the settings behind it.
3. **Create the objective.** **Swarm** → **New objective**. With the project's server set, *How is
   "better" measured?* is already **Task server**, and only that server's tasks are offered. Pick
   the task and press **check data timing** (below). The split, the look-ahead cuts, the
   description and the rules all come from the server.
4. **Turn the swarm on.**

## How a candidate is run

1. **Rows.** `harness_export_rows` writes the task's rows to a folder mounted **read-only at
   `/task`**, and nothing else is mounted: not the project's data folder, not forecasts. The
   full export is cached per task version.
2. **Run.** The candidate reads `ft.rows()` (`ft.rows(columns=[...])` on big tasks) and
   `ft.task()`, and reports one action per row with `ft.report_actions(pd.Series(values,
   index=rows["t"]))`.
3. **Value.** `harness_evaluate` manages the actions and values them. The leaderboard score is
   the **weaker of the in-sample and holdout scores** (robust ranking), or the holdout alone. A
   server can mark a candidate `unranked` with a reason, such as a one-sided trading strategy.
4. **Look-ahead test** (in the background). The code is re-run on rows exported only up to each
   cut:
   - the holdout boundary
   - the mid in-sample cut
   - up to eight cuts placed seconds to 1.5 hours after the candidate's own action changes

   Every earlier action must be identical to the full run's, or the candidate fails and can
   never hold the title.
5. **Feedback.** The agent gets its in-sample score, the server's in-sample diagnostics and
   in-sample notes, and never anything about the holdout.

Agents' experiments (`run_python`) read the rows cut at the split. The brief for a task objective
is the task's own: its description, rules, target, action, score and column descriptions.

**Every field of the schema is open to the analysis tools.** When a task objective is created (and
whenever the server's data version changes), the MCP's **in-sample** rows are written to the project's
data folder as `mcp_tasks/<server>_<task>.parquet`, the objective's dataset view. The file holds only
rows before the split, delayed exactly as the server serves them. On it, agents run:
- `deci_plot` and `field_scan`, measured against the project's target
- `regime_map`
- `query_data`
- the field guide, built from the MCP's own column descriptions (GEX: 145 columns in 21 families)

The task server's `task_query`, `task_sample_rows` and `task_column_stats` are offered too (in-sample
only). Forecast features, the Regime Lab and ensembles stay withdrawn: a task candidate can't read
forecast features at run time and has no positions of its own to replay. The console shows each candidate's
valuation per segment, its diagnostics, the notes the agent got, and its curve (compounded
returns, or summed values such as profit). `GET /api/objectives/{id}/candidates/{cid}/actions`
returns its managed actions from the server.

**Drill-down:** click a day on a task candidate's curve to see the day as the server manages it
(`harness_actions`):
- the target as candles or a line, in the server's time zone (GEX: 1-minute candles, New York)
- the managed state underneath (the position, or the battery's charge)
- what the actions did: trades or charge/sell blocks, shaded over the bars and listed below

## The guarantees

| risk | how it is closed |
| --- | --- |
| code reads rows after its decision time | look-ahead test on truncated exports; any changed earlier action fails the candidate |
| code reads around the cut | in a task run the **only** mount is the (cut) task folder |
| code learns where the fixed cuts are | `task.json` in the sandbox omits the cuts, and the eight active cuts are random per candidate (behaving differently on fewer rows would change earlier actions, which is exactly what fails) |
| agents see holdout rows | experiments read rows cut at the split; `task_*` tools are in-sample only; `harness_*` tools are hidden from and refused to agents (any letter case) |
| agents see holdout scores | feedback, `get_candidate`, the brief and the mentor carry in-sample numbers only; a holdout's own "why not scored" note stays with the operator |
| full-sample statistics (a z-score over all rows) | changes earlier actions when rows are removed, so the same test catches it |
| a slower table joined into the rows | as-of backward join: a row gets its latest value at or before its time |
| someone else talking to a server | stdio has no network surface; HTTP needs OAuth 2.1 (PKCE, the operator's passphrase, signed short-lived tokens) |

`ui/backend/tests/test_task_integration.py` proves this end to end (opt-in, needs Docker): an
honest strategy passes the look-ahead test, while one that peeks at the next hour's price and one
that normalises with full-sample statistics both fail it, and an agent experiment sees rows only
up to the split.

## Data timing: leaks no code test can catch

The look-ahead test proves the *code* uses no future **rows**. It can't see a **column** that was
computed after the time it's filed under: a snapshot stamped at the start of a bar but taken at
its end, or a daily figure copied onto every bar of that day. Such a column lets any strategy
"predict" the next move.

**Check it:** **New objective** → **Task server** → pick the task → **check data timing**
(`harness_leak_scan`; operator-only, since it lists exactly the columns that leak). For each
column it compares how its *change* correlates with the target's move over the **current** row
and over the **next** one. A column that predicts the next move clearly better is a **suspect**.

**Fix it in the server:** delay suspect columns until they're known (`shift_rows` in a JSON
table task, `data.delay` in `mcp/gex/config.json`). Columns that are legitimately known in advance,
such as published forecasts, are declared `ahead_columns` so the scan lists them separately.

> **A real example.** On the GEX 10-second bars, 16 options-derived columns correlated about 0.2
> with the next bar's move and only about 0.1 with the current one. `Meta_SpotGex` is gamma at the
> spot price and should move with the current bar, but it peaks a bar late. A one-line rule on
> `Pressure_Total` scored an annualised Sharpe of about 160 before costs. Delaying every non-price
> column by two rows removed every suspect.

## Troubleshooting

| symptom | cause and fix |
| --- | --- |
| New objective lists no task servers | register the server in `ui/backend/mcp_servers.json` and make sure it runs; an OAuth server must also be connected |
| "task server 'gex' needs sign-in" | **Connectors** → `gex` → **Connect**, and approve with the passphrase from its `make_oauth_secrets.py` |
| a server refuses to start with "no OAuth secrets" | run `make_oauth_secrets.py` in its folder (or use stdio) |
| candidate error *no actions reported* | the script must call `ft.report_actions(pd.Series(values, index=rows["t"]))` |
| candidate error *N actions, 0 inside the rows' time range* | the series was indexed by row numbers, not by `rows["t"]` |
| look-ahead `error: the script failed on rows cut at ...` | the code assumes more rows than a shorter history has; make it robust to fewer rows |
| look-ahead `fail` | an earlier action changed when later rows were removed: `shift(-1)`, centred windows, full-sample statistics, or "the day's last bar" (use the clock instead) |
