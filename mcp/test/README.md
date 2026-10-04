# Example task servers

Two small servers that show both ways to build a task server. Both speak exactly the same
interface as `mcp/gex` ([`mcp/README.md`](../README.md)); only their internals differ.

## `battery/`: run a home battery on a power grid (custom logic)

A 13.5 kWh home battery trades against an hourly power price. Prices follow the grid's net load
(demand minus solar and wind), and the market publishes a day-ahead forecast 12+ hours ahead.

- **Data**: two years of hours, generated deterministically with numpy and cached as parquet.
  Columns: weather, solar, wind, demand, the real price, and the day-ahead forecasts
  (`da_price_*`, declared known-in-advance).
- **Actions**: battery power for the *next* hour, from -1 (sell at 5 kW) to +1 (charge at 5 kW).
  The server manages them: the state of charge, round-trip losses, wear, and clipping what a full
  or empty battery can't do. `harness_actions` lists each charge/sell block and the charge per hour.
- **Valuation**: a daily profit curve. The score is the **share of the perfect-foresight profit
  captured**, from a dynamic-programming oracle: 0 = idle, 1 = clairvoyant. It also reports the
  fixed-schedule baseline, profitable-day share, worst/best day and drawdown.

  | strategy | in-sample capture |
  | --- | --- |
  | fixed "charge 11–3, sell 6–10" | ~0.48 |
  | a simple day-ahead threshold rule | ~0.52 |
  | perfect foresight | 1.00 |

  The headroom is what the swarm must find.

- **Project settings it offers** (Projects → Data/action MCP): value functions *share of perfect
  foresight*, *total profit* and *Sharpe of daily profit*; the target `price` (the only one that
  makes sense for a battery); and a **backup reserve** as its action rule -- none, 20% or 50% of the
  battery kept for outages. The simulator never sells below the reserve, and the perfect-foresight
  benchmark keeps the same reserve, so the score stays comparable. Its agent guidance explains the
  day-ahead prices, the battery's limits and the reserve.

Built on the optional polars `taskkit.task.Task`.

## `tables/`: tasks from JSON files (no code)

Every `*.json` here (and in folders listed in `FREESWARM_TASK_DIRS`) is a task: sources joined
as of time, a target, action bounds, and a `trading` or `forecast` valuation. See
`taskkit/table.py` for the format. `bike_rentals.json` asks for next hour's bike rentals from
weather and the calendar, scored by skill against "same as this hour". The data comes from
`data/make_bike_rentals.py` (a CSV).

- **Sources** may be CSV/TSV, Parquet, NDJSON or **Excel** (`.xlsx`/`.xls`, the first sheet or
  `"sheet": "<name>"`; needs the `fastexcel` package, in the requirements) -- point a JSON file at
  your own spreadsheet and it is a task.
- **Project settings it offers:** the value functions of its evaluator (forecast: skill, RMSE,
  direction accuracy; trading: Sharpe, Sortino, Calmar, total return) and, with
  `"target_options": "numeric"`, every numeric column as the target -- bike rentals can be switched
  to forecasting the temperature or the humidity instead.

## Running

`start-services.cmd` starts both, like every task server under `mcp\`, over HTTP with OAuth:
battery on <http://127.0.0.1:8521/mcp>, tables on <http://127.0.0.1:8522/mcp>. On the first start
it runs each one's `make_oauth_secrets.py` (which registers it in `ui/backend/mcp_servers.json` as
http + OAuth) and saves the approval passphrase to `<folder>/.oauth/approval_passphrase.txt`.
Connect each once from the console (**Connectors** → the server → **Connect**, approving with that
passphrase); the connection then refreshes itself. One alone: `ui\run-mcp.bat mcp\test\battery`.

They can also run over **stdio** (the control plane launches them on demand, no network surface,
nothing to connect) -- register them that way instead:

```json
{"name": "battery-demo", "kind": "task", "transport": "stdio", "command": "<repo>\\.venv\\Scripts\\python.exe",
 "args": ["<repo>\\mcp\\test\\battery\\server.py"], "enabled": true}
```
