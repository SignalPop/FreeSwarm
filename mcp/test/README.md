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

Built on the optional polars `taskkit.task.Task`.

## `tables/`: tasks from JSON files (no code)

Every `*.json` here (and in folders listed in `FREESWARM_TASK_DIRS`) is a task: sources joined
as of time, a target, action bounds, and a `trading` or `forecast` valuation. See
`taskkit/table.py` for the format. `bike_rentals.json` asks for next hour's bike rentals from
weather and the calendar, scored by skill against "same as this hour". The data comes from
`data/make_bike_rentals.py`.

## Running

Both are registered as **stdio** servers (the control plane launches them on demand, with no
network surface):

```json
{"name": "battery-demo", "transport": "stdio", "command": "<repo>\\.venv\\Scripts\\python.exe",
 "args": ["<repo>\\mcp\\test\\battery\\server.py"], "enabled": true}
```

To serve one over the network instead, run `python make_oauth_secrets.py` in its folder
(re-registers it as http + OAuth), start it with `python server.py --http`, and connect from the
console (**Connectors** → **Connect**).
