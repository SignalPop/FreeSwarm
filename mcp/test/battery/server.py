"""Demo task server: run a home battery on a power grid -- buy cheap electricity, sell it dear.

This is the reference example of a *custom* FreeSwarm task server (a data/action MCP) built
with the optional taskkit helpers: it generates its own data (numpy, cached as parquet, served
with polars), defines what an action means, MANAGES the actions (a battery's state of charge,
clipping what a full or empty battery cannot do), and values the result -- a daily profit curve
scored against a perfect-foresight oracle. It speaks exactly the same interface as mcp/gex
(mcp/README.md); only its internals differ.

THE PROBLEM (what the agents read)
    A 13.5 kWh home battery (5 kW) sits on a power grid whose price changes every hour. Every
    hour you decide how hard to charge (buy) or discharge (sell) during the NEXT hour. Prices
    follow the grid's net load -- demand minus solar and wind -- so they dip at sunny middays
    and windy nights and spike on hot or cold evenings. The market publishes a day-ahead price
    forecast for every hour, 12+ hours in advance; real prices then deviate from it.

    Profit = energy sold x price - energy bought x price - battery wear. The score is the share
    of the perfect-foresight profit you capture: 0 = doing nothing, 1 = a clairvoyant, below 0 =
    losing money. The battery loses 5% of the energy going in and 5% coming out, so a trade
    only pays if the later price beats the earlier one by more than ~11% plus wear.

WHY IT IS A GOOD TEST
    * Intuitive -- everybody understands "charge when cheap, sell when dear".
    * State -- the battery's charge carries from hour to hour; a strategy must track it.
    * Legitimate foresight -- the day-ahead forecast columns ARE known in advance and are the
      key to planning; the realised next-hour price is NOT, and a strategy that peeks at it
      (price.shift(-1)) fails the harness's look-ahead test.
    * Headroom -- a fixed "charge at the solar midday, sell at the evening peak" schedule
      captures about half; the rest must come from forecasts, weather and outages while
      managing the charge; the oracle bounds it at 1.

Register in ui/backend/mcp_servers.json (the control plane launches it on demand):

    {"name": "battery-demo", "transport": "stdio",
     "command": "<repo>/.venv/Scripts/python.exe",
     "args": ["<repo>/mcp/test/battery/server.py"], "enabled": true}

or serve it over HTTP:  python server.py --http --port 8201
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))                         # mcp/, for taskkit

from taskkit.server import TasksProvider, serve  # noqa: E402
from taskkit.task import DAY_NS, Task, cache_dir, iso, keys_ns  # noqa: E402

# The battery (a Powerwall-sized home battery).
CAPACITY_KWH = 13.5
POWER_KW = 5.0
EFF_IN = 0.95          # energy stored per kWh bought
EFF_OUT = 0.95         # energy sold per kWh drawn from the battery
WEAR_PER_KWH = 0.01    # $ of battery wear per kWh moved in or out
START_SOC = 0.5        # the battery starts half full

SEED = 20240601
START, END = "2023-01-01", "2025-01-01"   # two years of hours
HOLDOUT_FROM = "2024-09-01"
DA_HOURS = 12                              # day-ahead prices published for the next 12 hours


# ---------------------------------------------------------------------------------------------
# The world: weather -> demand, solar, wind -> net load -> prices (deterministic, seeded)
# ---------------------------------------------------------------------------------------------
def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    e = rng.normal(0.0, sd, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def generate(dem_err: float = 0.06, sol_err: float = 0.5, wind_err: float = 0.55, rt_sd: float = 9.0,
             outage_rate: float = 1 / 150, solar_mw: float = 950, wind_mw: float = 1100,
             wind_sd: float = 1.0, cloud_sd: float = 0.5) -> pl.DataFrame:
    """Two years of hourly grid rows. Every column of row t is known by the end of hour t,
    except the da_* columns, which the market publishes in advance (that is their point).

    The defaults are tuned so the problem has headroom: the fixed midday/evening schedule
    captures about half the perfect-foresight profit, a simple day-ahead threshold rule a little
    more, and the rest has to come from reading weather, forecast errors and outages while
    managing the battery's charge. `*_err` are day-ahead forecast errors, `rt_sd` real-time
    noise ($/MWh), `outage_rate` the hourly chance a plant trips (a real-time price spike)."""
    rng = np.random.default_rng(SEED)
    t = np.arange(np.datetime64(START, "h"), np.datetime64(END, "h"), np.timedelta64(1, "h"))
    n = len(t)
    days = t.astype("datetime64[D]")
    hour = (t - days).astype(int)
    doy = (days - t.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1
    dow = (days.astype(int) + 3) % 7                                 # 1970-01-01 was a Thursday
    season = np.cos(2 * np.pi * (doy - 200) / 365.25)            # +1 mid-July, -1 mid-January

    # Weather: seasonal + daily cycle + multi-day weather systems (AR over hours).
    temp = 12 + 11 * season + 4.5 * np.sin(2 * np.pi * (hour - 9) / 24) + _ar1(rng, n, 0.995, 0.35)
    cloud = 1 / (1 + np.exp(-(_ar1(rng, n, 0.985, cloud_sd) + 0.2 - 0.6 * season)))
    wind_speed = np.clip(6 + 1.5 * -season + _ar1(rng, n, 0.99, wind_sd), 0, None)

    # Supply and demand, MW on a small grid.
    elev = np.clip(np.sin(np.pi * (hour + 0.5 - 6 - 1.2 * season) / (12 + 2.4 * season)), 0, None)
    solar = solar_mw * elev * (0.75 + 0.25 * season) * (1 - 0.8 * cloud)
    wind = wind_mw * np.clip((wind_speed - 3) / 9, 0, 1) ** 2
    daily = (0.82 + 0.10 * np.exp(-((hour - 8) ** 2) / 4) + 0.20 * np.exp(-((hour - 19) ** 2) / 5)
             - 0.12 * np.exp(-((hour - 3.5) ** 2) / 6))
    weekend = np.where(dow >= 5, 0.92, 1.0)
    comfort = 22 * np.clip(temp - 22, 0, None) + 14 * np.clip(12 - temp, 0, None)
    demand = (1500 * daily * weekend + comfort) * (1 + 0.02 * rng.normal(size=n))
    net = demand - solar - wind

    def price_of(net_load: np.ndarray) -> np.ndarray:
        # Merit order: cheap when net load is low (negative when renewables flood the grid),
        # convex as it climbs, scarcity spikes near the top of the stack.
        return (-15 + 0.055 * net_load + 2.2e-5 * np.clip(net_load, 0, None) ** 2
                + np.minimum(260 * np.exp((net_load - 2150) / 110), 400.0))   # the market's price cap

    # Day-ahead: the same model on yesterday's forecasts of demand, solar and wind.
    f_err = dem_err * _ar1(rng, n, 0.97, 0.45) * demand
    s_err = solar * sol_err * _ar1(rng, n, 0.97, 0.25)
    w_err = wind * wind_err * _ar1(rng, n, 0.97, 0.25)
    net_da = (demand + f_err) - (solar + s_err) - (wind + w_err)
    da = price_of(net_da)
    # Real time: the realised net load, plus plant outages (rare, sudden, lasting a few hours).
    outage = np.zeros(n)
    for start in np.flatnonzero(rng.random(n) < outage_rate):
        outage[start: start + rng.integers(2, 7)] += rng.uniform(120, 380)
    price = price_of(net + outage) + rng.normal(0, rt_sd, n)

    cols = {
        "t": t.astype("datetime64[ns]"), "hour": hour, "weekday": dow,
        "temperature_c": temp.round(2), "cloud_cover": cloud.round(3), "wind_speed": wind_speed.round(2),
        "demand_mw": demand.round(1), "solar_mw": solar.round(1), "wind_mw": wind.round(1),
        "price": price.round(2),
    }
    for h in range(0, DA_HOURS + 1):
        cols[f"da_price_{h}" if h else "da_price_now"] = np.r_[da[h:], np.full(h, np.nan)].round(2)
    return pl.DataFrame(cols)


# ---------------------------------------------------------------------------------------------
# The battery: simulate a strategy, and the perfect-foresight oracle
# ---------------------------------------------------------------------------------------------
def simulate(price: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the battery. action[i] in [-1, 1] (decided at row i) sets the power over hour i+1,
    settled at price[i+1]. Returns per-hour profit ($), the state of charge (kWh) at the end of
    each hour, and the energy actually moved (kWh, + = bought) -- requests beyond what the
    battery can take or give are clipped."""
    n = len(price)
    profit = np.zeros(n)
    soc = np.zeros(n)
    moved = np.zeros(n)
    s = START_SOC * CAPACITY_KWH
    soc[0] = s
    for i in range(1, n):
        want = float(np.clip(action[i - 1], -1, 1)) * POWER_KW
        if want > 0:
            e = min(want, (CAPACITY_KWH - s) / EFF_IN)          # kWh bought
            s += e * EFF_IN
            profit[i] = -e * price[i] / 1000 - WEAR_PER_KWH * e * EFF_IN
            moved[i] = e
        elif want < 0:
            d = min(-want / EFF_OUT, s)                            # kWh drawn from the battery
            s -= d
            profit[i] = d * EFF_OUT * price[i] / 1000 - WEAR_PER_KWH * d
            moved[i] = -d * EFF_OUT
        soc[i] = s
    return profit, soc, moved


def oracle(price: np.ndarray, levels: int = 55) -> np.ndarray:
    """Per-hour profit of the best possible schedule with every price known in advance: dynamic
    programming over a grid of charge levels. The upper bound strategies are measured against."""
    grid = np.linspace(0, CAPACITY_KWH, levels)
    step = grid[1]
    n = len(price)
    # Moves between levels: storing k steps costs k*step/EFF_IN bought; releasing k steps sells k*step*EFF_OUT.
    kmax_in = int(np.floor(POWER_KW * EFF_IN / step + 1e-9))
    kmax_out = int(np.floor(POWER_KW / EFF_OUT / step + 1e-9)) if EFF_OUT else 0
    kmax_out = min(kmax_out, int(np.floor(POWER_KW / step / EFF_OUT + 1e-9)))
    ks = np.arange(-kmax_out, kmax_in + 1)
    ks = ks[np.where(ks < 0, -ks * step * EFF_OUT <= POWER_KW + 1e-9, ks * step / EFF_IN <= POWER_KW + 1e-9)]
    V = np.zeros(levels)
    choice = np.zeros((n, levels), dtype=np.int16)
    idx = np.arange(levels)
    for i in range(n - 1, 0, -1):
        best = np.full(levels, -np.inf)
        arg = np.zeros(levels, dtype=np.int16)
        for k in ks:
            j = idx + k
            ok = (j >= 0) & (j < levels)
            if k > 0:
                r = -(k * step / EFF_IN) * price[i] / 1000 - WEAR_PER_KWH * k * step
            elif k < 0:
                r = (-k * step) * EFF_OUT * price[i] / 1000 - WEAR_PER_KWH * (-k * step)
            else:
                r = 0.0
            v = np.where(ok, r + V[np.clip(j, 0, levels - 1)], -np.inf)
            better = v > best
            best = np.where(better, v, best)
            arg = np.where(better, k, arg)
        V = best
        choice[i] = arg
    profit = np.zeros(n)
    lvl = int(round(START_SOC * (levels - 1)))
    for i in range(1, n):
        k = int(choice[i][lvl])
        if k > 0:
            profit[i] = -(k * step / EFF_IN) * price[i] / 1000 - WEAR_PER_KWH * k * step
        elif k < 0:
            profit[i] = (-k * step) * EFF_OUT * price[i] / 1000 - WEAR_PER_KWH * (-k * step)
        lvl += k
    return profit


def fixed_schedule(rows: pl.DataFrame) -> np.ndarray:
    """The naive baseline: charge 11 am-3 pm (the solar dip), sell 6-10 pm (the evening peak),
    every day, no forecasts."""
    h = rows["hour"].to_numpy()
    nxt = (h + 1) % 24                               # the action at row i sets hour i+1
    return np.where((nxt >= 11) & (nxt < 15), 1.0, np.where((nxt >= 18) & (nxt < 22), -1.0, 0.0))


# ---------------------------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------------------------
class HomeBattery(Task):
    name = "home_battery"
    title = "Run a home battery: buy electricity cheap, sell it dear (demo task)"
    description = __doc__.split("THE PROBLEM (what the agents read)")[1].split("WHY IT IS A GOOD TEST")[0]
    brief = (
        "ROWS are complete hours: row t's `price` is what hour t actually cost ($/MWh). YOUR ACTION at row t "
        "is the battery power for the NEXT hour, from -1 (sell at the full 5 kW) through 0 (idle) to +1 (charge "
        "at the full 5 kW); it is settled at row t+1's price, which you do not know yet. `da_price_now` and "
        "`da_price_1`..`da_price_12` are the market's day-ahead prices for this hour and the next 12 hours -- "
        "published in advance, so you MAY use them; they are forecasts and the real price deviates. The battery "
        f"holds {CAPACITY_KWH} kWh, starts half full, and loses {100 - EFF_IN * 100:.0f}% of energy each way plus "
        f"${WEAR_PER_KWH}/kWh of wear, so trade only on spreads that beat the losses. The simulator clips requests "
        "a full or empty battery cannot serve -- track the state of charge in your code (simulate it the same "
        "way) so your plan stays feasible. Never use a future row (e.g. price.shift(-1)): the harness truncates "
        "the rows and fails any strategy whose earlier actions change."
    )
    target = "price"
    action = {"kind": "setpoint", "min": -1.0, "max": 1.0, "initial": 0.0,
              "description": "Battery power over the NEXT hour: -1 = sell 5 kW, 0 = idle, +1 = charge 5 kW."}
    holdout_from = HOLDOUT_FROM
    ahead_columns = ["da_price_*"]          # published by the market in advance -- they lead by design
    valuation_info = {
        "summary": "daily profit of the managed battery; score = share of the perfect-foresight profit captured "
                   "(0 = idle, 1 = a clairvoyant dynamic-programming oracle)",
        "capacity_kwh": CAPACITY_KWH, "power_kw": POWER_KW, "efficiency_in": EFF_IN, "efficiency_out": EFF_OUT,
        "wear_per_kwh": WEAR_PER_KWH, "baseline": "charge 11 am-3 pm, sell 6-10 pm (reported beside the score)",
    }
    value_functions = [
        {"name": "profit_capture", "title": "Share of perfect foresight", "stat": "score", "higher_is_better": True,
         "description": "profit as a share of what a clairvoyant battery would have made (0 = idle, 1 = perfect); "
                        "comparable across seasons and price levels"},
        {"name": "profit", "title": "Total profit ($)", "higher_is_better": True,
         "description": "dollars made over the period, net of wear -- rewards the busiest trading periods"},
        {"name": "daily_sharpe", "title": "Sharpe of daily profit", "higher_is_better": True,
         "description": "mean over volatility of the daily profit, annualised -- rewards steady days over a few big ones"},
    ]
    score_name = "profit_capture"
    higher_is_better = True
    column_notes = {
        "t": "start of the hour (UTC, a synthetic grid)",
        "hour": "hour of day 0-23", "weekday": "0 = Monday",
        "temperature_c": "air temperature -- heating and cooling drive demand",
        "cloud_cover": "0 clear .. 1 overcast -- clouds cut solar output",
        "wind_speed": "m/s -- drives wind output",
        "demand_mw": "grid demand this hour", "solar_mw": "solar output this hour", "wind_mw": "wind output this hour",
        "price": "the realised price of this hour, $/MWh (the target) -- can be negative when renewables flood the grid",
        "da_price_now": "day-ahead price published for this hour",
        **{f"da_price_{h}": f"day-ahead price published for the hour {h} hour(s) ahead (known now)" for h in range(1, DA_HOURS + 1)},
    }

    def version_parts(self) -> list[str]:
        return [self.name, str(SEED), START, END, "v2"]

    def load_rows(self) -> pl.DataFrame:
        f = cache_dir(HERE / ".cache") / f"battery-{self.version()}.parquet"
        if f.is_file():
            return pl.read_parquet(f)
        df = generate()
        df.write_parquet(f)
        return df

    def _oracle(self) -> np.ndarray:
        f = cache_dir(HERE / ".cache") / f"battery-oracle-{self.version()}.npy"
        if f.is_file():
            return np.load(f)
        o = oracle(self.rows()["price"].cast(pl.Float64).to_numpy())
        np.save(f, o)
        return o

    def evaluate(self, rows: pl.DataFrame, actions: np.ndarray) -> dict[str, Any]:
        price = rows["price"].to_numpy()
        profit, soc, moved = simulate(price, actions)
        best = self._oracle()
        base, _, _ = simulate(price, fixed_schedule(rows))
        t = keys_ns(rows)
        day = t // DAY_NS
        days_all, first_all = np.unique(day, return_index=True)
        daily_all = np.add.reduceat(profit, first_all)
        hold = t >= self.holdout_ns()
        requested = np.r_[0.0, np.clip(actions[:-1], -1, 1)] * POWER_KW
        clipped = np.abs(np.abs(requested) - np.abs(np.where(moved > 0, moved, moved / EFF_OUT))) > 1e-6
        segments, diagnostics = {}, {}
        for name, m in {"in_sample": ~hold, "holdout": hold}.items():
            if not m.any():
                continue
            ob = float(best[m].sum())
            capture = float(profit[m].sum()) / ob if ob > 0 else None
            dm = (days_all < self.holdout_ns() // DAY_NS) if name == "in_sample" else (days_all >= self.holdout_ns() // DAY_NS)
            daily = daily_all[dm]
            cum = np.cumsum(daily)
            sd = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
            segments[name] = {"score": None if capture is None else round(capture, 4),
                              "daily_sharpe": round(float(daily.mean()) / sd * 365 ** 0.5, 4) if sd > 1e-12 else None,
                              "profit": round(float(profit[m].sum()), 2), "oracle_profit": round(ob, 2),
                              "baseline_capture": round(float(base[m].sum()) / ob, 4) if ob > 0 else None,
                              "days": int(len(daily)),
                              "profitable_days": round(float((daily > 0).mean()), 4) if len(daily) else None,
                              "worst_day": round(float(daily.min()), 2) if len(daily) else None,
                              "best_day": round(float(daily.max()), 2) if len(daily) else None,
                              "max_drawdown": round(float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min()), 2)
                              if len(daily) else None}
            diagnostics[name] = {
                "kwh_bought": round(float(moved[m][moved[m] > 0].sum()), 1),
                "kwh_sold": round(float(-moved[m][moved[m] < 0].sum()), 1),
                "full_cycles": round(float(moved[m][moved[m] > 0].sum()) * EFF_IN / CAPACITY_KWH, 1),
                "hours_charging": int((moved[m] > 0).sum()), "hours_selling": int((moved[m] < 0).sum()),
                "requests_clipped": int((clipped & m & (requested != 0)).sum()),
                "avg_buy_price": round(float((price[m] * np.clip(moved[m], 0, None)).sum() / max(1e-9, np.clip(moved[m], 0, None).sum())), 2),
                "avg_sell_price": round(float((price[m] * np.clip(-moved[m], 0, None)).sum() / max(1e-9, np.clip(-moved[m], 0, None).sum())), 2),
            }
        curve = [[str(np.datetime64(int(d), "D")), round(float(v), 4)] for d, v in zip(days_all, daily_all)]
        ins, dins = segments.get("in_sample", {}), diagnostics.get("in_sample", {})
        notes = (f"In-sample: captured {ins.get('score')} of the perfect-foresight profit (${ins.get('profit')} of "
                 f"${ins.get('oracle_profit')}); the fixed 'charge 11 am-3 pm, sell 6-10 pm' schedule captures "
                 f"{ins.get('baseline_capture')}. Bought {dins.get('kwh_bought')} kWh at ${dins.get('avg_buy_price')}/MWh on "
                 f"average, sold {dins.get('kwh_sold')} kWh at ${dins.get('avg_sell_price')}/MWh; "
                 f"{dins.get('requests_clipped')} hour(s) asked for more than a full or empty battery could do.") if ins else ""
        return {"segments": segments, "diagnostics": diagnostics, "curve": curve, "curve_kind": "additive",
                "notes": notes}

    def action_log(self, rows: pl.DataFrame, actions: np.ndarray, lo_ns: int, hi_ns: int, limit: int) -> dict[str, Any]:
        """The managed actions between lo and hi: each block of hours spent charging or selling (with
        the energy moved, average price and profit) and the battery's state of charge per hour."""
        price = rows["price"].to_numpy()
        profit, soc, moved = simulate(price, actions)
        t = keys_ns(rows)
        w = np.flatnonzero((t >= lo_ns) & (t < hi_ns))
        mode = np.sign(moved[w])
        starts = np.flatnonzero(np.r_[True, mode[1:] != mode[:-1]])
        events = []
        for a, b in zip(starts, np.r_[starts[1:], len(w)]):
            if mode[a] == 0:
                continue
            idx = w[a:b]
            e = np.abs(moved[idx])
            events.append({"from": iso(t[idx[0]]), "to": iso(t[idx[-1]]), "what": "charge" if mode[a] > 0 else "sell",
                           "kwh": round(float(e.sum()), 2), "avg_price": round(float((price[idx] * e).sum() / e.sum()), 2),
                           "profit": round(float(profit[idx].sum()), 4)})
            if len(events) >= limit:
                break
        return {"events": events, "state": [[iso(t[i]), round(float(soc[i]), 3)] for i in w[: limit * 24]],
                "state_kind": "battery charge, kWh (end of hour)", "bars": self.bars(rows, lo_ns, hi_ns)}


TASKS = [HomeBattery()]

if __name__ == "__main__":
    serve(TasksProvider(lambda: TASKS), name="battery-demo", default_port=8201, oauth_dir=HERE / ".oauth")
