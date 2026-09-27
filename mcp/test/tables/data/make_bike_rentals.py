"""Generate bike_rentals.csv for the example task tasks/bike_rentals.json (deterministic).

One year of hourly rentals at a city bike-share station, driven by what anyone would guess:
commuter peaks on weekdays, leisure afternoons at weekends, fewer riders in rain, cold and dark,
holidays behaving like weekends. Run it again to rebuild the CSV:

    python make_bike_rentals.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
HERE = Path(__file__).resolve().parent


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    t = pd.date_range("2024-01-01", "2025-01-01", freq="h", inclusive="left")
    n = len(t)
    hour, dow, doy = t.hour.to_numpy(), t.dayofweek.to_numpy(), t.dayofyear.to_numpy()
    holidays = {1, 15, 50, 148, 186, 246, 332, 359, 360}           # a few days off in the year
    holiday = np.isin(doy, list(holidays)).astype(int)
    weekend = ((dow >= 5) | (holiday == 1)).astype(int)

    season = -np.cos(2 * np.pi * (doy - 15) / 366)                 # -1 mid-January, +1 mid-July
    e = rng.normal(0, 1, n)
    weather = np.zeros(n)
    for i in range(1, n):                                           # weather systems lasting days
        weather[i] = 0.995 * weather[i - 1] + 0.1 * e[i]
    temp = 11 + 10 * season + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + 3 * weather
    rain_p = 1 / (1 + np.exp(-(-2.3 - 1.2 * weather + 0.6 * rng.normal(size=n))))
    rain_mm = np.where(rng.random(n) < rain_p, rng.gamma(1.3, 1.4, n), 0.0)
    humidity = np.clip(60 + 25 * rain_p - 0.6 * (temp - 11) + rng.normal(0, 5, n), 15, 100)

    commute = np.exp(-((hour - 8) ** 2) / 1.6) + 0.9 * np.exp(-((hour - 17.5) ** 2) / 2.2)
    leisure = np.exp(-((hour - 14) ** 2) / 10)
    shape = np.where(weekend == 1, 0.15 * commute + 0.9 * leisure, commute + 0.35 * leisure)
    comfort = np.exp(-((temp - 22) ** 2) / 150)
    dark = np.where((hour < 6) | (hour > 21), 0.25, 1.0)
    rate = 6 + 140 * shape * comfort * dark * np.exp(-0.45 * rain_mm)
    rentals = rng.poisson(rate)

    # The weather service's forecast of the NEXT hour's temperature, published an hour ahead.
    temp_next_fc = np.r_[temp[1:], temp[-1]] + rng.normal(0, 0.8, n)
    return pd.DataFrame({
        "time": t.strftime("%Y-%m-%d %H:%M:%S"), "hour": hour, "weekday": dow, "holiday": holiday,
        "temperature_c": temp.round(1), "rain_mm": rain_mm.round(1), "humidity": humidity.round(0),
        "temp_next_hour_forecast": temp_next_fc.round(1), "rentals": rentals,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv(HERE / "bike_rentals.csv", index=False)
    print(f"wrote {len(df)} rows to {HERE / 'bike_rentals.csv'}")
