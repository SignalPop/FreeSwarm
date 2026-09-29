"""Seed candidate: the noise-area breakout of Zarattini, Aziz & Barbon (2024) on the GEX task's rows.

A published intraday-momentum baseline for SPY, submitted as a starting point for the swarm to
improve -- not a finished strategy. Around each day's open it draws a "noise area" as wide as the
average move from the open at that minute over the previous 14 sessions. Every 30 minutes from
10:00 it goes long above the area and the session VWAP, short below both, flat otherwise, and it
is flat from 15:55 New York. The size is set at entry, larger when recent volatility is low.

Things worth trying from here: a gamma-regime gate (gate=ft.gamma_regime(rows) < 0), a trailing
stop instead of the area/VWAP exit (ft.trend_exits), checks every 15 minutes, a wider band
(band_mult) to trade less, and a meta-filter over the entries (ft.meta_filter).

Submit it from the console, or: POST /api/objectives/<id>/candidates
    {"code": <this file>, "rationale": "...", "model": "operator", "mode": "explore"}
"""

import ft

rows = ft.rows(columns=["Close", "Volume", "GEX"])
size = (ft.inverse_vol(rows["Close"], lookback=360) * 1.5).clip(0.5, 3.0)
pos = ft.noise_area_breakout(rows, size=size, lookback_days=14, check_every=30, first_check="10:00",
                             flat_at="15:55")
ft.report_actions(pos)
changes = int((pos.diff().abs() > 0).sum())
print(f"noise-area breakout: {changes} position changes over {rows['t'].dt.normalize().nunique()} sessions")
