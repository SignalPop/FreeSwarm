"""Starter regime detector: dealer gamma (GEX) x volatility -> four regimes. Causal.

Dealer gamma is the classic intraday regime split: with dealers LONG gamma (GEX > 0) their
hedging sells rallies and buys dips, so moves tend to mean-revert and pin; SHORT gamma
(GEX < 0) hedging chases the move, so trends and volatility extend. Crossing that with
realised volatility against its own recent norm gives four regimes:

    pos_gamma_calm, pos_gamma_volatile, neg_gamma_calm, neg_gamma_volatile  (+ "unknown" warm-up)

Everything is a trailing window, so the label at row t uses rows <= t only. Rows must be
sorted by time. Window lengths are in base bars (10 s: 360 = 1 hour).

    from lib import regime_gex_vol
    regime = regime_gex_vol.detect(df)
"""

import numpy as np
import pandas as pd

LABELS = ["pos_gamma_calm", "pos_gamma_volatile", "neg_gamma_calm", "neg_gamma_volatile"]


def detect(df, gex_col="GEX", price_col="Close", gex_window=360, vol_window=360, vol_ref_window=360 * 20):
    gex = pd.to_numeric(df[gex_col], errors="coerce")
    g = gex.rolling(gex_window, min_periods=max(1, gex_window // 10)).mean()
    r = np.log(pd.to_numeric(df[price_col], errors="coerce")).diff()
    vol = r.rolling(vol_window, min_periods=max(2, vol_window // 4)).std()
    ref = vol.rolling(vol_ref_window, min_periods=vol_window).median()
    gamma = np.where(g >= 0, "pos_gamma", "neg_gamma")
    state = np.where(vol > ref, "volatile", "calm")
    out = pd.Series(np.char.add(np.char.add(gamma.astype(str), "_"), state.astype(str)), index=df.index)
    out[(g.isna() | ref.isna()).to_numpy()] = "unknown"
    return out
