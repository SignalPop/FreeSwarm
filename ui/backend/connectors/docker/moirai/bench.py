"""Does the covariate help ON AVERAGE, or was the single bad trial noise?

A covariate path that is wired correctly but degrades accuracy is worth knowing about
before anyone builds on it, so this runs several independent series rather than one.
"""
import math
import random

import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

MODEL = "Salesforce/moirai-2.0-R-small"
module = Moirai2Module.from_pretrained(MODEL)
N, H = 200, 12


def predict(target, cov=None):
    idx = pd.date_range("2000-01-01", periods=len(target), freq="h")
    frame = pd.DataFrame({"target": target}, index=idx)
    names = []
    if cov is not None:
        frame["cov_0"] = cov
        names = ["cov_0"]
    ds = PandasDataset(frame, target="target", past_feat_dynamic_real=names or None)
    model = Moirai2Forecast(
        module=module, prediction_length=H, context_length=len(target),
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=len(names),
    )
    fc = next(iter(model.create_predictor(batch_size=1).predict(ds)))
    return [float(fc.quantile(0.5)[i]) for i in range(H)]


def rmse(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(b))


solo_wins = cov_wins = 0
solo_tot = cov_tot = naive_tot = 0.0
TRIALS = 8

for seed in range(TRIALS):
    random.seed(seed)
    drv = [math.sin(i / 9.0) * 5 + random.gauss(0, 0.3) for i in range(N + H)]
    tgt = [50 + 2.0 * drv[max(0, i - 3)] + random.gauss(0, 0.25) for i in range(N + H)]
    hist, truth = tgt[:N], tgt[N:N + H]

    r_solo = rmse(predict(hist), truth)
    r_cov = rmse(predict(hist, drv[:N]), truth)
    r_naive = rmse([hist[-1]] * H, truth)

    solo_tot += r_solo
    cov_tot += r_cov
    naive_tot += r_naive
    if r_cov < r_solo:
        cov_wins += 1
    else:
        solo_wins += 1
    print(f"  seed {seed}: naive {r_naive:6.3f} | solo {r_solo:6.3f} | +cov {r_cov:6.3f}"
          f"  {'cov better' if r_cov < r_solo else 'solo better'}")

print()
print(f"  mean naive RMSE     : {naive_tot / TRIALS:6.3f}")
print(f"  mean solo  RMSE     : {solo_tot / TRIALS:6.3f}")
print(f"  mean +cov  RMSE     : {cov_tot / TRIALS:6.3f}")
print(f"  covariate helped in : {cov_wins}/{TRIALS} trials")
