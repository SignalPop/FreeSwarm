"""Is the covariate actually reaching the model, and does Moirai 2.0 support it at all?

Three checks:
  1. Does changing the covariate's CONTENT change the forecast? If not, it is being
     ignored and the whole covariate path is decorative.
  2. Does the model advertise covariate support?
  3. Does GluonTS carry the field into the batch?
"""
import math
import random

import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

MODEL = "Salesforce/moirai-2.0-R-small"
module = Moirai2Module.from_pretrained(MODEL)

N, H = 200, 8
random.seed(5)
target = [50 + 2 * math.sin(i / 9.0) + random.gauss(0, 0.2) for i in range(N)]


def run(cov):
    idx = pd.date_range("2000-01-01", periods=N, freq="h")
    frame = pd.DataFrame({"target": target}, index=idx)
    names = []
    if cov is not None:
        frame["cov_0"] = cov
        names = ["cov_0"]
    ds = PandasDataset(frame, target="target",
                       past_feat_dynamic_real=names or None)
    model = Moirai2Forecast(
        module=module, prediction_length=H, context_length=N,
        target_dim=1, feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=len(names),
    )
    fc = next(iter(model.create_predictor(batch_size=1).predict(ds)))
    return [round(float(fc.quantile(0.5)[i]), 5) for i in range(H)]


base = run(None)
cov_a = run([math.sin(i / 5.0) for i in range(N)])
cov_b = run([1000.0 + 50 * math.cos(i / 3.0) for i in range(N)])

print("  no covariate   :", base[:4])
print("  covariate A    :", cov_a[:4])
print("  covariate B    :", cov_b[:4])
print()
print("  A differs from no-cov :", cov_a != base)
print("  B differs from A      :", cov_b != cov_a)
print()
if cov_a == base and cov_b == base:
    print("  => the covariate is IGNORED. Moirai 2.0 is not consuming")
    print("     past_feat_dynamic_real through this path.")
else:
    print("  => the covariate IS reaching the model.")

# What does the checkpoint itself say?
print()
print("  module config fields mentioning feat/cov/variate:")
cfg = getattr(module, "config", None) or getattr(module, "hparams", None)
if cfg is not None:
    keys = [k for k in dict(cfg).keys()
            if any(t in k.lower() for t in ("feat", "cov", "variate", "dim"))]
    print("   ", keys or "(none)")
