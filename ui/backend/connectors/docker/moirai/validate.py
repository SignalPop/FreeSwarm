"""Does conditioning on a covariate actually improve the forecast?

Constructed so the answer is knowable: the target genuinely depends on a leading
indicator. If covariates are wired up correctly, supplying that indicator should beat
forecasting the target alone. If it does not, the covariate path is decorative.
"""
import importlib.util
import math
import random
import sys

spec = importlib.util.spec_from_file_location("m", "/app/moirai_forecast.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def median_of(out):
    rows = []
    for line in out.splitlines():
        s = line.strip()
        if "|" not in s:
            continue
        head, _, rest = s.partition("|")
        if not head.strip().isdigit():
            continue
        nums = [float(x) for x in rest.split()]
        if len(nums) == 5:
            rows.append(nums[2])
    return rows


def rmse(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)) / n)


H = 12
N = 240

# Target depends on a leading indicator: driver[t-3] shifts the target. A model that sees
# the driver has information a univariate model simply does not.
random.seed(11)
driver = [math.sin(i / 9.0) * 5 + random.gauss(0, 0.3) for i in range(N + H)]
target = [
    50 + 0.02 * i + 2.0 * driver[max(0, i - 3)] + random.gauss(0, 0.25)
    for i in range(N + H)
]

hist_t, truth = target[:N], target[N:N + H]
hist_d = driver[:N]

solo = median_of(m.moirai_forecast(hist_t, horizon=H))
withcov = median_of(m.moirai_forecast(hist_t, horizon=H, covariates=[hist_d]))
naive = [hist_t[-1]] * H

print("=== target driven by a lagged covariate ===")
print(f"  naive (carry)        RMSE: {rmse(naive, truth):8.4f}")
print(f"  moirai, no covariate RMSE: {rmse(solo, truth):8.4f}")
print(f"  moirai + covariate   RMSE: {rmse(withcov, truth):8.4f}")
if solo and withcov:
    delta = rmse(solo, truth) - rmse(withcov, truth)
    print(f"  covariate helped by      : {delta:+.4f} "
          f"({'yes' if delta > 0 else 'no'})")

# Guard: a mismatched covariate must be rejected, not silently ignored.
print()
print("=== validation ===")
bad = m.moirai_forecast(hist_t, horizon=H, covariates=[hist_d[:10]])
print("  short covariate ->", bad.splitlines()[0][:100])
