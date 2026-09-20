"""MCP connector serving Salesforce Moirai 2.0 as a multivariate forecasting tool.

Moirai's distinguishing feature is **covariates**: unlike Chronos it can condition on other
series alongside the target. For intraday market data that is the interesting capability --
forecast volume while the model also sees spread, volatility and time-of-day, rather than
forcing it to infer everything from one column.

Runs CPU-only inside a container. `uni2ts` pins `scipy~=1.11.3`, which has no Python 3.13
wheel, so the host would have to compile scipy from source; pinning the whole connector to
Python 3.11 in an image is cleaner and keeps a heavy scientific stack out of the engine
environment. MCP's stdio transport works through `docker run -i` unchanged.

**Licence: CC-BY-NC-4.0 — non-commercial use only.** Every other forecaster in this
project is Apache-2.0. That restriction travels with the weights, not this code.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("moirai")

MODEL_ID = os.getenv("FREESWARM_MOIRAI_MODEL", "Salesforce/moirai-2.0-R-small")
DEFAULT_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

_module: Any = None


def _load_module():
    """Load the pretrained weights once. Forecast heads are built per request, because
    prediction length and covariate count are part of the model's construction."""
    global _module
    if _module is None:
        import torch  # noqa: PLC0415
        from uni2ts.model.moirai2 import Moirai2Module  # noqa: PLC0415

        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        _module = Moirai2Module.from_pretrained(MODEL_ID)
    return _module


@mcp.tool()
def moirai_info() -> str:
    """Describe the Moirai forecaster, including its covariate support and its licence.

    Call this before `moirai_forecast` if you are deciding between forecasters.
    """
    return (
        f"Model: {MODEL_ID} (Salesforce Moirai 2.0)\n"
        "  type      : zero-shot probabilistic forecaster\n"
        "  covariates: YES -- condition the target on other series (volume, spread, VIX)\n"
        "  output    : quantiles (default 10/25/50/75/90th percentile)\n"
        "  runs on   : CPU, inside a container\n"
        "  LICENCE   : CC-BY-NC-4.0, NON-COMMERCIAL USE ONLY\n\n"
        "Use this over Chronos when other series plausibly explain the target. Use Chronos\n"
        "when you only have the one series, or when the licence matters.\n\n"
        "On market data: forecasting asset PRICE is close to forecasting a random walk and\n"
        "will look more confident than it deserves. Volume, realised volatility and spread\n"
        "carry real structure. Always compare against a seasonal-naive baseline."
    )


@mcp.tool()
def moirai_forecast(
    values: list,
    horizon: int = 24,
    covariates: list | None = None,
    context_length: int = 0,
) -> str:
    """Forecast a series, optionally conditioned on other series that explain it.

    `values` is the target, a flat list of numbers, oldest first.
    `covariates` is an optional list of equally long series (each a flat list) observed
    over the same timestamps -- for example volume, spread or realised volatility when
    forecasting price, or time-of-day when forecasting volume. They must be the SAME
    length as `values`; only their history is used.
    `horizon` is how many steps ahead to predict.
    `context_length` limits the history used; 0 uses all of it.

    Returns quantiles, not a single number. The width between them is the model's
    uncertainty and is the part that matters for any decision carrying risk.
    """
    if not values:
        return "error: `values` is empty."
    try:
        target = [float(v) for v in values]
    except (TypeError, ValueError):
        return "error: `values` must be a flat list of numbers."
    if len(target) < 16:
        return f"error: need at least 16 historical points, got {len(target)}."

    covs: list[list[float]] = []
    for i, c in enumerate(covariates or []):
        try:
            series = [float(x) for x in c]
        except (TypeError, ValueError):
            return f"error: covariate {i} is not a list of numbers."
        if len(series) != len(target):
            return (
                f"error: covariate {i} has {len(series)} points but the target has "
                f"{len(target)}. Covariates must cover exactly the same timestamps."
            )
        covs.append(series)

    horizon = max(1, min(int(horizon), 512))
    ctx = len(target) if context_length <= 0 else min(int(context_length), len(target))

    try:
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from gluonts.dataset.pandas import PandasDataset  # noqa: PLC0415
        from uni2ts.model.moirai2 import Moirai2Forecast  # noqa: PLC0415

        module = _load_module()

        # GluonTS wants a time index; the actual dates are irrelevant to the model, only
        # the ordering and regular spacing are.
        index = pd.date_range("2000-01-01", periods=len(target), freq="h")
        frame = pd.DataFrame({"target": target}, index=index)
        cov_names = []
        for i, series in enumerate(covs):
            name = f"cov_{i}"
            frame[name] = series
            cov_names.append(name)

        ds = PandasDataset(
            frame,
            target="target",
            past_feat_dynamic_real=cov_names or None,
        )

        model = Moirai2Forecast(
            module=module,
            prediction_length=horizon,
            context_length=ctx,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=len(cov_names),
        )
        predictor = model.create_predictor(batch_size=1)
        forecast = next(iter(predictor.predict(ds)))

        qs = DEFAULT_QUANTILES
        grid = [[float(forecast.quantile(q)[step]) for q in qs] for step in range(horizon)]
    except Exception as exc:  # noqa: BLE001 - returned as text so the caller can react
        return f"error: forecast failed: {type(exc).__name__}: {exc}"

    median_idx = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5))
    median = [row[median_idx] for row in grid]
    band = sum(row[-1] - row[0] for row in grid) / len(grid)
    last = target[-1]
    change = median[-1] - last

    lines = [
        f"Moirai forecast ({MODEL_ID}), {horizon} steps"
        + (f", conditioned on {len(covs)} covariate(s)" if covs else ", no covariates"),
        f"  last observed  : {last:.6g}",
        f"  median endpoint: {median[-1]:.6g}  ({change:+.6g}"
        + (f", {change / last * 100:+.2f}%)" if last else ")"),
        f"  median range   : {min(median):.6g} .. {max(median):.6g}",
        f"  mean {qs[0]:.0%}-{qs[-1]:.0%} band: {band:.6g}"
        + (f"  ({band / abs(last) * 100:.1f}% of last value)" if last else ""),
        "",
        "  step |  " + "  ".join(f"q{int(q * 100):02d}".rjust(10) for q in qs),
    ]
    show = list(range(min(5, horizon)))
    if horizon > 10:
        show += ["..."] + list(range(horizon - 3, horizon))
    elif horizon > 5:
        show += list(range(5, horizon))
    for i in show:
        if i == "...":
            lines.append("   ... |")
            continue
        lines.append(f"  {i + 1:4d} | " + "  ".join(f"{v:10.6g}" for v in grid[i]))

    lines += [
        "",
        "Quantiles, not a point estimate: the band is the model's uncertainty.",
        "Weights are CC-BY-NC-4.0 (non-commercial).",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
