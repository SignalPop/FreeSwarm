"""MCP connector serving Amazon's Chronos-Bolt as a probabilistic forecasting tool.

Chronos-Bolt is a zero-shot forecaster: no per-series training, feed it history and it
predicts. Unlike the PatchTST connector it returns a **distribution** (quantiles), which
is the difference between a number and a usable forecast — anything that informs sizing or
risk needs the spread of outcomes, not just the middle of them.

Runs on **CPU** in its own virtual environment (`.venv-ts`). Two reasons: the model is
small enough that a GPU buys nothing, and `chronos-forecasting` pins transformers versions
that would fight with the engine venv. An MCP server is a separate process with its own
interpreter, so it can have its own dependency set.

Register in `mcp_servers.json`:

    {"name": "chronos", "transport": "stdio",
     "command": "<repo>/.venv-ts/Scripts/python.exe",
     "args": ["<repo>/ui/backend/connectors/chronos_forecast.py"],
     "enabled": true}

**On market data specifically:** these models are trained mostly on energy, traffic and
web-metrics series, which have real autocorrelation. Intraday *price* is close to a
martingale, so a price forecast will look confident and backtest badly. The honest uses
are volume, realised volatility and spread — quantities that genuinely repeat.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("chronos")

MODEL_ID = os.getenv("FREESWARM_CHRONOS_MODEL", "amazon/chronos-bolt-base")
DEFAULT_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

_pipeline: Any = None


def _load():
    """Load once, lazily -- listing tools must not pay for importing torch."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    import torch  # noqa: PLC0415
    from chronos import BaseChronosPipeline  # noqa: PLC0415

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    _pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float32
    )
    return _pipeline


@mcp.tool()
def chronos_info() -> str:
    """Describe the Chronos forecaster: what it does and what it is and is not good for.

    Call this before `chronos_forecast` if you are unsure whether forecasting suits the
    task.
    """
    return (
        f"Model: {MODEL_ID} (Amazon Chronos-Bolt)\n"
        "  type       : zero-shot probabilistic forecaster, univariate\n"
        "  input      : any length of history; longer context generally helps\n"
        "  output     : quantiles (default 10/25/50/75/90th percentile)\n"
        "  horizon    : caller-specified; accuracy degrades as it grows\n"
        "  runs on    : CPU\n\n"
        "Good for: demand, volume, load, traffic, realised volatility -- series with\n"
        "genuine repeating structure.\n\n"
        "Poor for: asset PRICE levels. Financial prices are close to a random walk, so a\n"
        "price forecast will look confident and mean very little. Prefer forecasting\n"
        "volume, volatility or spread, and always compare against a seasonal-naive\n"
        "baseline before trusting it."
    )


@mcp.tool()
def chronos_forecast(
    values: list,
    horizon: int = 24,
    quantiles: list | None = None,
) -> str:
    """Forecast a numeric series, returning prediction quantiles rather than one number.

    `values` is a flat list of numbers, oldest first. `horizon` is how many steps ahead to
    predict. `quantiles` defaults to 10/25/50/75/90 -- the spread between them is the
    model's uncertainty, and is the part that matters for any decision with risk attached.

    Returns the median path plus the outer quantile band. A wide band means the model does
    not know; treat that as information, not noise.
    """
    if not values:
        return "error: `values` is empty."
    try:
        series = [float(v) for v in values]
    except (TypeError, ValueError):
        return "error: `values` must be a flat list of numbers."
    if len(series) < 8:
        return f"error: need at least 8 historical points, got {len(series)}."

    horizon = max(1, min(int(horizon), 512))
    qs = [float(q) for q in (quantiles or DEFAULT_QUANTILES)]
    if any(not 0.0 < q < 1.0 for q in qs):
        return "error: quantiles must be strictly between 0 and 1."
    qs = sorted(set(qs))

    try:
        import torch  # noqa: PLC0415

        pipeline = _load()
        # chronos 2.x takes `inputs`, not `context`.
        q_tensor, _mean = pipeline.predict_quantiles(
            torch.tensor(series, dtype=torch.float32),
            prediction_length=horizon,
            quantile_levels=qs,
        )
    except Exception as exc:  # noqa: BLE001 - returned as text so the caller can react
        return f"error: forecast failed: {type(exc).__name__}: {exc}"

    # (batch, horizon, n_quantiles)
    grid = q_tensor[0].tolist()
    median_idx = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5))
    median = [row[median_idx] for row in grid]
    lo = [row[0] for row in grid]
    hi = [row[-1] for row in grid]

    last = series[-1]
    change = median[-1] - last
    band = sum(hi[i] - lo[i] for i in range(len(median))) / len(median)

    lines = [
        f"Chronos forecast ({MODEL_ID}), {horizon} steps",
        f"  last observed  : {last:.6g}",
        f"  median endpoint: {median[-1]:.6g}  ({change:+.6g}"
        + (f", {change / last * 100:+.2f}%)" if last else ")"),
        f"  median range   : {min(median):.6g} .. {max(median):.6g}",
        f"  mean {qs[0]:.0%}-{qs[-1]:.0%} band width: {band:.6g}"
        + (f"  ({band / abs(last) * 100:.1f}% of last value)" if last else ""),
        "",
        "  step |    " + "  ".join(f"q{int(q * 100):02d}".rjust(10) for q in qs),
    ]
    # Show the first few and last few steps rather than every one: a 512-step dump would
    # swamp the calling model's context for no benefit.
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
        "The band between the outer quantiles is the model's uncertainty. A wide band "
        "means it does not know -- that is a finding, not noise.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
