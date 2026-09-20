"""MCP connector serving a PatchTST time-series forecaster as a tool.

FreeToken cannot serve this model: it is an encoder that maps a numeric window to a
forecast, with no tokenizer, no KV cache and no autoregressive decode. None of a
causal-LM engine's machinery applies. As an MCP tool, though, it becomes something any
chat model can call -- which is the more useful shape anyway, because the language model
handles the reasoning and this handles the numbers.

Runs on **CPU** deliberately. The checkpoint is ~2.4 MiB and a 512-step forecast takes
milliseconds, so putting it on a GPU would buy nothing while taking VRAM away from the
language models that actually need it.

Register in `mcp_servers.json`:

    {"name": "timeseries", "transport": "stdio",
     "command": "<repo>/.venv/Scripts/python.exe",
     "args": ["<repo>/ui/backend/connectors/timeseries.py"],
     "enabled": true}
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("timeseries")

# Override to point at a different checkpoint.
MODEL_ID = os.getenv("FREESWARM_TS_MODEL", "ibm-granite/granite-timeseries-patchtst")

_model: Any = None
_config: Any = None


def _load():
    """Load the forecaster once, on first use.

    Lazy because an MCP server is started for every tool listing, and importing torch
    eagerly would make simply enumerating the connectors cost several seconds.
    """
    global _model, _config
    if _model is not None:
        return _model, _config

    import torch  # noqa: PLC0415
    from transformers import PatchTSTForPrediction  # noqa: PLC0415

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    _model = PatchTSTForPrediction.from_pretrained(MODEL_ID)
    _model.eval()
    _config = _model.config
    return _model, _config


def _local_config() -> dict:
    """Read config.json off disk without importing torch, for cheap introspection."""
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--"
        + MODEL_ID.replace("/", "--")
        + "/snapshots/*/config.json"
    )
    hits = glob.glob(pattern)
    if not hits:
        return {}
    try:
        with open(hits[0], encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _as_matrix(values: list, channels: int) -> tuple[list[list[float]], bool]:
    """Normalise input into (timesteps x channels), reporting whether it was univariate.

    A univariate series is replicated across every channel. That is sound here rather
    than a fudge: this checkpoint has `channel_attention = False`, so channels never mix
    -- each is forecast independently, and channel 0's output depends only on channel 0's
    input. Verified against the config before relying on it.
    """
    if not values:
        raise ValueError("`values` is empty")

    if isinstance(values[0], (int, float)):
        series = [float(v) for v in values]
        return [[v] * channels for v in series], True

    matrix = [[float(x) for x in row] for row in values]
    width = len(matrix[0])
    if any(len(r) != width for r in matrix):
        raise ValueError("every row of `values` must have the same number of channels")
    if width != channels:
        raise ValueError(
            f"this model expects {channels} channels per timestep, got {width}. "
            f"Pass a flat list of numbers for a single series instead."
        )
    return matrix, False


@mcp.tool()
def timeseries_info() -> str:
    """Describe the forecasting model: how much history it needs and how far it predicts.

    Call this before `forecast` so you know the required window length.
    """
    cfg = _local_config()
    if not cfg:
        return f"Model {MODEL_ID} is not downloaded."
    return (
        f"Model: {MODEL_ID}\n"
        f"  architecture     : {(cfg.get('architectures') or ['?'])[0]}\n"
        f"  context_length   : {cfg.get('context_length')} timesteps of history REQUIRED\n"
        f"  prediction_length: {cfg.get('prediction_length')} timesteps forecast\n"
        f"  channels         : {cfg.get('num_input_channels')} "
        f"(a single series is replicated across them; channels are independent)\n"
        f"  runs on          : CPU\n\n"
        "Use `forecast` with at least context_length numbers. It predicts the NEXT "
        "prediction_length steps; it does not extrapolate further."
    )


@mcp.tool()
def forecast(values: list, horizon: int = 0) -> str:
    """Forecast the continuation of a numeric time series.

    `values` is either a flat list of numbers (one series) or a list of equal-length rows
    (one row per timestep, one column per channel). Supply at least `context_length`
    points -- call `timeseries_info` for that number. Only the most recent
    `context_length` are used; earlier points are ignored.

    `horizon` trims the output; 0 returns the model's full prediction length. The model
    cannot forecast further than its trained prediction length.

    Returns the forecast values plus simple summary statistics. The numbers are a
    statistical extrapolation of the pattern supplied, not a prediction of real-world
    events, and carry no confidence interval.
    """
    try:
        import torch  # noqa: PLC0415

        model, config = _load()
    except Exception as exc:  # noqa: BLE001 - reported to the calling model as text
        return f"error: could not load {MODEL_ID}: {type(exc).__name__}: {exc}"

    ctx = int(config.context_length)
    pred = int(config.prediction_length)
    channels = int(config.num_input_channels)

    try:
        matrix, univariate = _as_matrix(list(values), channels)
    except ValueError as exc:
        return f"error: {exc}"

    if len(matrix) < ctx:
        return (
            f"error: need at least {ctx} timesteps of history, got {len(matrix)}. "
            f"This model reads a fixed {ctx}-step window."
        )
    window = matrix[-ctx:]  # most recent context only

    try:
        with torch.no_grad():
            out = model(past_values=torch.tensor([window], dtype=torch.float32))
        # (batch, prediction_length, channels)
        preds = out.prediction_outputs[0].tolist()
    except Exception as exc:  # noqa: BLE001
        return f"error: forecast failed: {type(exc).__name__}: {exc}"

    series = [row[0] for row in preds] if univariate else preds
    if horizon and 0 < horizon < pred:
        series = series[:horizon]

    flat = series if univariate else [row[0] for row in series]
    last = window[-1][0]
    first_f, last_f = flat[0], flat[-1]
    lines = [
        f"Forecast from {MODEL_ID} ({len(flat)} of {pred} steps"
        + (", channel 0" if not univariate else "")
        + ")",
        f"  last observed : {last:.6g}",
        f"  first forecast: {first_f:.6g}",
        f"  last forecast : {last_f:.6g}",
        f"  change        : {last_f - last:+.6g} ({(last_f - last) / last * 100:+.2f}%)"
        if last
        else "  change        : n/a (last observed is zero)",
        f"  min / max     : {min(flat):.6g} / {max(flat):.6g}",
        f"  mean          : {sum(flat) / len(flat):.6g}",
        "",
        "values: " + ", ".join(f"{v:.6g}" for v in flat),
        "",
        "Note: a statistical extrapolation of the supplied pattern only. No confidence "
        "interval, and no knowledge of events outside the series.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
