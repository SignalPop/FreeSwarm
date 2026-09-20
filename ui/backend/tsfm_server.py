"""Serves ONE time-series forecasting model on ONE GPU over HTTP.

The control plane starts one of these per loaded time-series model (see app/tsfm.py), with
CUDA_VISIBLE_DEVICES set to the card it chose, so inside this process the model is always
on `cuda:0`. It runs from `.venv-ts`, not the engine venv: `chronos-forecasting` pins
transformers versions that would fight the LLM engine's, and a separate process is what lets
the two dependency sets coexist.

Why a process per model rather than one server holding them all: a forecaster that crashes or
leaks VRAM takes down only itself, and stopping it returns exactly its memory to the card --
which matters because these share GPUs with LLM engines that must never be starved.

Models are adapted behind one interface, `Forecaster.forecast(series, horizon, quantiles)`:

* **Chronos / Chronos-Bolt** -- general zero-shot forecasters over a single series, returning
  a distribution (quantiles). This is the one to reach for.
* **PatchTST** -- fixed to the channel count and context length it was trained on. The IBM
  Granite checkpoint is 7 channels x 512 steps (the ETTh1 benchmark layout), so it is not a
  general forecaster; a single series is accepted only by repeating it across channels, and
  the response says so rather than pretending the result means much.
* **Chronos-2** -- multivariate and covariate-aware: forecast one or several target series
  jointly while reading any number of other series as past covariates (and, where values
  are known ahead, future covariates). Accepts ``inputs`` (a list of {target,
  past_covariates, future_covariates}); a plain ``series`` works univariately.
* **Kronos** (NeoQuasar) -- a foundation model for financial candlesticks: it reads OHLCV bars
  (up to 512) with their timestamps and generates future candles autoregressively. Given
  ``candles`` it is used as intended; given a plain ``series`` it synthesises candles from it
  (degraded, and the response says so). Quantiles come from independently sampled paths.
  Model code is vendored in tsfm_vendor/kronos (pinned commit, reviewed; see VENDOR.md).

Adding a model family is one `Forecaster` subclass and one line in `_ADAPTERS`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MAX_HORIZON = 1024
MAX_POINTS = 100_000


class ForecastRequest(BaseModel):
    # One series (list of numbers) or several (list of lists) -- several means channels for
    # a multivariate model, or a batch for a univariate one.
    series: list[float] | list[list[float]] | None = None
    horizon: int = Field(24, ge=1, le=MAX_HORIZON)
    quantiles: list[float] | None = None
    # Candle models: a batch of [bars x (open, high, low, close[, volume])] histories, their
    # bar timestamps (ISO strings), the bar spacing in seconds, and sample paths to draw.
    candles: list[list[list[float]]] | None = None
    timestamps: list[list[str]] | None = None
    freq_seconds: int | None = None
    samples: int | None = Field(None, ge=1, le=200)
    # Covariate models (Chronos-2): [{"target": [..] | [[..],[..]], "past_covariates": {name: [..]},
    # "future_covariates": {name: [..horizon]}}], all items with the same schema.
    inputs: list[dict[str, Any]] | None = None


def _as_2d(series: Any) -> list[list[float]]:
    if not series:
        raise HTTPException(status_code=400, detail="series is empty")
    rows = series if isinstance(series[0], list) else [series]
    total = sum(len(r) for r in rows)
    if total > MAX_POINTS:
        raise HTTPException(status_code=400, detail=f"too many points ({total} > {MAX_POINTS})")
    clean: list[list[float]] = []
    for r in rows:
        if not r:
            raise HTTPException(status_code=400, detail="a series is empty")
        vals = [float(v) for v in r]
        if any(not math.isfinite(v) for v in vals):
            raise HTTPException(status_code=400, detail="series contains NaN or infinity")
        clean.append(vals)
    return clean


MAX_INPUT_POINTS = 4_000_000


class Forecaster:
    supports_covariates = False
    family = "unknown"
    context_length: int | None = None
    native_horizon: int | None = None
    channels: int | None = None
    probabilistic = False

    def forecast(self, rows: list[list[float]], horizon: int, quantiles: list[float] | None) -> dict:
        raise NotImplementedError


class ChronosForecaster(Forecaster):
    family = "chronos"
    probabilistic = True

    def __init__(self, path: str, device: str):
        from chronos import BaseChronosPipeline  # noqa: PLC0415

        # bf16 on GPU: half the memory, and Chronos-Bolt is trained robust to it.
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.pipe = BaseChronosPipeline.from_pretrained(path, device_map=device, torch_dtype=dtype)
        cfg = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        cc = cfg.get("chronos_config") or {}
        self.context_length = cc.get("context_length")
        self.native_horizon = cc.get("prediction_length")
        self.default_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    def forecast(self, rows, horizon, quantiles):
        qs = sorted(set(quantiles or self.default_quantiles) | {0.5})
        # Chronos takes a batch of independent univariate series. Longer history than the
        # model's context is trimmed to the most recent points -- the model would do it anyway.
        ctx = [torch.tensor(r[-self.context_length :] if self.context_length else r) for r in rows]
        q_tensor, _mean = self.pipe.predict_quantiles(
            ctx, prediction_length=horizon, quantile_levels=qs
        )
        q = q_tensor.float().cpu().numpy()  # [batch, horizon, len(qs)]
        mid = qs.index(0.5)
        out = []
        for b in range(q.shape[0]):
            out.append(
                {
                    "median": q[b, :, mid].tolist(),
                    "quantiles": {str(level): q[b, :, i].tolist() for i, level in enumerate(qs)},
                }
            )
        return {"forecasts": out, "notes": []}


class PatchTSTForecaster(Forecaster):
    family = "patchtst"

    def __init__(self, path: str, device: str):
        from transformers import PatchTSTForPrediction  # noqa: PLC0415

        self.device = device
        self.model = PatchTSTForPrediction.from_pretrained(path).to(device).eval()
        c = self.model.config
        self.context_length = int(c.context_length)
        self.native_horizon = int(c.prediction_length)
        self.channels = int(c.num_input_channels)

    def forecast(self, rows, horizon, quantiles):
        notes: list[str] = []
        if len(rows) == self.channels:
            channels = rows
        elif len(rows) == 1:
            # Honest about it: this model was trained on a specific multichannel layout.
            channels = rows * self.channels
            notes.append(
                f"This PatchTST checkpoint expects {self.channels} channels; the single series "
                "was repeated across all of them. Treat the result as a rough baseline -- "
                "Chronos is the right model for a single series."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"this model takes 1 or {self.channels} series; got {len(rows)}",
            )
        L = self.context_length
        padded = []
        for r in channels:
            r = r[-L:]
            if len(r) < L:
                # Left-pad with the first observed value: flat history is the least-committal
                # assumption, and the model's own scaling normalises the level.
                r = [r[0]] * (L - len(r)) + r
            padded.append(r)
        x = torch.tensor(padded, dtype=torch.float32, device=self.device).T.unsqueeze(0)  # [1, L, C]
        with torch.no_grad():
            pred = self.model(past_values=x).prediction_outputs[0]  # [H_native, C]
        if horizon > self.native_horizon:
            notes.append(
                f"Horizon capped at {self.native_horizon}: this model predicts a fixed "
                f"{self.native_horizon} steps and does not roll forward."
            )
        h = min(horizon, self.native_horizon)
        p = pred[:h].float().cpu().numpy()
        n = 1 if len(rows) == 1 else self.channels
        return {
            "forecasts": [{"median": p[:, c].tolist(), "quantiles": None} for c in range(n)],
            "notes": notes,
        }


class Chronos2Forecaster(Forecaster):
    family = "chronos2"
    probabilistic = True
    supports_covariates = True
    channels = None  # any number of targets and covariates

    def __init__(self, path: str, device: str):
        from chronos import Chronos2Pipeline  # noqa: PLC0415

        self.pipe = Chronos2Pipeline.from_pretrained(path, device_map=device)
        cfg = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        cc = cfg.get("chronos_config") or {}
        self.context_length = cc.get("context_length")
        self.native_horizon = (cc.get("max_output_patches") or 64) * (cc.get("output_patch_size") or 16)
        self.trained_quantiles = cc.get("quantiles") or []

    @staticmethod
    def _arr(v) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        if not np.isfinite(a).all():
            raise HTTPException(status_code=400, detail="inputs contain NaN or infinity")
        return a

    def forecast(self, rows, horizon, quantiles, inputs=None):
        qs = sorted(set(quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]) | {0.5})
        notes: list[str] = []
        if inputs:
            prepared = []
            schema = None
            for it in inputs:
                tgt = self._arr(it.get("target") or [])
                if tgt.size == 0:
                    raise HTTPException(status_code=400, detail="each input needs a target")
                if self.context_length and tgt.shape[-1] > self.context_length:
                    tgt = tgt[..., -self.context_length:]
                hist = tgt.shape[-1]
                past = {k: self._arr(v)[-hist:] for k, v in (it.get("past_covariates") or {}).items()}
                fut = {k: self._arr(v) for k, v in (it.get("future_covariates") or {}).items()}
                if any(len(v) != hist for v in past.values()):
                    raise HTTPException(status_code=400, detail="past covariates must match the target's history length")
                if any(len(v) != horizon for v in fut.values()):
                    raise HTTPException(status_code=400, detail="future covariates must have `horizon` values")
                if not set(fut) <= set(past):
                    raise HTTPException(status_code=400, detail="every future covariate needs its past values too")
                sig = (tgt.shape[0] if tgt.ndim == 2 else 1, tuple(sorted(past)), tuple(sorted(fut)))
                if schema is None:
                    schema = sig
                elif sig != schema:
                    raise HTTPException(status_code=400, detail="all inputs in one request must share the same targets/covariates")
                item = {"target": tgt}
                if past:
                    item["past_covariates"] = past
                if fut:
                    item["future_covariates"] = fut
                prepared.append(item)
        else:
            prepared = [self._arr(r[-self.context_length:] if self.context_length else r) for r in rows]
        if horizon > (self.native_horizon or horizon):
            notes.append(f"Horizon beyond {self.native_horizon} steps is rolled forward autoregressively (less accurate).")
        q_list, _mean = self.pipe.predict_quantiles(prepared, prediction_length=horizon, quantile_levels=qs)
        mid = qs.index(0.5)
        out = []
        for q in q_list:                      # per input: [n_variates, horizon, len(qs)]
            q = q.float().cpu().numpy()
            variates = [{"median": q[v, :, mid].tolist(),
                         "quantiles": {str(level): q[v, :, i].tolist() for i, level in enumerate(qs)}}
                        for v in range(q.shape[0])]
            out.append({**variates[0], "variates": variates})
        return {"forecasts": out, "notes": notes}


class KronosForecaster(Forecaster):
    family = "kronos"
    probabilistic = True
    channels = 5  # open, high, low, close, volume (+ amount derived)
    context_length = 512
    native_horizon = None
    DEFAULT_SAMPLES = 20

    def __init__(self, path: str, device: str):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tsfm_vendor.kronos import Kronos, KronosTokenizer  # noqa: PLC0415
        from tsfm_vendor.kronos.kronos import auto_regressive_inference  # noqa: PLC0415

        self._infer = auto_regressive_inference
        self.device = device
        tok = _kronos_tokenizer_for(Path(path))
        self.tokenizer = KronosTokenizer.from_pretrained(str(tok)).to(device).eval()
        self.model = Kronos.from_pretrained(path).to(device).eval()
        self.tokenizer_path = str(tok)

    @staticmethod
    def _stamps(ts: list, freq_s: int, horizon: int):
        import pandas as pd

        x = pd.to_datetime(pd.Series(ts))
        if getattr(x.dt, "tz", None) is not None:
            x = x.dt.tz_convert("UTC").dt.tz_localize(None)
        step = pd.Timedelta(seconds=freq_s)
        y = pd.Series([x.iloc[-1] + step * (k + 1) for k in range(horizon)])

        def feats(t):
            return np.stack([t.dt.minute, t.dt.hour, t.dt.weekday, t.dt.day, t.dt.month], axis=1).astype(np.float32)

        return feats(x), feats(y)

    def forecast(self, rows, horizon, quantiles, candles=None, timestamps=None, freq_seconds=None, samples=None):
        notes: list[str] = []
        qs = sorted(set(quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]) | {0.5})
        n_samples = int(samples or self.DEFAULT_SAMPLES)
        freq = int(freq_seconds or 60)
        if candles is None:
            # Plain series: synthesise candles (open = previous close; high/low = the pair's
            # extremes; no volume). Kronos was trained on real OHLCV, so this is a fallback.
            candles = []
            for r in rows:
                prev = [r[0]] + r[:-1]
                candles.append([[o, max(o, c), min(o, c), c, 0.0] for o, c in zip(prev, r)])
            notes.append("Kronos forecasts candles; a plain series was converted to synthetic candles "
                         "(no real high/low/volume). Pass OHLCV candles for its intended use.")
        if not timestamps:
            import datetime as dt

            end = dt.datetime.utcnow().replace(second=0, microsecond=0)
            timestamps = [[(end - dt.timedelta(seconds=freq * (len(c) - 1 - i))).isoformat() for i in range(len(c))]
                          for c in candles]
            notes.append(f"No timestamps given: assumed {freq}s bars ending now (Kronos uses time-of-day features).")
        out = []
        for c, ts in zip(candles, timestamps):
            arr = np.asarray(c, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 4:
                raise HTTPException(status_code=400, detail="each candle needs open, high, low, close[, volume]")
            if len(ts) != len(arr):
                raise HTTPException(status_code=400, detail="timestamps must match the candles one to one")
            arr, ts = arr[-self.context_length:], ts[-self.context_length:]
            vol = arr[:, 4] if arr.shape[1] > 4 else np.zeros(len(arr), dtype=np.float32)
            amount = vol * arr[:, :4].mean(axis=1)
            x = np.column_stack([arr[:, :4], vol, amount]).astype(np.float32)
            if not np.isfinite(x).all():
                raise HTTPException(status_code=400, detail="candles contain NaN or infinity")
            mean, std = x.mean(axis=0), x.std(axis=0)
            xn = np.clip((x - mean) / (std + 1e-5), -5, 5)
            x_stamp, y_stamp = self._stamps(ts, freq, horizon)
            # Independent sample paths: batch the same history n times with sample_count=1
            # (the library's own sample_count AVERAGES paths, which would erase the spread).
            xt = torch.from_numpy(np.repeat(xn[None], n_samples, axis=0)).to(self.device)
            xs = torch.from_numpy(np.repeat(x_stamp[None], n_samples, axis=0)).to(self.device)
            ys = torch.from_numpy(np.repeat(y_stamp[None], n_samples, axis=0)).to(self.device)
            with torch.no_grad():
                paths = self._infer(self.tokenizer, self.model, xt, xs, ys, self.context_length, horizon,
                                    5, 1.0, 0, 0.9, 1, False)[:, -horizon:, :]      # [n, h, 6]
            paths = paths * (std + 1e-5) + mean
            close = paths[:, :, 3]
            out.append({
                "median": np.quantile(close, 0.5, axis=0).tolist(),
                "quantiles": {str(q): np.quantile(close, q, axis=0).tolist() for q in qs},
                "candles_mean": paths.mean(axis=0)[:, :5].tolist(),   # [h, o/h/l/c/v]
                "high_q90": np.quantile(paths[:, :, 1], 0.9, axis=0).tolist(),
                "low_q10": np.quantile(paths[:, :, 2], 0.1, axis=0).tolist(),
                "samples": n_samples,
            })
        return {"forecasts": out, "notes": notes}


def _kronos_tokenizer_for(model_dir: Path) -> Path:
    """The tokenizer a Kronos checkpoint needs: -2k for Kronos-mini, -base for small/base.
    Looked for beside the model: in the Hugging Face cache and in a models folder."""
    name = str(model_dir).lower()
    want = "Kronos-Tokenizer-2k" if "kronos-mini" in name else "Kronos-Tokenizer-base"
    candidates: list[Path] = []
    for parent in model_dir.parents:
        hub = parent / f"models--NeoQuasar--{want}" / "snapshots"
        if hub.is_dir():
            candidates += sorted(hub.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if (parent / want).is_dir():
            candidates.append(parent / want)
    for c in candidates:
        if (c / "config.json").is_file() and (c / "model.safetensors").is_file():
            return c
    raise SystemExit(f"Kronos needs its tokenizer NeoQuasar/{want}: download it (Models > Download models) "
                     f"into the same place as the model ({model_dir.parent}).")


def _is_kronos(cfg: dict) -> bool:
    return {"s1_bits", "s2_bits", "n_layers", "learn_te"} <= cfg.keys() and "d_in" not in cfg


# config "architectures" -> adapter.
_ADAPTERS: dict[str, type[Forecaster]] = {
    "Chronos2Model": Chronos2Forecaster,
    "ChronosBoltModelForForecasting": ChronosForecaster,
    "ChronosModelForForecasting": ChronosForecaster,
    "PatchTSTForPrediction": PatchTSTForecaster,
}


def adapter_for(path: str) -> type[Forecaster]:
    cfg = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    for arch in cfg.get("architectures") or []:
        if arch in _ADAPTERS:
            return _ADAPTERS[arch]
    # Original (non-Bolt) Chronos checkpoints are plain T5 with a chronos_config block.
    if "chronos_config" in cfg:
        return ChronosForecaster
    if _is_kronos(cfg):
        return KronosForecaster
    raise SystemExit(f"no time-series adapter for architectures={cfg.get('architectures')}")


def build_app(path: str, name: str) -> FastAPI:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    forecaster = adapter_for(path)(path, device)
    load_s = time.time() - t0
    gpu_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
    print(f"[tsfm] {name}: {forecaster.family} on {device} ({gpu_name}) in {load_s:.1f}s", flush=True)

    app = FastAPI(title=f"FreeToken time-series: {name}")

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "model": name,
            "family": forecaster.family,
            "device": device,
            "gpu": gpu_name,
            "vram_bytes": torch.cuda.memory_allocated(0) if device.startswith("cuda") else 0,
            "context_length": forecaster.context_length,
            "native_horizon": forecaster.native_horizon,
            "channels": forecaster.channels,
            "probabilistic": forecaster.probabilistic,
            "supports_covariates": forecaster.supports_covariates,
            "load_seconds": round(load_s, 1),
        }

    @app.post("/forecast")
    def forecast(req: ForecastRequest) -> dict:
        if req.quantiles and any(not 0 < q < 1 for q in req.quantiles):
            raise HTTPException(status_code=400, detail="quantiles must be between 0 and 1")
        t = time.time()
        if isinstance(forecaster, Chronos2Forecaster) and req.inputs:
            size = 0
            for it in req.inputs:
                tg = it.get("target") or []
                size += sum(len(x) for x in tg) if tg and isinstance(tg[0], list) else len(tg)
                size += sum(len(v) for v in (it.get("past_covariates") or {}).values())
            if size > MAX_INPUT_POINTS:
                raise HTTPException(status_code=400, detail=f"too many points ({size} > {MAX_INPUT_POINTS}); send fewer inputs per request")
            result = forecaster.forecast([], req.horizon, req.quantiles, inputs=req.inputs)
        elif isinstance(forecaster, KronosForecaster):
            if not req.candles and not req.series:
                raise HTTPException(status_code=400, detail="send candles (or a series)")
            if req.candles and sum(len(c) for c in req.candles) > MAX_POINTS:
                raise HTTPException(status_code=400, detail=f"too many bars (> {MAX_POINTS})")
            result = forecaster.forecast(_as_2d(req.series) if req.series else [], req.horizon, req.quantiles,
                                         candles=req.candles, timestamps=req.timestamps,
                                         freq_seconds=req.freq_seconds, samples=req.samples)
        else:
            if not req.series:
                raise HTTPException(status_code=400, detail="send a series")
            result = forecaster.forecast(_as_2d(req.series), req.horizon, req.quantiles)
        return {
            "model": name,
            "family": forecaster.family,
            "horizon": req.horizon,
            "seconds": round(time.time() - t, 3),
            **result,
        }

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--name", required=True, help="display name")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    app = build_app(args.model, args.name)
    # Loopback only: the control plane is the only client, and it proxies for the console.
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
