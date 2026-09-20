# FreeToken - native Windows port

> **Where this came from.** FreeSwarm's copy of the engine derives from the native-Windows
> port of FreeToken, not from upstream directly. This document describes that port and is
> carried here with it. Both are Apache-2.0; see [`NOTICE`](NOTICE).
>
> - Windows port: **https://github.com/UnsignedChad/freetoken-windows**
> - Upstream engine: **https://github.com/FlashML-org/FreeToken**
>
> Go to the port for its own issues and updates, and to upstream for the engine itself.

A native-Windows port of [FreeToken](https://github.com/FlashML-org/FreeToken) (edge-native
MoE serving, arXiv:2608.16157), Apache-2.0. Upstream is Linux-only; this fork builds and
serves natively on Windows with MSVC + CUDA.

**Verified:** builds with MSVC v14.5x, serves `openai/gpt-oss-20b` (MXFP4, fused) on an
RTX 3090 (CUDA 13.2, Python 3.13) at **~142 tok/s decode**. KV-cache allocation and the
JIT kernel path both work natively (WSL2's pinned-memory + allocator limits do not apply).

## Requirements

- Windows 10/11, NVIDIA GPU + recent driver (CUDA 13 class)
- Visual Studio 2022+ Build Tools (MSVC, cl.exe)
- CUDA Toolkit 13.x (nvcc) - CUDA **major** must match the torch wheel (cu13)
- Python 3.13

## Install

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip wheel setuptools
:: torch (CUDA 13 Windows build)
.venv\Scripts\python -m pip install "torch>=2.11,<2.12" --index-url https://download.pytorch.org/whl/cu130
:: runtime deps (triton-windows replaces Linux triton; sglang-kernel is Linux-only, dropped;
:: flashinfer optional - serve with --attention-backend triton; tornado is REQUIRED on Windows
:: so pyzmq's asyncio works under uvicorn's Proactor loop)
.venv\Scripts\python -m pip install apache-tvm-ffi flashlib triton-windows tornado einops fastapi ^
  gguf huggingface_hub msgpack modelscope numpy openai partial-json-parser prompt_toolkit ^
  pydantic pyzmq safetensors tqdm transformers uvicorn
:: nvcc runtime toolchain (align nvcc/ptxas/nvvm to one CUDA minor)
.venv\Scripts\python -m pip install "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm<13.1" "nvidia-cuda-crt<13.1"
:: build the two C++ extensions with MSVC
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
.venv\Scripts\python -m pip install -e . --no-build-isolation --no-deps
:: one-time dependency patch (tvm-ffi Windows JIT flags)
.venv\Scripts\python scripts\patch_deps_windows.py
```

## Run (from a vcvars64 environment so nvcc can find cl.exe for the runtime JIT)

```bat
call "<VS>\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_PATH%\bin;C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%
.venv\Scripts\python -m freetoken --model-path <hf_dir> --moe-backend fused ^
  --attention-backend triton --num-pages 8192 --host 127.0.0.1 --port 1919
```

OpenAI-compatible API on :1919 (the request body must include `"model"`).

## Changes vs upstream

Build (`setup.py`)
- MSVC compile flags: `/O2 /std:c++17 /DNOMINMAX /DWIN32_LEAN_AND_MEAN` (no `-pthread`);
  add the Windows CUDA `lib/x64` library dir.

CUDA kernel sources (JIT)
- `kernel/csrc/include/freetoken/warp.cuh`: guard the glibc-only `<sys/cdefs.h>` behind
  `#ifndef _WIN32`.
- `kernel/csrc/include/freetoken/utils.cuh`: define `__always_inline` (a glibc macro absent
  on MSVC) as `__forceinline__`/`__forceinline` on Windows.
- `kernel/csrc/**`: drop the (ignored) explicit template args on `TensorMatcher.with_dtype<...>(arg)`
  calls - MSVC's overload resolution rejects the `typename...` + `requires` pair; the
  arg-taking overload ignores them anyway.
- `kernel/utils.py`: add `cudart.lib` + a quoted `/LIBPATH:"<CUDA>\lib\x64"` to the JIT
  linker flags (tvm-ffi's Windows default omits the CUDA runtime; Linux gets `-lcudart`).

IPC / runtime
- `utils/mp.py` + `scheduler/config.py` + `server/args.py`: ZMQ `ipc://` (unsupported by
  libzmq on Windows) -> TCP loopback via a `zmq_endpoint()` helper (pid-derived ports).
- `_win_compat.py` (new, imported at package init): sets `WindowsSelectorEventLoopPolicy`
  on Windows.
- `tornado` dependency: pyzmq's `zmq.asyncio` needs a selector loop; under uvicorn's
  Proactor loop it bridges via tornado. Install `tornado>=6.1`.

Dependencies
- `triton` -> `triton-windows`; `sglang-kernel` dropped (Linux-only; falls back to triton
  kernels); `flashinfer` optional (serve with `--attention-backend triton`).
- `scripts/patch_deps_windows.py`: post-install fix for tvm-ffi's Windows nvcc flags
  (`-Xcompiler /std:c++17 /O2` -> `-Xcompiler /std:c++20,/O2`, and host `/std:c++20`).

## Why native Windows (vs WSL2)

FreeToken's expert-offload path pins large host buffers with `cudaHostRegister`. Under WSL2
that is capped (~14.5 GiB) and corrupts the GPU allocator, so offload dies at the KV-cache
`cudaMalloc`. Native Windows CUDA has full pinned-memory support, so the offload path can run.

### Offload path (Windows)

The expert-offload path also runs natively (experts pinned in host RAM via `cudaHostRegister`
-- which works on native Windows, unlike WSL2 -- and streamed over PCIe):

- `utils/winio.py` (new): `FILE_FLAG_NO_BUFFERING` sequential reads - the O_DIRECT analog.
  The shard readers in `moe/host_banks.py` and `models/weight.py` use it, falling back to
  buffered `readinto` if it is unusable (sector size > 4096, unaligned buffer, open failure).
- `utils/hostmem.py` (new): cross-platform host-RAM probe (`GlobalMemoryStatusEx` on Windows)
  and the budget below. The previous probe read `/proc/meminfo` only.
- `moe/host_banks.py`, `models/weight.py`: guarded `madvise`/`posix_fadvise` (Unix-only);
  `HostBank.release()` falls back to `DiscardVirtualMemory`.
- `models/loader.py`: `drop_page_cache` is a no-op on Windows (no `posix_fadvise`).

Verified: gpt-oss-20B `--moe-backend offload` serves at ~96-100 tok/s (vs ~142 fused) on the
3090.

### The pinning ceiling (measure this on your machine)

**You cannot page-lock all of host RAM.** There is a hard driver ceiling on
`cudaHostRegister`'d memory well below physical RAM, and **free RAM does not predict it**.

Measured here (127.9 GiB RAM; the two A6000s are in **TCC** mode, so this is *not* the WDDM
shared-system-memory limit -- that does not apply to a TCC device):

| test | result |
|---|---|
| solo, 1 GiB chunks | 69 GiB, then `cudaHostRegister failed: invalid argument`, ~58 GiB still free |
| solo, 4 GiB chunks | 68 GiB (17 regions vs 69) -- a **byte** ceiling, not a region-count one |
| 10 GiB held by another process | the next process reached only **62 GiB** |

That last row is the one that matters. gpt-oss-20b pins 9.5 GiB; Qwen3.6 needs 61.5; the load
reached its final layer and died on a 1.0 GiB bank. The ceiling is shared enough between
processes to matter **even across two different GPUs**.

Only the **experts** are pinned, not the whole checkpoint -- `/api/models` reports
`expert_bytes` per entry, and budgeting against `size_bytes` wrongly rejects models that fit:

| model | checkpoint | pinned |
|---|---|---|
| openai/gpt-oss-20b | 12.8 GiB | 9.5 GiB |
| gpt-oss-120b | 60.8 GiB | 56.8 GiB |
| Qwen3.6-35B-A3B | 67.0 GiB | 61.5 GiB |
| DeepSeek-V4-Flash-0731 | 155.4 GiB | 146.6 GiB |

Budgeting the **sum** of `expert_bytes` against ~52% of RAM (66.5 GiB here) gives the right
verdict for every combination measured:

| combination | pinned | verdict |
|---|---|---|
| Qwen3.6 alone | 61.5 GiB | allowed — solo ceiling is 68-69 |
| gpt-oss-120b alone | 56.8 GiB | allowed |
| gpt-oss-120b + gpt-oss-20b | 66.3 GiB | allowed — 62 GiB was reachable with 10 held |
| **Qwen3.6 + gpt-oss-20b** | **71.0 GiB** | **refused — measured to fail** |
| DeepSeek-V4-Flash | 146.6 GiB | refused |

`preflight_host_ram` in the control plane enforces this before the engine reads a byte off
disk, naming what is already pinned. **0.52 is an empirical fit, not a derived constant** --
the mechanism is a driver/OS limit nothing here can query. Re-measure on different hardware
and override with `FREETOKEN_HOST_PIN_FRACTION` or `FREETOKEN_HOST_PIN_GB`.

### Host RAM budget

Host RAM bounds the offloaded model: experts are pinned (`cudaHostRegister`) for the process
lifetime, so the OS cannot page them out. Oversubscribing does not degrade gracefully - it
makes the whole machine unresponsive. Three things keep that in check:

- **A hard budget, checked before any bank is allocated.** Default 75% of physical RAM, and
  never more than total minus 8 GiB. A load that would exceed it fails immediately with the
  numbers rather than melting the box. Override with `--host-ram-gb <gib>`,
  `FREETOKEN_HOST_RAM_GB`, or `FREETOKEN_HOST_RAM_FRACTION` (0-1);
  `FREETOKEN_HOST_RAM_HEADROOM_GB` sets the reserve left for everything else.
- **Cache-bypassing reads.** Reading a 67 GiB checkpoint with buffered I/O also builds ~67 GiB
  of system file cache on top of the pinned banks. The unbuffered path avoids that entirely at
  the same throughput (measured 434 vs 476 MiB/s - both disk-bound).
- **A RAM-capped prefetch depth.** The parallel reader's in-flight whole-shard buffers are
  anonymous and non-reclaimable; the depth now shrinks to what free RAM leaves after the banks,
  so `--expert-load serial` is no longer needed as a manual workaround (`--expert-load
  serial|parallel` still force the choice).

Sizing rule: bank footprint ~= the checkpoint's expert weights. Qwen3.6-35B-A3B BF16 is
40 layers x 256 experts = ~64 GiB pinned, so it needs a 128 GiB box; the same model at NVFP4
(~19 GB experts) fits in ~22 GB pinned + overhead.
