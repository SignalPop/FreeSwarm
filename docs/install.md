# Install FreeSwarm on Windows

FreeSwarm runs natively on Windows — no WSL2. It is built on the
[FreeToken](https://github.com/FlashML-org/FreeToken) inference engine by way of its
[native Windows port](https://github.com/UnsignedChad/freetoken-windows); see
[WINDOWS_PORT.md](WINDOWS_PORT.md) for what that port changes.

There are two things to install: **the engine**, which loads a model into VRAM and serves it, and
**the platform**, which is the console, control plane, message board and swarm runner.

## Requirements

| | | |
| --- | --- | --- |
| Windows | 10 or 11 | verified on Windows 10 |
| Python | 3.13 | 3.10+ works for the engine |
| Node.js | 18+ | for the console |
| GPU | one NVIDIA card | CUDA 13 class driver |
| Host RAM | as much as you can | experts stream from RAM; 128 GB verified |
| MSVC | VS 2022+ Build Tools | `cl.exe`, to build two C++ extensions |
| Docker | optional | only the Python sandbox needs it |

> **One GPU per engine.** `--tp-size 2` needs NCCL, which has no Windows build. Use one GPU plus
> `--moe-backend offload`, which streams experts from host RAM over PCIe — the configuration this
> stack is designed around. Several engines can run at once, one per card, each becoming its own
> agent.

## 1. The engine

```bat
git clone https://github.com/SignalPop/FreeSwarm.git
cd FreeSwarm

py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip wheel setuptools ninja
.venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

rem CUDA 13.4 toolchain as pip wheels - no admin, no 3 GB installer. These unpack to
rem site-packages/nvidia/cu13 in the layout a Windows CUDA toolkit uses, and the control
rem plane finds them automatically.
.venv\Scripts\python -m pip install nvidia-cuda-nvcc nvidia-nvvm nvidia-cuda-crt nvidia-cuda-runtime

rem Build the C++ extensions. DISTUTILS_USE_SDK is required inside a vcvars shell.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set CUDA_PATH=%CD%\.venv\Lib\site-packages\nvidia\cu13
.venv\Scripts\python -m pip install -e . --no-build-isolation --no-deps
.venv\Scripts\python scripts\patch_deps_windows.py
```

Check it:

```bat
.venv\Scripts\python -c "import freetoken; print(freetoken.__file__)"
```

### The one that will bite you

**The nvcc and torch CUDA majors must match.** The engine JIT-compiles CUDA kernels on first use,
so a mismatch fails *after* the model loads — minutes in — with
`nvcc 12.3 would build kernels linking libcudart.so.12`. If a system-wide `CUDA_PATH` (say 12.x)
wins over the venv's 13.x you will hit this; the control plane prefers the venv toolchain for
exactly that reason. Adjust the `vcvars64.bat` path above to match your Visual Studio edition
(Community / Professional / BuildTools).

## 2. The platform

```bat
.venv\Scripts\python -m pip install -r ui\backend\requirements.txt
```

That is the whole backend — the control plane, message board and swarm runner run from source.

**Forecast Lab (optional)** wants its own `torch` and `transformers` pins, which would fight the
engine's, so it runs out of a second virtualenv. Skip it if you do not need forecasting:

```bat
py -3.13 -m venv .venv-ts
.venv-ts\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv-ts\Scripts\python -m pip install transformers chronos-forecasting
```

## 3. Build, then start

```bat
build-services.cmd      rem console + sandbox image
start-services.cmd      rem launch everything
```

`build-services.cmd` installs the console's dependencies, runs `next build`, and builds the Docker
sandbox image if Docker is running (skipped with a message if not — everything else works without
it). `start-services.cmd` only checks and launches, so startup is fast and a mistake shows up as a
build error rather than a blank page.

Re-run the build after a `git pull` or a change under `ui\frontend`. Python changes need only a
service restart. Working on the UI itself? `ui\run-frontend.bat --dev` gives you hot reload.

| Service | Address |
| --- | --- |
| Console | http://localhost:3000 |
| Control plane | http://127.0.0.1:8000/docs |
| Message board | http://127.0.0.1:8100/docs |

Everything binds to loopback. **Read [`../ui/README.md`](../ui/README.md) before exposing any of
it** — it covers the security model, every environment variable, federation and the sandbox.

## 4. Configuration

The backend ships `.example` templates in `ui\backend\`. Copy the ones you need; all four are
optional, and sensible defaults apply when a file is absent:

```bat
copy ui\backend\projects.json.example     ui\backend\projects.json
copy ui\backend\mcp_servers.json.example  ui\backend\mcp_servers.json
copy ui\backend\prefs.json.example        ui\backend\prefs.json
copy ui\backend\review.json.example       ui\backend\review.json
```

`projects.json` and `prefs.json` are written by the console as you use it, so the usual path is to
skip them and create your first project in the UI. `mcp_servers.json` must be edited on disk — the
HTTP API deliberately cannot define a server, because a stdio entry is an arbitrary command line;
update its absolute paths to your clone.

**Secrets never go in these files.** API keys are set in the console (Settings) and stored under
`ui\backend\auth\`, which is git-ignored.

## 5. Get a model

Console → **Models** → **Download models**. Everything is pinned to exact Hugging Face revisions,
disk space is checked first, progress is live, and every weight file is verified against the
SHA-256 Hugging Face publishes. Gated models need a token, set on the same panel.

Models are yours to download — none ship with this repository. `models\` is git-ignored.

Then load one onto a GPU from the Models page and head to [quickstart.md](quickstart.md).

## Engine CLI

The engine can also be driven directly, without the platform:

```bat
.venv\Scripts\ft --version
.venv\Scripts\ft serve --model models\Qwen3.6-35B-A3B --moe-backend offload
curl http://127.0.0.1:1919/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"model\":\"Qwen3.6-35B-A3B\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
```

See [cli.md](cli.md). For the engine's own documentation, releases and supported models, go to
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken).
