<div align="center">
  <h1>FreeSwarm</h1>
  <p><b>A multi-agent research platform built on the FreeToken inference engine.</b></p>
  <p>Put several local models to work on one problem, together, with shared memory, a shared code library, and a scoreboard that refuses to be fooled.</p>
</div>

---

## FreeSwarm and FreeToken

**FreeToken runs the models. FreeSwarm puts them to work.**

[FreeToken](https://github.com/FlashML-org/FreeToken) is an edge-native Mixture-of-Experts serving
engine: it loads a frontier-scale open-weight model into GPU memory and serves it fast, on hardware
you already own. That is its whole job, and it is very good at it.

**FreeSwarm is not a replacement for FreeToken, and not a competitor to it.** This repository
includes the FreeToken engine source — by way of its native-Windows port — and adds the FreeSwarm
platform on top of it. FreeToken keeps its own name throughout: the `freetoken` package, the `ft`
CLI, the engine subprocess, and the `FREETOKEN_*` settings that configure GPU placement and kernels
are all upstream work, licensed Apache-2.0 and credited in [`NOTICE`](NOTICE).

**Go upstream for the engine itself.** FreeSwarm tracks FreeToken; it does not speak for it. For
engine documentation, releases, supported models, benchmarks and updates, always use the source:

> ### → [github.com/FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)
>
> [Install](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md) ·
> [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md) ·
> [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md) ·
> [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md) ·
> [Desktop app](https://www.flashml.ai/)

**By way of the Windows port.** Upstream FreeToken is Linux-only. FreeSwarm runs on Windows because
of [**UnsignedChad/freetoken-windows**](https://github.com/UnsignedChad/freetoken-windows), a native
port that builds the engine with MSVC + CUDA — no WSL2, and none of its pinned-memory or allocator
limits. That port is where this repository's engine source comes from, and it is the right place for
issues with the Windows build itself. See [`docs/WINDOWS_PORT.md`](docs/WINDOWS_PORT.md).

```
FlashML-org/FreeToken  ──►  UnsignedChad/freetoken-windows  ──►  SignalPop/FreeSwarm
   the MoE engine              native Windows build              the swarm platform
```

If you only want to run a model locally and chat with it, **you want FreeToken, not this.** Use
FreeSwarm when you want a *team* of models working a problem over hours, keeping what they learn.

---

## How it all fits together

```
                                   ┌────────────────────────────┐
   YOU ──────────────────────────► │   Console  ·  :3000        │  Next.js
                                   │   Projects · Swarm · Models│  one origin,
                                   │   Library · Network · Logs │  proxies both
                                   └─────────────┬──────────────┘  backends
                                                 │
                    /api/*  ───────────────┬─────┴──────┬────────────  /mb/*
                                           ▼            ▼
             ┌───────────────────────────────────┐   ┌──────────────────────────┐
             │   CONTROL PLANE   ·  :8000        │   │  MESSAGE BOARD  ·  :8100 │
             │                                   │   │                          │
             │  engine lifecycle & telemetry     │   │  channels, sessions,     │
             │  objectives · scoring · leaderbd  │◄─►│  tasks + leases,         │
             │  code library · projects · MCP    │   │  blackboard, checkpoints │
             │  external review · federation     │   │                          │
             └───┬───────────┬───────────┬───────┘   └────────────┬─────────────┘
                 │           │           │                        │
        spawns   │           │ mounts    │ relays                 │ claims tasks
                 ▼           ▼           ▼                        ▼
      ┌──────────────────┐ ┌──────────┐ ┌────────────┐  ┌────────────────────────┐
      │  FreeToken       │ │ SANDBOX  │ │ OTHER      │  │   SWARM RUNNER         │
      │  ENGINE  :1919   │ │ docker   │ │ COMPUTERS  │  │   one agent per model  │
      │                  │ │          │ │  TLS :8443 │  │                        │
      │  model in VRAM   │ │ 1 per run│ │            │  │  team_post/team_board  │
      │  experts stream  │ │ no net   │ │ model@node │  │  run_python            │
      │  from host RAM   │ │ ro data  │ │            │  │  library_save/get      │
      │                  │ │          │ │            │  │  submit_candidate      │
      │  (upstream)      │ │ OPTIONAL │ │  OPTIONAL  │  │  query_sql / forecast  │
      └──────────────────┘ └──────────┘ └────────────┘  └───────────┬────────────┘
                 ▲                                                  │
                 └──────────────────────────────────────────────────┘
                        agents think with the loaded models
```

**The flow of one candidate:** the swarm runner gives an agent the objective, the team board and
the code library → the agent reasons with a model served by FreeToken → writes Python → it runs in
the sandbox against read-only project data → `submit_candidate` → the control plane scores it
*itself* from prices, re-runs it with future rows deleted to catch look-ahead, has a second model
audit the source, and optionally sends it to Claude for external review → it lands on the
leaderboard, or it is disqualified and the reason becomes a team lesson, a project pitfall, a board
post, and a quarantine on the library modules it was built on.

---

## What FreeSwarm adds

A serving engine answers one request at a time and remembers nothing. Everything below is the
difference between that and a research team.

### Swarm collaboration
Every loaded model becomes an **agent** with its own identity, working the same objective in
parallel. Agents read each other's work, build on each other's code, and are credited for it — the
console tracks who built on whom, how often each agent reused someone else's module, and how many
iterations each has run. One agent's breakthrough becomes the whole team's starting point.

### The message board
A persistent, multi-channel bus (`#general`, `#planning`, `#team`, `#results`, `#errors`) that agents
post to and read from with `team_post` and `team_board`. It is the swarm's shared working memory:
sessions, task claims with leases, checkpoints and a blackboard survive restarts, so an objective
worked overnight is still coherent in the morning. It runs as its own service with its own database.

### The Python sandbox
Agents write and run real code. Each run executes in a throwaway Docker container with **no network
route off the host**, dropped capabilities, no new privileges, and hard memory, CPU and PID caps.
The project's data is mounted read-only. This is what makes `run_python` safe to hand to a model.

### Objectives, scoring and a leaderboard that fights back
An objective is a goal with a metric, a dataset and a **holdout split the agents never see**. Agents
submit candidates; the harness scores them itself from prices rather than trusting what the candidate
reports. Then the defences:

- **Look-ahead detection** — every candidate is re-run with future rows deleted and its positions
  compared, so a strategy that reads the future is caught mechanically.
- **Audit** — a second local model reviews the submission's source.
- **External review** *(optional)* — the candidate **and the full transitive closure of the library
  modules it imports** are sent to a frontier Claude model, which hunts the class of bug the
  mechanical test cannot see: positions computed correctly and then mis-aligned onto earlier bars.
- **Demotion that propagates** — disqualifying a result records the reason as a team lesson, a
  project pitfall and a board post, re-crowns the next eligible candidate, and **quarantines the
  library modules the result was built on**, with the reason stamped into their source. Without
  that last step a swarm happily spends the night improving a signal that was already thrown out.

### The code library
A versioned, project-scoped module library agents build up with `library_save` and reuse with
`library_get`. Modules carry a kind, a description, a test, comments, and **evidence** — how many
candidates used each one and how they scored. Reusable work accumulates instead of being buried in
one-off scripts.

### Projects
Separate workspaces, each with its own data folder, allowed model list, SQL tables, connectors and
objectives. A project decides exactly what its agents can reach.

### Forecast Lab
Time-series foundation models (Chronos, Moirai, Granite/PatchTST) served alongside the LLMs and
exposed to agents as a `forecast` tool, so a strategy can consume a real forecast rather than
inventing one.

### MCP connectors
Agents and chat reach outside tools over the Model Context Protocol, with OAuth where needed.
Ships with connectors for engine control, model routing, time series and forecasting.

### LAN federation
Pair several computers and their models join one pool, addressed as `model@computer`.

### The console
One Next.js web UI over all of it: live engine telemetry, GPU and VRAM pressure, the swarm feed,
leaderboards, the library, logs and settings.

### The agent tool surface
`team_post` · `team_board` · `run_python` · `submit_candidate` · `get_candidate` ·
`library_save` · `library_get` · `library_list` · `library_comment` · `query_data` ·
`describe_data` · `list_data` · `query_sql` · `describe_sql_table` · `list_sql_tables` ·
`forecast` · `forecast_feature` · `list_forecasters` · `field_scan` · `regime_map` · `ask_model`

---

# Build and install

## 0. What you need

| | Required | Notes |
| --- | --- | --- |
| OS | Windows 10/11 | verified on Windows 10, MSVC 19.51 (VS 18) |
| Python | 3.13 | 3.10+ supported by the engine |
| Node.js | 18+ | for the console |
| GPU | one NVIDIA card | see the table below |
| Host RAM | **as much as you can** | experts stream from RAM; 128 GB verified |
| Disk | fast NVMe | weights are large (see below) |
| Docker | **optional** | only the Python sandbox needs it |

> **Tensor parallelism does not work on native Windows.** `--tp-size 2` needs NCCL, which has no
> Windows build. Use **one GPU per engine plus `--moe-backend offload`** — that is the configuration
> this stack is designed around. You can still run *several engines*, one per GPU, and each becomes
> its own agent.

## 1. Build the engine (FreeToken)

FreeSwarm expects a virtualenv at the repo root with FreeToken installed into it.

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip wheel setuptools ninja
.venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

rem CUDA 13.4 toolchain as pip wheels - no admin, no 3 GB installer. These unpack to
rem site-packages/nvidia/cu13 in the layout a Windows CUDA toolkit uses, and the control
rem plane finds them automatically.
.venv\Scripts\python -m pip install nvidia-cuda-nvcc nvidia-nvvm nvidia-cuda-crt nvidia-cuda-runtime

rem Build the C++ extensions. DISTUTILS_USE_SDK is required inside a vcvars shell.
call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set CUDA_PATH=%CD%\.venv\Lib\site-packages\nvidia\cu13
.venv\Scripts\python -m pip install -e . --no-build-isolation --no-deps
.venv\Scripts\python scripts\patch_deps_windows.py
```

`build-kernel.cmd` does the vcvars / `CUDA_PATH` / `pip install -e .` steps above for you, and
`build-services.cmd` runs it on every build. It only compiles when `_pinned_tensor` or `_cpu_moe`
is missing or older than `csrc\` / `setup.py`; `build-kernel.cmd --force` always rebuilds.

**The nvcc and torch CUDA majors must match.** If a system-wide `CUDA_PATH` (say 12.x) wins over the
venv's 13.x you get `nvcc 12.3 would build kernels linking libcudart.so.12` — and it fails *after*
the model loads, minutes in. The control plane deliberately prefers the venv toolchain.

Check it:

```bat
.venv\Scripts\python -c "import freetoken; print(freetoken.__file__)"
```

## 2. Build the FreeSwarm backend

```bat
.venv\Scripts\python -m pip install -r ui\backend\requirements.txt
```

That is the whole backend build — the control plane, the message board and the swarm runner are
Python and run straight from source.

**Forecast Lab needs a second virtualenv.** The time-series models want their own `torch` and
`transformers` pins, which would fight the engine's, so `tsfm_server.py` runs out of `.venv-ts`
instead (`app/tsfm.py` launches it, pinned to one GPU). Skip this if you do not need forecasting:

```bat
py -3.13 -m venv .venv-ts
.venv-ts\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv-ts\Scripts\python -m pip install transformers chronos-forecasting
```

Point `FREESWARM_TS_PYTHON` elsewhere if you keep it somewhere other than `.venv-ts`.

## 3. Build the console

```bat
cd ui\frontend && npm install
```

`start-services.cmd` does this for you on first run if `node_modules` is missing.

## 4. Build the sandbox image — optional

**Docker is optional.** Engines, the swarm, the board, scoring and the console all run without it.
Only `run_python` — the Run button in chat and the agents' code execution — needs the sandbox.

```bat
ui\run-sandbox.bat              rem builds freeswarm-sandbox:latest on first run
ui\run-sandbox.bat --rebuild    rem after editing ui\sandbox\requirements.txt
```

The image is `python:3.12-slim` plus pandas, matplotlib, openpyxl, python-docx and friends — chosen
so model-written code works on the first try instead of failing on a missing import. The script does
not just check the image exists; it **runs a container and imports those libraries**, so a
half-built or architecture-mismatched image fails here rather than at your first Run click.

There is no long-lived container. The control plane starts **one container per run** with
`--network none`, a bind-mounted working directory and the project's data read-only. That flag is
not cosmetic: on Docker Desktop/WSL2, `--internal` breaks published ports and disabling bridge
masquerade is bypassed by the VM's own NAT (both measured). `--network none` is the only
arrangement that verifiably blocks egress, and it rules out a listening socket by construction.

## 5. Download models

Console → **Models** → **Download models**. Everything is pinned to exact Hugging Face revisions, so
a second computer gets byte-identical weights — which federation relies on.

- **Only what the engine loads.** gpt-oss repos carry the weights three times (safetensors,
  `original/`, `metal/`); only the safetensors are fetched — 61 GiB instead of 182 GiB.
- **No silent code.** Pickle/HDF5 weights (`.bin .pt .pth .pkl .ckpt .h5`) are never fetched. `.py`
  files come only for a known model at its pinned revision, or for a custom repo when you tick
  *Include the repo's Python code* — with every file named first.
- **Disk space is checked first, progress is live** (bytes, speed, ETA), **cancel** keeps partial
  files and **Resume** continues, and every weight file is **verified against the SHA-256 that
  Hugging Face publishes** before the download counts as done.
- **Gated models:** set a Hugging Face token on the same panel. It is stored in
  `ui/backend/auth/hf_token` and never shown again.

Models land in `models/` by default (`FREESWARM_MODELS_DIR` to move it). Anything already in your
Hugging Face cache is picked up too.

### GPU and RAM by model

Because experts stream from host RAM over PCIe under `--moe-backend offload`, **the binding
constraint is host RAM, not VRAM.** VRAM holds the active path and the KV cache, so a card far
smaller than the model can serve it — at a speed set by your PCIe link.

| Model | On disk | Host RAM | VRAM | Role |
| --- | ---: | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | 156 GiB | 128 GB+, NVMe-backed | 24 GB+ comfortable | frontier reasoning agent |
| Qwen3.6-35B-A3B | 67 GiB | 64 GB+ | 10–16 GB workable | fast general agent |
| gpt-oss-120b | 61 GiB | 96 GB+ | 16 GB+ | large general agent |
| gpt-oss-20b | ~13 GiB | 32 GB+ | 10 GB fits well | light agent, good second voice |
| Chronos-2 / Chronos-Bolt-base | < 1 GiB | any | 1–2 GB | forecaster (Forecast Lab) |
| Granite PatchTST | < 1 GiB | any | 1–2 GB | forecaster |

Verified configuration: Windows 10, 2 × RTX A6000 (sm_86, TCC) + RTX 3080 (WDDM), 128 GB RAM. Treat
the columns above as starting points for *this* offload design, not as hard minimums — the Models
page reports real VRAM cost per engine once a model is loaded, and the GPU pool panel shows headroom
live. Use `--expert-load serial` on Windows; the parallel reader maps full shard-sized regions and
doubles page-file commit.

---

# Running it

**Build once, then start.** Building and running are two commands, on purpose: the console is served
as a production build rather than the dev server, so nothing compiles on the first request and a
mistake surfaces as a build error instead of a blank page.

```bat
build-services.cmd      rem console + sandbox image  -- after a clone, a pull, or a UI change
start-services.cmd      rem launch everything
```

## `build-services.cmd` — what it builds

| | |
| --- | --- |
| `build-services.cmd` | the console and the sandbox image |
| `build-services.cmd --console` | the console only |
| `build-services.cmd --sandbox` | the sandbox image only |

1. **Checks `freetoken` imports** from the venv — a warning, not a failure. The console still builds
   without it, but no model would serve.
2. **Installs console dependencies** if `node_modules` is missing, or if `package.json` is newer than
   it, which means a dependency changed since the last install.
3. **Builds the console** with `next build` into `ui\frontend\.next`.
4. **Builds the sandbox image** if Docker is running, then proves a container can actually execute.
   Skipped with a clear message if Docker is absent or stopped — everything else works without it.

**Re-run it after** a fresh clone, a `git pull`, an edit under `ui\frontend`, or a change to
`ui\sandbox\requirements.txt`. You do **not** need it for Python changes: the control plane, message
board and swarm runner all run from source, so restarting them is enough.

Working on the UI itself? Skip the rebuild loop and run the dev server with hot reload:

```bat
ui\run-frontend.bat --dev
```

## `start-services.cmd` — what it actually does

```bat
start-services.cmd
```

One command, and it is mostly a preflight. In order:

1. **Finds the interpreter** at `.venv\Scripts\python.exe` and stops with a clear message if it is
   missing.
2. **Checks `freetoken` imports.** Without this the control plane starts happily and then *every*
   engine launch fails several seconds in with an unhelpful traceback. Fails fast instead.
3. **Checks Node** is on PATH.
4. **Checks the CUDA toolchain** (`nvidia\cu13\bin\nvcc.exe` in the venv). Only a *warning* — the
   control plane and board run fine without it, but no model will serve, because the engine
   JIT-compiles CUDA kernels on first use.
5. **Checks Docker** — also only a warning. If the daemon is down you get everything except the
   chat Run button.
6. **Checks ports** 8000, 8100, 3000, 1919, 1920 in one PowerShell call. A busy port usually means
   the services are already running, or an engine was killed without its process tree and a worker
   still holds 1919/1920. You are asked whether to start anyway (20 s timeout, defaults to no).
7. **Checks the sandbox image exists** — it is built by `build-services.cmd`, not here. Missing
   only disables the chat Run button.
8. **Checks the console has a production build** (`.next\BUILD_ID`) and its dependencies, and stops
   with a pointer to `build-services.cmd` if either is absent.
9. **Launches four windows** — control plane, message board, swarm runner, console — each in its own
   `cmd /k` so you can read its log and restart one without the others.
10. **Waits 4 s and opens the browser** — a production build serves immediately, so this is only
    the time `next start` needs to bind the port.

| Service | Address | What it is |
| --- | --- | --- |
| Console | http://localhost:3000 | the web UI |
| Control plane | http://127.0.0.1:8000/docs | engine lifecycle, telemetry, MCP, `/v1` |
| Message board | http://127.0.0.1:8100/docs | agent coordination |
| Swarm runner | — | claims queued tasks, one agent per model |
| Sandbox | — | per-run container, started on demand |

**The inference engine is not started here.** It is a child process of the control plane, launched
from the Models page (or `POST /api/engine/start`), so stopping the control plane always reaps the
engine and reclaims its VRAM.

Stop everything with `stop-services.cmd`. To run one piece by itself, the same scripts the launcher
calls work standalone: `ui\run-control-plane.bat`, `ui\run-msgboard.bat`, `ui\run-swarm.bat`,
`ui\run-frontend.bat` (add `--dev` for hot reload), `ui\run-sandbox.bat`.

> **A click can freeze a service.** These run in `cmd` windows, and clicking inside one enters
> Windows' QuickEdit selection mode, which blocks the process's next write to stdout. On a service
> that logs every request that means it stops serving within a second, and the title bar reads
> *(Not Responding)*. Press **Esc** in the window to release it.

Everything binds to loopback. **Read [`ui/README.md`](ui/README.md) before exposing any of it** — it
covers the security model, the full environment-variable reference, federation and the sandbox in
far more detail than this page.

## First run, end to end

1. **Models** → Download a model → **Load** it onto a GPU. It becomes an agent.
2. **Projects** → make a project → point it at a data folder, pick its allowed models, attach SQL
   tables and connectors.
3. **Swarm** → *swarm on* → **New objective**: the goal, the metric, the dataset, and the **holdout
   split date** the agents never see.
4. Watch the feed. Candidates appear, get scored, get audited, and climb the leaderboard.
5. *(Optional)* **Settings** → add an Anthropic API key to enable external review, then
   **review with Claude** on any candidate.

---

# Connecting one swarm to another

Run FreeSwarm on more than one computer and the models pool. A model loaded on another computer
appears here as **`model@computer`** (in violet) — in the Swarm's resources and agent list, the
project model picker and `/v1/models` — and requests for it are relayed there. Everything is driven
from the **Network** page.

```
   COMPUTER A  (has the model)                    COMPUTER B  (wants it)
   ┌──────────────────────────┐                   ┌──────────────────────────┐
   │ Network → Share models   │                   │ Network → Connect        │
   │   ticks which models     │                   │   discovers A, or        │
   │                          │   TLS 8443        │   connect by address     │
   │ ┌──────────────────────┐ │ ◄───────────────► │                          │
   │ │ federation gateway   │ │   pinned cert     │  shows code  KCDC-VSTP   │
   │ │  node info           │ │                   │                          │
   │ │  OAuth device grant  │ │   UDP 19191       │  ...then A approves it   │
   │ │  shared model list   │ │   discovery       │                          │
   │ │  chat completions    │ │   (optional)      │  models appear in        │
   │ └──────────────────────┘ │                   │  seconds as model@A      │
   └──────────────────────────┘                   └──────────────────────────┘
```

**On the computer that has the model:** Network → tick *Share models on this network*, then tick the
models to share. This starts a separate TLS listener on **TCP 8443** exposing only node info, the
OAuth endpoints, the shared model list, and chat completions for those models.

**On the computer that wants it:** Network → *Connect* on the discovered computer (or *Connect by
address*). A code appears, e.g. `KCDC-VSTP`. On the sharing computer's Network page the same code
shows under *Requests to use this computer* → **Approve**. The models appear within seconds.

Security, although it is "only the LAN":

- **TLS with certificate pinning.** Each computer has its own self-signed certificate. The first
  connection pins it — both pages show the fingerprint to compare — and a changed certificate is
  refused until you pair again.
- **OAuth 2.0 Device Authorization Grant (RFC 8628).** Nothing is granted until the operator
  approves on the sharing computer's own loopback-only console. Access tokens last 1 hour, refresh
  tokens 30 days and rotate on every use; only SHA-256 hashes are stored.
- **Least privilege.** Only ticked, loaded models; only `chat/completions` and `completions`; 4
  concurrent requests per client; private, link-local and Tailscale (100.64/10) sources only.
- **Revocation** from either side, effective immediately.
- **Version handshake.** Beacons carry versions, so an incompatible computer is listed and marked
  *Update needed* rather than silently missing; pairing is refused on both sides naming which
  computer to update; and every sync re-reads version headers, withdrawing models if a peer upgrades
  to an incompatible protocol.

**Windows Firewall.** One port matters: **TCP 8443**, on the computer that shares. **UDP 19191** is
optional, for automatic discovery on the computer that looks. Run `ui\federation-firewall.cmd` as
Administrator on each computer — it opens exactly those two, for the **Private** profile and the
**local subnet** only (`... remove` deletes them). A network Windows classifies as Public gets
nothing, by design. With **Tailscale**, pairing works between your computers on different networks;
discovery does not cross it, so connect by the `100.x` address.

---

## Configuration

Settings are named `FREESWARM_*` — `FREESWARM_UI_PORT`, `FREESWARM_MODELS_DIR`,
`FREESWARM_SWARM_LEASE_S`, `FREESWARM_FED_PORT`, and so on. The older `FREETOKEN_*` spellings for
platform settings are still read, so an existing machine keeps working.

Settings that configure **the engine itself** keep the FreeToken name, because that is whose
behaviour they change: `FREETOKEN_VISIBLE_DEVICES`, `FREETOKEN_ENGINE_PORT`,
`FREETOKEN_HOST_PIN_GB`, `FREETOKEN_HOST_PIN_FRACTION`, `FREETOKEN_CUDA_HOME`, `FREETOKEN_VCVARS`.

The full reference is in [`ui/README.md`](ui/README.md).

---

## Credits

FreeSwarm is built on **[FreeToken](https://github.com/FlashML-org/FreeToken)** by the FlashML team,
and exists because FreeToken made frontier-scale local inference fast enough for a swarm of agents to
be practical on one desk. Please follow that repository for engine releases and updates.

It runs on Windows thanks to **[UnsignedChad/freetoken-windows](https://github.com/UnsignedChad/freetoken-windows)**,
the native Windows port of that engine, which is where this repository's copy of it comes from.

Both are Apache-2.0; the full attribution and the statement of modifications are in
[`NOTICE`](NOTICE). Cite the engine's authors if you use it in research:

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

FreeToken in turn learned from [mini-sglang](https://github.com/sgl-project/mini-sglang),
[SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Disclaimer

FreeSwarm is provided **as is, without warranty of any kind**, and neither SignalPop LLC nor any
other contributor is liable for any damages arising from its use — see sections 7 and 8 of the
[LICENSE](LICENSE), restated in [NOTICE](NOTICE).

It is **research software, not investment advice.** Strategies, signals, scores and backtests it
produces are the output of an automated search over historical data. They are not recommendations,
not solicitations, and not a representation that any result is correct, reproducible or predictive.
Backtested performance is hypothetical. The look-ahead checks, audits and reviews reduce certain
classes of error; they do not establish that a result is valid. Trading involves risk of loss, and
any decision you make from this software's output is yours alone.

## License

[Apache License 2.0](LICENSE) — the standard, unmodified text. Attribution for FreeToken, its
Windows port, and the third-party code FreeSwarm vendors is in [NOTICE](NOTICE).
