# FreeSwarm Console — Windows control plane, web UI, and agent bus

FreeSwarm is the system: the control plane, the console, the agent bus and everything
built on them. **FreeToken** is one part of it — the native-Windows inference engine that
loads a checkpoint onto the GPUs and serves tokens from it. Three FreeSwarm services sit
on top of that engine:

| Service | Port | What it does |
| --- | --- | --- |
| **Control plane** (FastAPI) | 8000 | Starts/stops the engine, discovers models, GPU telemetry, log capture, MCP connectors, OpenAI-compatible passthrough |
| **Message board** (FastAPI) | 8100 | Agent coordination: message log, task queue with atomic claim, shared blackboard |
| **Console** (Next.js 16) | 3000 | The web UI — Console, Models, Chat, Swarm, Connectors, Logs, Settings |

The FreeToken engine itself runs as a **child process of the control plane** on port 1919.

```
browser ──► Next.js :3000 ──► control plane :8000 ──► engine :1919 (child process)
                         └──► message board :8100
```

The browser only ever talks to `:3000`; Next rewrites `/api/*` and `/mb/*` to the two
backends. Nothing but the Next dev server needs to be reachable.

---

## Quick start

```bat
ui\run-all.bat
```

Opens three windows and serves the console at <http://localhost:3000>.

To run them individually: `run-control-plane.bat`, `run-msgboard.bat`, `run-frontend.bat`.

### First-time setup

Requires **Node 20.9+** (Next 16). The Python venv must already exist at the repo root with
the FreeToken engine installed into it —
see the **Engine setup** section at the bottom. Frontend dependencies install themselves
on first `run-frontend.bat`.

---

## Security — read this before exposing anything

**The FreeToken engine has no authentication of any kind.** There is no API-key option in
`server/args.py`. This is fine because the engine always binds `127.0.0.1` and is only
reachable through the control plane.

The control plane is therefore the only front door, and it is the thing that must be
protected. Two interlocks enforce that; both live in `Settings.validate()`:

1. **Binding off-loopback with no accounts → refused.** `/api/engine/start` launches
   processes. An unauthenticated network endpoint that starts processes is a remote
   shell with extra steps.
2. **Binding off-loopback over plain HTTP → refused.** OAuth bearer tokens are replayable
   credentials; in clear text on a shared network they are a free login for anyone
   sniffing. Override only with `FREESWARM_UI_ALLOW_INSECURE=1`, and only on a segment
   you actually trust.

### Reaching the box from another machine

**Preferred — SSH tunnel.** Nothing listens on the LAN at all:

```bash
ssh -L 8000:127.0.0.1:8000 -L 8100:127.0.0.1:8100 you@the-box
```

Point the frontend at `localhost` as usual. Traffic is encrypted and authenticated by SSH.

**Alternative — LAN bind with TLS + accounts:**

```bat
python -m app.usercli add alice
set FREESWARM_UI_HOST=0.0.0.0
set FREESWARM_UI_SSL_CERT=C:\certs\ui.crt
set FREESWARM_UI_SSL_KEY=C:\certs\ui.key
run-control-plane.bat
```

### Accounts and tokens

```bat
cd ui\backend
..\..\.venv\Scripts\python -m app.usercli add alice        rem create (prompts for password)
..\..\.venv\Scripts\python -m app.usercli list
..\..\.venv\Scripts\python -m app.usercli token alice      rem long-lived token for an agent
..\..\.venv\Scripts\python -m app.usercli rotate-key       rem invalidate every token
```

Authentication is **presence-based**: with zero accounts it is off (loopback-only
convenience); creating the first account turns it on for `/api/*`, `/v1/*` and `/mb/*`.

Passwords are scrypt hashes (N=2¹⁵) with per-user salts. Access tokens are 30-minute
HS256 JWTs; refresh tokens last 14 days and carry `typ: refresh` so they cannot be
replayed as access tokens.

> **Scope caveat.** A token is currently all-or-nothing: it opens `/v1/*` *and*
> `/api/engine/start`. If you hand tokens to autonomous agents, be aware they can start
> and stop engines. Per-scope tokens are the obvious next step.

---

## Using it from agent code

The control plane speaks OpenAI at `/v1`, so any framework works unmodified:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<token or anything>")
resp = client.chat.completions.create(
    model="gpt-oss-120b",                       # or omit; the loaded model is used
    messages=[{"role": "user", "content": "..."}],
)
```

`api_key` is ignored while no accounts exist; once one does, pass an access token.

**Concurrency.** The engine batches continuously, but `max_running_req` defaults to **4**.
For a swarm raise `--max-running-requests` (the Models page exposes it) *and* size
`--num-pages` to cover *N agents × their typical context* — the engine refuses configs
where KV pages cannot cover the slot count.

---

## MCP connectors

Connectors let a model *act*. Tools are converted to OpenAI function-tool schemas, so
they drop straight into a chat completion's `tools` array.

Servers are declared in **`ui/backend/mcp_servers.json`**. Both config shapes are
accepted — the native form and the Claude-Desktop `mcpServers` form you can paste from
another tool's README:

```json
{
  "mcpServers": {
    "backtest": {
      "command": "python",
      "args": ["DNN.net.app.BacktestRunner/mcp/backtest_mcp.py", "--transport", "stdio"],
      "cwd": "C:/repos/DNN.fin"
    }
  }
}
```

> **Registration is filesystem-only, deliberately.** A stdio MCP server is an arbitrary
> command line. Letting an HTTP caller define one would make the control plane a
> remote-code-execution service. The API can list, probe and invoke — never define.

Each server runs as its **own process**, so it can use its own interpreter and its own
dependency set. A server written against `FastMCP` (mcp v1) just needs a Python with
`mcp<2` installed — it does not have to match this venv, which is on mcp 2.x where
`FastMCP` was renamed `MCPServer`.

### Remote servers with OAuth

An `http` / `sse` server usually sits behind OAuth 2.1. Set `"oauth": true` and press
**Connect** on the Connectors page: the control plane discovers the authorization server,
opens the provider's consent page in a new tab, and completes the token exchange when the
browser redirects back to its loopback callback.

```json
{
  "servers": [
    {
      "name": "github",
      "transport": "http",
      "url": "https://api.githubcopilot.com/mcp/",
      "oauth": true,
      "client_id": "Iv1.xxxxxxxxxxxx",
      "client_secret_env": "GITHUB_MCP_SECRET",
      "scopes": ["repo", "read:user"],
      "enabled": true
    }
  ]
}
```

**`client_id` is usually required.** The MCP spec prefers Dynamic Client Registration
(RFC 7591) and the SDK tries it first, but most real providers do not implement it —
GitHub answers the registration endpoint with a 404 and expects an OAuth App you
registered by hand. Supplying `client_id` skips registration entirely. Omit it only for a
server you know supports DCR.

Register the provider's redirect URI as exactly:

```
http://127.0.0.1:8000/api/mcp/oauth/callback
```

(RFC 8252's loopback redirect for native apps. It must match the control plane's port.)

Prefer **`client_secret_env`** over `client_secret`: `mcp_servers.json` is ordinary config
with no special permissions, whereas an env var is not left readable on disk.

**Where the tokens go.** `ui/backend/mcp_tokens.json`, written with owner+SYSTEM-only ACLs
(inheritance stripped), same treatment as the JWT signing key. They are never logged and
never returned by any endpoint — the API reports only *whether* a server is authorised,
plus scope and expiry. Access tokens refresh automatically; **Disconnect** deletes them.

The `/api/mcp/oauth/callback` route is deliberately **not** behind the bearer-token guard,
because the provider redirects a plain browser to it and that redirect cannot carry a
token. It is protected instead by the OAuth `state` parameter — 24 bytes of randomness
minted per flow, matched on return, expiring after 5 minutes — which is exactly the CSRF
defence the spec prescribes.

Endpoints: `GET /api/mcp/servers`, `GET /api/mcp/probe`, `GET /api/mcp/tools`,
`POST /api/mcp/call` (`{"tool": "<server>__<tool>", "arguments": {...}}`),
`POST /api/mcp/oauth/start`, `GET /api/mcp/oauth/status`, `POST /api/mcp/oauth/disconnect`.

`ui/backend/connectors/freetoken_tools.py` is a working template — sandboxed filesystem
reads plus GPU/host status. Copy it to write your own.

---

## Projects

A project is the unit of isolation. It owns three things:

| | |
| --- | --- |
| **A message board** | Sessions, messages, tasks, agents and blackboard entries are all scoped to it. Two projects never see each other's coordination traffic. |
| **A connector set** | MCP servers are declared once globally; each project picks which of them it may use. |
| **A data directory** | Documents, inputs, artefacts — everything that is not coordination traffic. |

Manage them on the **Projects** page; switch with the picker at the top of the sidebar.

### The active project

Switching projects changes the **server's** active project, not just the browser's view.
That matters: an agent that calls the board without naming a project resolves to the same
one the console is showing, so the UI and the swarm never disagree about which work is
current. An agent that wants to pin itself to one project passes `project_id` explicitly —
every board endpoint accepts it, and the `/api/mcp/*` routes take it as a query parameter.

On an upgraded install the first project created **adopts the pre-projects history**, so
existing sessions and tasks stay visible instead of being orphaned by the new scoping.

### Connectors are an intersection

A connector must be enabled in **both** places to be reachable:

* globally, in `mcp_servers.json` — "this machine may run it at all"
* per project, on the Projects page — "this work may use it"

This is enforced on `POST /api/mcp/call`, not merely hidden in the UI: a model in a project
that has not enabled a connector gets `no enabled MCP server named '...'`. A project that
selects a connector which is globally off is shown as inactive rather than failing
silently.

New projects start with **no** connectors, so granting tool access is always deliberate.

### Data directory

Defaults to `projects/<slug>/data` under the repo, but may point anywhere.

Every file operation resolves through a containment check that runs *after* `resolve()`,
so `..` segments and symlinks are collapsed before the boundary is tested. Both a
traversal path and a hostile upload filename are rejected or neutralised rather than
escaping the directory.

**Deleting a project only removes files inside the managed `projects/` root.** Point a
project at your Documents folder and the delete button will not touch it — verified,
because "a web button wiped my documents" is not a recoverable mistake.

```
GET  /api/projects                       list + which is active
POST /api/projects                       {name, data_dir?, connectors?}
POST /api/projects/{id}                  update name / data_dir / connectors
POST /api/projects/{id}/activate
POST /api/projects/{id}/delete?remove_files=
GET  /api/projects/{id}/files?subdir=
GET  /api/projects/{id}/file?path=
POST /api/projects/{id}/upload           multipart: file, path
POST /api/projects/{id}/mkdir            {path}
POST /api/projects/{id}/files/delete     {path}
```

## Agent message board

| Concept | Why it exists |
| --- | --- |
| **Messages** | Append-only per-channel log with monotonic `seq`. Supports long-polling (`wait=25`), so an idle swarm costs one parked connection per agent, not a request per second. |
| **Tasks** | Work queue with **atomic claim** — a conditional UPDATE, so two agents racing the same task cannot both start it. Claims carry a **lease**; a crashed agent's task returns to the queue instead of being lost. |
| **Blackboard** | Shared key/value facts with compare-and-set, so concurrent writers do not silently clobber each other. |
| **Sessions** | Every message and task is stamped with a session, so a run can be reviewed afterwards instead of being one undifferentiated log. One session is open at a time; the first message of a run opens one automatically. |

### Storage, and why SQLite

SQLite in WAL mode at `ui/backend/msgboard.sqlite3`. Measured on this machine:

| operation | throughput | latency |
| --- | --- | --- |
| insert (durable, commit per message) | 3,840 msg/s | 0.26 ms |
| cursor read (200 rows) | 3,647 reads/s | 0.27 ms |
| atomic task claim | 2,022 claims/s | 0.50 ms |

**DuckDB would be slower here.** This is an OLTP workload — tiny point inserts, single-row
cursor reads, and row-level conditional `UPDATE`s. DuckDB is a columnar OLAP engine built
for scanning millions of rows for aggregates, and it pays for that with expensive
single-row updates and a single-writer concurrency model that would not give the
exactly-one-winner task claim its guarantee. WAL mode's concurrent-readers-plus-one-writer
is precisely the swarm's access pattern.

Where DuckDB *does* earn a place is later analytics over accumulated history. That needs
no migration: DuckDB's `sqlite_scanner` attaches this file and queries it in place, so the
write path stays SQLite and analysis gets a columnar engine.

**Backups:** in WAL mode most recent writes live in `msgboard.sqlite3-wal`, not the
`.sqlite3` file, so copying the latter alone silently loses them. `POST /mb/checkpoint`
folds the WAL back in and makes the single file self-contained.

```
POST /mb/agents/register      {name, role, model, capabilities}
POST /mb/agents/{id}/heartbeat
POST /mb/tasks                {title, description, tier}
POST /mb/tasks/claim          {agent_id, tier?}   -> exactly one winner
POST /mb/tasks/{id}/complete  {agent_id, status, result}
POST /mb/tasks/{id}/extend    {agent_id, lease_s}
GET  /mb/messages?since=&channel=&wait=
PUT  /mb/state/{key}          {value, expect_version?}

POST /mb/sessions             {title, note}      -> closes the open one, starts a new one
POST /mb/sessions/{id}/close
GET  /mb/sessions                                -> all sessions + message/task counts
GET  /mb/sessions/{id}                           -> full transcript of one session
GET  /mb/messages?session_id=<id>                -> filter the log to one session
POST /mb/checkpoint                              -> fold the WAL into the .sqlite3 file
```

## The Python sandbox

Chat writes Python; the **run** button on any `python` code block executes it and shows what
it produced — stdout, errors, charts inline, and spreadsheets/documents/CSVs/PDFs as
downloads. **send to chat** drops that output into the composer so the model can see what
its own code actually did and fix it.

Docker is **optional**: without it everything else works and the run button is disabled with
a tooltip saying exactly why (daemon stopped, image not built, ...).

### One container per run, `--network none`

Nobody reviews this code before it runs. A model writes it, a button runs it, and it could
be anything — so the container is the trust boundary, and the isolation has to be real.
Three arrangements were measured on this stack (Docker Desktop, WSL2 backend):

| arrangement | result |
|---|---|
| long-lived service on an `--internal` network | **unusable** — `--internal` disables published ports, so the control plane cannot reach it |
| long-lived service on a bridge with `enable_ip_masquerade=false` | **not isolated** — the port works, but the WSL2 VM's own NAT still routes out: `1.1.1.1:53` and `8.8.8.8:443` were both reachable from inside |
| **one container per run with `--network none`** | **isolated** — every outbound connect fails, and no listening socket exists at all |

So there is no long-lived sandbox container. `ui/backend/app/sandbox.py` starts one per run:

```
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges   --memory 4g --memory-swap 4g --cpus 2 --pids-limit 256   -v <run dir>:/work -w /work freeswarm-sandbox:latest python -I script.py
```

Container start costs about a second (a chart round-trips in ~1.4s), against a 60s default
budget. A timeout kills the **container**, not just the docker client — otherwise the
container would keep running with the run directory still mounted.

Verified: no egress on any of three addresses, no host filesystem visible (`/mnt/c`,
`/host`, `C:\` all absent; the working directory contains only the script), OOM and timeout
both reported rather than surfacing as a bare exit code, and artifact paths cannot traverse
out of their run directory.

### How a run works

The run directory is bind-mounted at `/work` and is the working directory, so the forms a
model actually writes just work, with no output-path convention to get wrong:

```python
plt.savefig("chart.png")      # shown inline in the chat
df.to_csv("data.csv")         # offered as a download
df.to_excel("book.xlsx")      # likewise
Document().save("report.docx")
```

Anything left behind is an artifact. Dot-files are skipped (tool caches, not output).
Artifacts live on the host and are served back through the control plane
(`/api/sandbox/artifacts/<run>/<name>`), with media types pinned in `sandbox.py` rather than
taken from the platform registry — Windows reports `.csv` as `application/vnd.ms-excel`,
which would send CSVs down the wrong rendering path.

The image carries numpy, pandas, scipy, matplotlib, pillow, openpyxl/XlsxWriter/odfpy,
python-docx, python-pptx, reportlab/fpdf2 and pyarrow. The list is deliberately broad: a
missing import turns a working answer into a traceback the user has to diagnose, and the
image is built once. The matplotlib font cache is baked into the image at a fixed
`MPLCONFIGDIR`, because every run is a fresh container and would otherwise rebuild it.

`start-services.cmd` builds the image on first run and verifies a container can execute.
After editing `ui/sandbox/requirements.txt`, rebuild with `ui
un-sandbox.bat --rebuild`.

### Who runs the tasks: the swarm runner

The queue is **pull-based** — `POST /mb/tasks` only records a task, and it stays `open`
until something calls `/mb/tasks/claim`. Loading a model gives you an inference endpoint,
not an agent, so without a consumer the Swarm page sits at "1 open · 0 active" forever with
"None registered" under Agents.

`ui/backend/swarm_runner.py` is that consumer. `start-services.cmd` launches it in its own
window (`ui
un-swarm.bat`); `stop-services.cmd` reaps it by command line, since it binds no
port.

- **One agent per resident model**, re-synced every 15s, so agents follow what the console
  loads and unloads. Re-registering a name keeps the same board id and history.
- **Tiering.** A task is `auto|small|mid|hard`; each model's tier comes from the parameter
  count in its name (`<25B` small, `<70B` mid, else hard) and an agent claims its own tier
  plus `auto`. `auto` is the default, so a single-model setup is never starved.
- **Leases are held.** A background thread extends the lease every `LEASE_S/3` for as long
  as generation runs, so a slow offloaded model's task is not reclaimed and answered twice.
- **Reasoning models.** Empty content with `finish_reason: length` means the whole budget
  went on the chain of thought; that is reported as a failure with the token count, not
  completed with an empty result.

Configuration (all optional): `FREESWARM_SWARM_MAX_TOKENS` (default 8192),
`FREESWARM_SWARM_LEASE_S`, `FREESWARM_SWARM_POLL_S`, `FREESWARM_SWARM_TIMEOUT_S`,
`FREESWARM_PROJECT_ID`. When accounts are enabled, authenticate it with
`FREESWARM_API_TOKEN`, or `FREESWARM_AGENT_USER` + `FREESWARM_AGENT_PASSWORD` (it then
refreshes its own 30-minute token).

The **Swarm** page has a session picker: "Live — current session" follows the running one,
and selecting a past session shows its transcript read-only (no long-poll, composer
disabled, since a finished session can never receive new messages).

The **Swarm** page drives all of this: post a directive, watch agents claim and report.

### Standing objectives: the swarm that keeps improving

A task is answered once. An **objective** ("create the best trading strategy possible") is
worked on continuously: whenever no one-off task is queued, each agent runs an *iteration*
and submits one **candidate**; a trusted harness scores it; the best-scoring candidate that
survives the checks is the **champion** until something beats it (announced in `#results`).
With an objective selected, messages typed on the Swarm page **steer** it (every agent reads
them at the start of its next iteration) instead of queuing unrelated tasks.

**"Better" is a metric computed by the harness, never by the agent** (`app/objectives.py`):
Sharpe, Sortino, Calmar, total return, CAGR, max drawdown, a script-reported score, or a
judge model's 0-10 against a rubric. For trading metrics a candidate is a Python script that
reports **positions** per bar (`ft.report_positions`); the harness holds each position to the
next bar, marks it to market on the dataset's price column, charges `cost_bps` per unit of
position change, compounds per day and computes the metric. The script never reports a
return it did not earn.

Three guards stand between a good number and the title:

- **Hidden holdout.** Data is split at a date (default: last 30% of days). Ranking uses
  only the holdout; agents see in-sample numbers only, and in objective mode their data
  tools (`query_data`, `run_python`) see only rows before the split. `query_sql` is withdrawn
  (the table holds every row; use the parquet export).
- **Look-ahead test.** Each candidate is re-run with every row after a cut removed (a
  truncated copy bind-mounted over the read-only data), at the split and at an intraday bar
  mid in-sample. If any position before a cut changes, the candidate is rejected. It compares
  *positions*, not returns: a strategy that trades on the next bar's move and books that
  bar's return is self-consistent in its returns (measured: a one-bar peek scoring Sharpe 40
  passed a returns-only check) but its last decision before the cut changes.
- **Audit.** A would-be champion's code is reviewed by a *different* loaded model (hard-coded
  dates, reading around `ft.load`, degenerate logic) before it takes the title.

**How the agents get better over time:**

- *Evolutionary search.* The control plane assigns each iteration: IMPROVE a parent chosen
  by a tournament over the top ranks (70%) or EXPLORE a genuinely different approach (30%).
- *Shared memory.* After every attempt the agent writes one lesson (`KEEP:` / `AVOID:` /
  `TRY:`); every later iteration reads the lessons, and once there are 40 an agent
  consolidates them into at most 15.
- *Context of the search.* Each brief carries the leaderboard, recent attempts and their
  failure causes (so dead ends are not repeated), and the operator's steering notes.

Candidates run in the same sandbox as chat code (`--network none`, 4 GiB, 2 CPUs), with the
project data folder mounted read-only at `/data`, two evaluations at a time, and a per-
candidate time limit (default 300 s). On the 713K-row GEX dataset a full evaluation (run +
two look-ahead re-runs + scoring) takes about 12 s. State lives in
`ui/backend/objectives.sqlite3`; truncated copies in `ui/sandbox/.objectives/<id>/`.

**Forecast models inside strategies.** Candidates have no network, so a strategy cannot call a
time-series model. `forecast_feature` runs the loaded forecaster over a column, or an arithmetic
expression over columns (`GEX / Pinning_TotalAbsGex`; validated by sqlglot, only columns,
numbers, `+ - * /` and pure math functions allowed), causally: the forecast stored at bar t
uses bars up to and including t. It is saved as a dataset (`ft.load("fc_<name>")`), truncated
at every cut like the data, and stored with its in-sample skill against a no-change forecast
and its direction accuracy. Measured on the GEX data (Chronos-Bolt-base, 30 bars ahead): no
field beats "no change" on error, but volatility-like series call the direction well
(IntrVol 0.66, SkewRR 0.64, HistVol 0.59, Pinning 0.57) while price and GEX are coin flips.

**Chronos-2** (Amazon, Apache-2.0) is the multi-input forecaster: it forecasts one or several
targets while reading any other columns as *inputs* (covariates). `forecast_feature(column=
"Close", covariates=[...], calendar=true)` builds such a feature; only past values of data columns
are given (future ones would leak), and the only inputs given for the horizon are calendar
features (time of day, weekday), which are known in advance. Every covariate feature records its
in-sample lift over the same forecast without the inputs, with a +/- 2 SE range.

**Forecast Lab** (sidebar; `app/tslab.py`) finds out which inputs actually help. *Test this
combination* scores a target plus chosen inputs against the target alone and plots example
forecasts. *Run full analysis* reverses the model: target alone, all inputs, each input removed
(**impact**), each input alone (**solo lift**), and the best combination built greedily --
every number a paired comparison at the same in-sample points with a +/- 2 SE error bar, so a
lucky result is shown as "within noise" rather than as a finding (the first real run: a +0.03
"best" at 150 points vanished at 600). The best combination can be built into a forecast
feature in one click, and the latest analysis appears in every agent's brief.

**Kronos** (NeoQuasar, MIT) is served like Chronos -- Models page, load on a GPU, agents use it
through `forecast` / `forecast_feature` -- but it is a *candle* model: it reads whole OHLCV bars
(with their timestamps, up to 512) and generates future candles. `forecast_feature` detects it and
builds candles itself at the requested `bar` size (1min default), each stamped when complete.
It is slow -- it generates bar by bar (about 0.4 s per forecast batched at 10 bars ahead, 4 s at
30), so a Kronos feature is capped at 3,000 forecasts and adds a forecast range (`fc_high_q90`,
`fc_low_q10`). Quantiles come from independently sampled paths. Model code is vendored in
`ui/backend/tsfm_vendor/kronos` at a pinned, reviewed commit (see its VENDOR.md); weights and
the tokenizer (`Kronos-Tokenizer-base`, needed alongside) come from the Download models panel.

**Code library** (`app/library.py`, per project). Agents save reusable modules (`regime`
detectors with `detect(df)`, `signal`s with `signal(df)`, `risk` rules, `util`s); each save is
smoke-tested in the sandbox and versioned, and candidates import them with `from lib import x`.
Every candidate that imports a module is recorded against it (version, score, look-ahead
result), and agents and the operator leave `works` / `broken` / `note` comments. `regime_map`
measures every signal module inside every regime of a detector, in-sample and net of the
objective's costs, and `ft.route(regime, {label: positions})` routes regimes to the functions
that work there. A starter `regime_gex_vol` (GEX sign x volatility) is seeded for the Gex project.
Review it all on the objective panel's **Code library** tab.

**Timeframes.** `ft.resample(df, "5min")` builds OHLCV bars stamped at their last underlying
bar (the moment they are complete) and `ft.align` carries decisions back to the 10-second
grid. The look-ahead test compares the positions of the full run; positions only the
truncated run has (a partial bar at the cut) are ignored.

**Playbook -- agents that improve their own instructions** (`app/playbook.py`). Every
iteration's system prompt starts with the project's playbook:

- the **charter** -- the operator's standing instructions (a default ships: coordinate through
  the board, review the library's `works`/`broken` verdicts and the teammates' results before
  building, post your plan first, build tested reusable modules, leave evidence), edited on
  the objective panel's **Playbook** tab;
- the **team practices** -- rewritten BY AN AGENT every 10 candidates from the evidence (recent
  candidates and how they were made, lessons, library verdicts, errors). They are about how
  the team works, so the agents' own process improves with every rewrite. Every version of
  both parts is kept and can be viewed or restored.

Coordination is mechanical as well as instructed: `team_board` reads what teammates just did,
`team_post` announces a plan in #planning, and every brief lists teammates' plans from the
last 45 minutes. **BUILD** iterations (about half while the library has fewer than 6 modules,
more for strong coding models) must save a tested library module and prove it in a candidate;
`run_python` is capped at 6 experiments per iteration so exploration ends in something the
team can reuse. `field_scan` ranks all 151 GEX columns by in-sample rank correlation with the
forward return (level and change, optionally per regime); every brief carries a field guide
of all columns by family and the latest scan's top fields.

**Agents talk to each other.** `team_post` can address one teammate (`to`) and answer a
message (`reply_to`); messages addressed to an agent (or @mentioning it) since its last
iteration appear in its next brief under MESSAGES TO YOU, with an instruction to answer. After
every iteration the runner posts a **collaboration record** to #team -- whose candidate it built
on, whose library modules it reused, what it contributed, who it messaged or answered, what it
left unanswered -- plus the agent's own `TEAM:` line. The Swarm page's **Team collaboration**
panel totals these per agent ("who builds on whom"), and the team-practices rewrite reads them,
so the agents' way of collaborating is revised from evidence like everything else.

What this does not protect against: selection bias from trying many candidates against the
same holdout. The holdout is never shown to agents, but the ranking itself is feedback, so a
long-running objective slowly fits it; the candidate count is shown next to the champion for
that reason. Before trading a champion, confirm it on data collected after the objective started.

---

## Downloading models

Models page -> **Download models**. The models this install runs (Qwen3.6-35B-A3B, gpt-oss-20b,
gpt-oss-120b, DeepSeek-V4-Flash-0731, Chronos-Bolt-base, PatchTST) are listed with their state
and pinned to the exact Hugging Face revisions in use -- a second computer gets byte-identical
weights, which federation relies on. Any other repo can be checked and downloaded too.
(`app/downloads.py`, worker `model_fetch.py`.)

- **Only what the engine loads.** gpt-oss repos carry the weights three times (safetensors,
  `original/`, `metal/`); only the safetensors are fetched -- 61 GiB instead of 182 GiB.
- **No silent code.** Pickle/HDF5 weights (`.bin .pt .pth .pkl .ckpt .h5`) are never fetched.
  `.py` files come only for a known model at its pinned revision (DeepSeek's `encoding_dsv4.py`,
  which the engine executes -- listed before download) or for a custom repo when you tick
  *Include the repo's Python code*, with every file named first.
- **Disk space is checked first; progress is live** (bytes, speed, ETA, from the worker);
  **cancel** keeps partial files and **Resume** continues; every weight file is **verified**
  against the SHA-256 Hugging Face publishes before the download counts as done.
- **Gated models:** set a Hugging Face token on the same panel -- stored in
  `ui/backend/auth/hf_token`, passed to the worker in its environment, never shown again.

## Several computers: LAN federation

Run FreeSwarm on more than one computer and use each other's models: a model loaded on another
computer appears here as `model@computer` (in violet) -- in the Swarm's resources and agents,
the project model picker and `/v1/models` -- and requests for it are relayed there. The
**Network** page (sidebar) does all of it; the design is in `app/federation.py`.

**On the computer that has the model:** Network -> tick *Share models on this network* and
tick the models to share. This starts a separate TLS listener (TCP 8443) that exposes only:
node info, the OAuth endpoints, the shared model list and chat completions for those models.

**On the computer that wants it:** Network -> *Connect* on the discovered computer (or
*Connect by address*). A code appears (e.g. `KCDC-VSTP`). On the other computer's Network
page the same code shows under *Requests to use this computer* -> **Approve**. The models
appear within seconds.

Security, although it is "only the LAN":

- **TLS with certificate pinning.** Each computer has its own self-signed certificate. The
  first connection pins the exact certificate (both pages show its fingerprint to compare);
  a changed certificate is refused until you pair again.
- **OAuth 2.0 Device Authorization Grant (RFC 8628).** Nothing is granted until the operator
  approves on the sharing computer's own console (which is loopback-only). Access tokens last
  1 hour, refresh tokens 30 days and rotate on every use; only SHA-256 hashes are stored.
- **Least privilege.** Only ticked, loaded models; only `chat/completions` and `completions`;
  4 concurrent requests per client; private, link-local and Tailscale (100.64/10) source
  addresses only.
- **Revocation** from either side, effective immediately.

**Versions.** `app/version.py` is the single source: `APP_VERSION` (the FreeSwarm release,
shown in the sidebar and on the Network page) and `FEDERATION_PROTOCOL` / `MIN_FEDERATION_PROTOCOL` (the wire
contract between computers). Two computers are compatible when each one's protocol is at least
the other's minimum; different releases on a compatible protocol work, with a note suggesting an
update. The check runs at every step: beacons carry versions (an incompatible computer is still
listed, marked *Update needed*, instead of silently missing), pairing is refused on both sides
with a message saying which computer to update, and every sync re-reads the other computer's
version headers -- one upgraded to an incompatible protocol after pairing has its models withdrawn
until both match. Bump the protocol only for a change that would break an older computer.

**Windows Firewall.** One port matters: **TCP 8443**, on the computer that shares. **UDP
19191** is optional, for automatic discovery on the computer that looks. Run
`uiederation-firewall.cmd` as Administrator on each computer: it opens exactly these two, for
the **Private** network profile and the **local subnet** only (`... remove` deletes them). A
network Windows classifies as Public gets nothing, by design; the Network page shows each
network's category. With **Tailscale**, pairing works between your computers even on
different networks (discovery does not cross it -- connect by the 100.x address).

## FreeToken engine setup (what this stack sits on)

Verified on this machine: Windows 10, 2 × RTX A6000 (sm_86, TCC) + RTX 3080 (WDDM),
128 GB RAM, MSVC 19.51 (VS 18), Python 3.13.

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip wheel setuptools ninja
.venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

rem CUDA 13.4 toolchain as pip wheels -- no admin, no 3 GB installer.
rem These unpack to site-packages/nvidia/cu13 in exactly the layout a Windows CUDA
rem toolkit uses, and the control plane finds them automatically.
.venv\Scripts\python -m pip install nvidia-cuda-nvcc nvidia-nvvm nvidia-cuda-crt nvidia-cuda-runtime

.venv\Scripts\python -m pip install apache-tvm-ffi==0.1.13.post3 flashlib triton-windows tornado ^
  einops fastapi gguf huggingface_hub msgpack modelscope numpy openai partial-json-parser ^
  prompt_toolkit pydantic pyzmq safetensors tqdm transformers uvicorn
.venv\Scripts\python -m pip install -r ui\backend\requirements.txt

rem Build the two C++ extensions. DISTUTILS_USE_SDK is required inside a vcvars shell.
call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set CUDA_PATH=%CD%\.venv\Lib\site-packages\nvidia\cu13
.venv\Scripts\python -m pip install -e . --no-build-isolation --no-deps
.venv\Scripts\python scripts\patch_deps_windows.py
```

### Things that will bite you

**The nvcc/torch CUDA major must match.** `kernel/_toolchain.py` refuses to compile
otherwise, and it fails *after* the model loads — minutes in. If this box's system
`CUDA_PATH` (12.3) wins over the venv's 13.4 you get
`nvcc 12.3 would build kernels linking libcudart.so.12`. The control plane prefers the
venv toolchain for exactly this reason.

**Tensor parallelism does not work on native Windows.** `--tp-size 2` needs NCCL:
`pynccl.py` links it with `-lnccl` (a GNU flag MSVC cannot consume) and the fallback path
asks for `backend="nccl"`. NCCL has no Windows build. **Use one GPU plus
`--moe-backend offload`**, which streams experts from host RAM over PCIe — that is the
configuration this engine is designed around, and 128 GB of RAM suits it well.

**`--expert-load serial` on Windows.** The parallel reader maps full shard-sized regions
and doubles page-file commit.

**Orphaned workers hold ports and VRAM.** The engine spawns scheduler and tokenizer
workers as separate processes under the *base* interpreter, so killing only the parent
leaves them holding port 1920 (the torch.distributed rendezvous) and their VRAM. The
supervisor uses `taskkill /T` and preflights both ports; if you kill an engine by hand,
use `taskkill /PID <pid> /T /F`.

**GPU ordering.** CUDA defaults to `FASTEST_FIRST`, which does *not* match `nvidia-smi`.
The control plane pins `CUDA_DEVICE_ORDER=PCI_BUS_ID` so a device list copied from
`nvidia-smi` selects the GPUs you meant. Set `FREETOKEN_VISIBLE_DEVICES` to choose.

### Hardware constraints on sm_86 (Ampere)

| Format | Runs on A6000? |
| --- | --- |
| BF16 | yes |
| MXFP4 (gpt-oss) | yes, via Triton dequant |
| **FP8** (`e4m3`) | **no** — needs sm_89 (Ada) or newer |
| **NVFP4** | **no** — needs sm_100, and the sm_80–99 Marlin fallback is incompatible with this dependency set |

`DeepSeek-V4-Flash-0731` is FP8 and 155 GiB, so it needs an Ada-or-newer GPU and enough
host RAM to pin its experts (~192 GB in the reference config).

---

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `FREESWARM_UI_HOST` | `127.0.0.1` | Control-plane bind. Non-loopback requires accounts + TLS. |
| `FREESWARM_UI_PORT` | `8000` | |
| `FREETOKEN_ENGINE_PORT` | `1919` | Engine also uses `port+1` for rendezvous. |
| `FREESWARM_MODELS_DIR` | `<repo>/models` | Extra model root. The HF hub cache is always scanned. |
| `FREETOKEN_VISIBLE_DEVICES` | `1,2` | `CUDA_VISIBLE_DEVICES` for the engine, in PCI-bus order. |
| `FREESWARM_UI_PYTHON` | `<repo>/.venv/Scripts/python.exe` | Interpreter used to launch the engine. |
| `FREETOKEN_CUDA_HOME` | auto | Overrides CUDA toolkit discovery. |
| `FREETOKEN_VCVARS` | auto (vswhere) | Overrides `vcvars64.bat` discovery. |
| `FREESWARM_UI_SSL_CERT` / `_KEY` | — | Enables TLS. |
| `FREESWARM_UI_ALLOW_INSECURE` | unset | Permits a plaintext non-loopback bind. |
| `FREESWARM_MCP_ROOT` | repo root | Sandbox for the example connector's file tools. |
