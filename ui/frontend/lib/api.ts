// Typed client for the FastAPI control plane. Every path is relative: next.config.mjs
// rewrites /api/* to the control plane, so the browser has exactly one origin.

export type GpuInfo = {
  index: number
  name: string
  memory_total_bytes: number
  memory_used_bytes: number
  utilization_pct: number | null
  temperature_c: number | null
  power_draw_w: number | null
  power_limit_w: number | null
  compute_cap: string
}

export type ModelEntry = {
  id: string
  path: string
  root: string
  size_bytes: number
  /** Bytes an OFFLOADED engine page-locks in host RAM: the experts only, not the whole
   *  checkpoint. 0 when unknown. The pinning-limit check budgets against this. */
  expert_bytes: number
  /** KV-cache bytes per token of context (full-attention layers only); null if unknown. */
  kv_bytes_per_token: number | null
  architecture: string | null
  num_experts: number | null
  quantization: string | null
  max_position_embeddings: number | null
  hidden_size: number | null
  /** True when the checkpoint is a mixture-of-experts model (read from nested configs too). */
  is_moe: boolean
  /** False when FreeToken has no implementation for this architecture. */
  supported: boolean
  unsupported_reason: string | null
  /** 'llm' is served by the FreeToken engine; 'timeseries' by a forecasting server. */
  category: 'llm' | 'timeseries'
  /** For time-series checkpoints: whether this build has an adapter for it. */
  ts_servable: boolean
  ts_note: string | null
}

export type EngineState = 'stopped' | 'starting' | 'running' | 'stopping' | 'error'

export type Diagnosis = {
  /** The real error line pulled out of the engine's output. */
  summary: string
  /** What to do about it, when the failure is one we recognise. */
  hint: string | null
  doc: string | null
  /** Last meaningful output lines, warnings stripped. */
  tail: string[]
  exit_code: number | null
}

export type EngineStatus = {
  state: EngineState
  error: string | null
  diagnosis: Diagnosis | null
  pid: number | null
  model_id: string | null
  model_path: string | null
  options: Record<string, unknown>
  command: string
  started_at: number | null
  uptime_s: number
  engine_url: string
}

export type Health = {
  status: 'ok' | 'loading' | 'error'
  model?: string | null
  phase?: string
  progress?: { done_bytes: number; total_bytes: number }
  message?: string
  uptime_s?: number
  version?: string
}

export type Stats = {
  model: { id: string; ctx: number; attn: string; moe: boolean }
  uptime_s: number
  kv: { used_pages: number; total_pages: number; page_size: number } | null
  swa: { used_pages: number; total_pages: number; page_size: number } | null
  mamba: { used_slots: number; total_slots: number } | null
  vram_bytes: number
  throughput: { decode_tps: number; prefill_tps: number }
  requests: {
    active: number
    completed: number
    p95_ms: number
    ttft_mean_ms: number
    prompt_tokens_total: number
    completion_tokens_total: number
  }
}

export type CacheGeometry = {
  num_pages: number
  page_size: number
  moe_cache_size: number
  num_experts: number
  num_moe_layers: number
  moe_cache_policy: string | null
  unit_bytes: {
    kv_per_token: number
    moe_per_expert: number
    mamba_per_slot: number
    swa_per_token: number
  }
  cache_budget_bytes: number
  limits?: Record<string, { min?: number; max?: number }>
}

export type CacheDoc = {
  state: string
  last_rebuild: Record<string, unknown> | null
  geometry: CacheGeometry
}

export type LoadStep = {
  slug: string
  label: string
  state: 'done' | 'active' | 'pending'
  pct: number | null
}

export type LoadStatus = {
  phase: string
  label: string
  detail: string
  /** Progress within the current phase; null when the phase reports no number. */
  phase_pct: number | null
  overall_pct: number
  done_bytes: number
  total_bytes: number
  /** phase_pct came from the memory-growth heuristic, not from the engine. */
  estimated: boolean
  steps: LoadStep[]
}

/** One engine instance with its own health, stats and load progress. */
export type EngineDetail = EngineStatus & {
  instance_id: string
  port: number
  /** GPU index this engine holds, in nvidia-smi order. '' means every card. */
  gpus: string | null
  served_name: string | null
  health: Health | null
  stats: Stats | null
  load: LoadStatus | null
}

export type LoadedModel = {
  instance_id: string
  model: string | null
  served_name: string | null
  state: string
  port: number
  gpus: string | null
  ready: boolean
  /** Set for a model served by another computer (`model@computer`). */
  remote?: { node_id: string; node: string; address: string; decode_tps: number | null; active: number | null } | null
  /** Set for an enabled hosted model (`id@groq` / `id@openrouter`); it has no engine instance. */
  external?: {
    provider: string
    provider_label: string
    id: string
    price_in: number | null
    price_out: number | null
    price_blended: number | null
    speed: number | null
  } | null
  context?: number | null
  /** Published scores (app/ratings.py): SWE-bench Verified %, AA Intelligence Index. */
  swe?: number | null
  aa?: number | null
}

export type UnloadReport = {
  engines: { instance: string; model: string | null; ok: boolean; error?: string }[]
  forecasters: string[]
  /** python.exe processes still listening on engine ports that no tracked engine owned. */
  orphans: { pid: number; ports: number[]; killed: boolean }[]
  vram_before: Record<string, number>
  seconds: number
}

export type ConsoleDoc = {
  engine: EngineStatus
  /** Every engine, each with its own numbers. */
  engines: EngineDetail[]
  loaded: LoadedModel[]
  health: Health | null
  stats: Stats | null
  cache: CacheDoc | null
  load: LoadStatus | null
  gpus: (GpuInfo & { enabled: boolean })[]
  /** Cards present in the machine but excluded from the engine pool. */
  gpus_hidden: number
  host_memory: {
    total_bytes: number
    available_bytes: number
    pin_budget_bytes: number
  }
  ts: number
}

export type LogEntry = { seq: number; ts: number; stream: string; text: string }

export type SystemDoc = {
  gpus: (GpuInfo & { enabled: boolean })[]
  host_memory: {
    total_bytes: number
    available_bytes: number
    pin_budget_bytes: number
  }
  toolchain: { vcvars: string | null; cuda_home: string | null; nvcc: string | null }
  config: {
    engine_url: string
    visible_devices: string
    model_roots: string[]
    python: string
  }
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    // FastAPI puts the useful text in `detail`; fall back to the raw body so an
    // unexpected 500 still surfaces something actionable instead of "500".
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status)
  }
  return res.json() as Promise<T>
}


// ---- Python sandbox -------------------------------------------------------------------
// Code from chat runs in the ui/sandbox container, which has no route off the host. A run
// returns whatever the script wrote to its working directory as `artifacts`.
export type SandboxArtifact = { name: string; size: number; mime: string }

export type SandboxRun = {
  run_id: string
  ok: boolean
  exit_code: number | null
  timed_out: boolean
  duration_s: number
  stdout: string
  stderr: string
  truncated: boolean
  artifacts: SandboxArtifact[]
}

export type SandboxStatus = { available: boolean; reason?: string; docker?: string; image?: string }

/** URL the browser fetches an artifact from (proxied, so :8200 is never exposed). */
export function artifactUrl(runId: string, name: string): string {
  return `/api/sandbox/artifacts/${runId}/${name.split('/').map(encodeURIComponent).join('/')}`
}

// ---- Time-series models ---------------------------------------------------------------
export type TsHealth = {
  model: string
  family: string
  device: string
  gpu: string | null
  vram_bytes: number
  context_length: number | null
  native_horizon: number | null
  channels: number | null
  probabilistic: boolean
  load_seconds: number
}

export type TsInstance = {
  id: string
  model_id: string
  gpu: string
  port: number
  state: 'starting' | 'running' | 'error' | 'stopped'
  error: string | null
  uptime_s: number
  health: TsHealth | null
  log_tail: string[]
  /** Forecasts running right now, lifetime call count, and the last call's time/latency. */
  in_flight?: number
  calls?: number
  last_call_at?: number | null
  last_seconds?: number | null
}

export type TsForecast = {
  model: string
  family: string
  horizon: number
  seconds: number
  forecasts: { median: number[]; quantiles: Record<string, number[]> | null }[]
  notes: string[]
}

/** Where a model runs: this computer's engines, a paired computer, or a hosted provider. */
export type TokenSource = 'local' | 'network' | 'external'
export type TokenWindow = 'all' | '24h' | '7d' | '30d'

export type TokenCounts = {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  requests: number
  /** Requests whose reply carried no usage block: counted, but their tokens are unknown. */
  unmetered: number
}

export type TokenUsageRow = TokenCounts & {
  model: string
  source: TokenSource
  first_seen: number | null
  last_seen: number | null
}

/** GET /api/usage/tokens -- tokens processed per model, largest first (app/tokens.py). */
export type TokenUsageDoc = {
  window: TokenWindow
  since: number | null
  models: TokenUsageRow[]
  totals: TokenCounts
  by_source: Record<TokenSource, TokenCounts>
  ts: number
}

export const api = {
  system: () => request<SystemDoc>('/api/system'),
  models: () => request<{ models: ModelEntry[] }>('/api/models'),
  console: () => request<ConsoleDoc>('/api/console'),
  tokenUsage: (since: TokenWindow = 'all') =>
    request<TokenUsageDoc>(`/api/usage/tokens?since=${since}`),
  engine: () => request<EngineStatus & { health: Health | null }>('/api/engine'),

  start: (model: string, options: Record<string, unknown>) =>
    request<EngineStatus>('/api/engine/start', {
      method: 'POST',
      body: JSON.stringify({ model, options }),
    }),

  stop: () => request<EngineStatus>('/api/engine/stop', { method: 'POST' }),

  sandboxStatus: () => request<SandboxStatus>('/api/sandbox/status'),

  tsInstances: () => request<{ instances: TsInstance[] }>('/api/ts'),
  tsStart: (model: string, gpu?: string) =>
    request<TsInstance>('/api/ts', { method: 'POST', body: JSON.stringify({ model, gpu }) }),
  tsStop: (id: string) => request<TsInstance>(`/api/ts/${id}/stop`, { method: 'POST' }),
  tsForecast: (body: { model?: string; series: number[]; horizon: number; quantiles?: number[] }) =>
    request<TsForecast>('/api/ts/forecast', { method: 'POST', body: JSON.stringify(body) }),

  runInSandbox: (code: string, timeoutS = 60) =>
    request<SandboxRun>('/api/sandbox/run', {
      method: 'POST',
      body: JSON.stringify({ code, timeout_s: timeoutS }),
    }),

  rebuildCache: (payload: Record<string, number>) =>
    request<unknown>('/api/engine/cache/rebuild', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  logs: (cursor: number, instanceId?: string) =>
    request<{ entries: LogEntry[]; next_cursor: number }>(
      `/api/logs?cursor=${cursor}${instanceId ? `&instance_id=${encodeURIComponent(instanceId)}` : ''}`,
    ),

  engines: () =>
    request<{ engines: EngineDetail[]; loaded: LoadedModel[] }>('/api/engines'),

  /** Load a model into a NEW engine, alongside anything already running. */
  startEngine: (model: string, options: Record<string, unknown>, gpus?: string) =>
    request<EngineStatus>('/api/engines', {
      method: 'POST',
      body: JSON.stringify({ model, options, gpus: gpus || null }),
    }),

  /** Options each model was last launched with, keyed by model id. */
  launchOptions: () =>
    request<{
      models: Record<string, { options: Record<string, unknown>; gpus: string | null }>
    }>('/api/launch-options'),

  stopEngine: (instanceId: string) =>
    request<EngineStatus>(`/api/engines/${instanceId}/stop`, { method: 'POST' }),

  /** Unload a model from its GPU and reload it on another, keeping its launch options. */
  /** Stop every engine and forecaster, and kill orphaned engine workers still holding memory. */
  unloadAll: () => request<UnloadReport>('/api/engines/unload-all', { method: 'POST' }),

  moveEngine: (instanceId: string, gpus: string) =>
    request<EngineStatus>(`/api/engines/${instanceId}/move`, {
      method: 'POST',
      body: JSON.stringify({ gpus }),
    }),

  requests: (since: number) =>
    request<{ entries: Record<string, unknown>[]; next_cursor: number }>(
      `/api/engine/requests?since=${since}`,
    ),
}

/**
 * Stream a chat completion. Yields incremental content deltas.
 *
 * The control plane relays the engine's SSE bytes untouched, so this parses the OpenAI
 * wire format directly. Chunks arrive split at arbitrary byte boundaries, so the tail of a
 * partial event is carried across reads rather than dropped.
 */
export async function* streamChat(
  body: {
    messages: { role: string; content: string }[]
    /** `name@computer` routes to a paired computer; omitted uses the local engine. */
    model?: string | null
    temperature?: number
    top_p?: number
    max_tokens?: number
  },
  signal?: AbortSignal,
): AsyncGenerator<{ delta?: string; reasoning?: string; error?: string; finish?: string }> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  })

  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      if (j?.detail) detail = String(j.detail)
    } catch {
      /* ignore */
    }
    yield { error: detail }
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE events are separated by a blank line; keep the trailing partial in `buffer`.
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''

    for (const event of events) {
      for (const line of event.split('\n')) {
        if (!line.startsWith('data:')) continue
        const data = line.slice(5).trim()
        if (!data || data === '[DONE]') continue
        try {
          const parsed = JSON.parse(data)
          if (parsed.error) {
            yield { error: String(parsed.error) }
            continue
          }
          const choice = parsed.choices?.[0]
          const delta = choice?.delta
          if (delta?.content) yield { delta: delta.content }
          // gpt-oss and the Qwen thinking models emit a separate reasoning channel.
          if (delta?.reasoning_content) yield { reasoning: delta.reasoning_content }
          // A thinking model can spend the WHOLE budget reasoning and emit no content at
          // all (finish_reason 'length' with an empty content channel). Without this the
          // caller cannot tell that apart from a successful empty reply, and the UI shows
          // a blank bubble.
          if (choice?.finish_reason) yield { finish: String(choice.finish_reason) }
        } catch {
          /* a keep-alive or a frame we do not model */
        }
      }
    }
  }
}
