/** External (hosted, pay-per-token) providers -- Groq and OpenRouter. See app/external.py. */

export type Provider = {
  id: 'groq' | 'openrouter'
  label: string
  base: string
  keys_url: string
  site: string
  key_set: boolean
}

export type Escalation = {
  enabled: boolean
  stuck_candidates: number
  stuck_minutes: number
  step_candidates: number
  step_minutes: number
}

export type Spend = {
  today: number
  limit: number
  /** Held back from search for "ideas when stuck" (already clamped to the limit). */
  ideas_reserve: number
  /** What search (every call but ideas) may still spend today. */
  search_left: number
  by_model: { provider: string; model: string; calls: number; prompt_tokens: number; completion_tokens: number; usd: number }[]
  days: { day: string; usd: number }[]
}

export type Overview = {
  providers: Provider[]
  enabled: string[]
  daily_limit_usd: number
  ideas_reserve_usd: number
  parallel_agents: number
  parallel_local_agents: number
  escalation: Escalation
  spend: Spend
}

/** GET /api/external/usage -- what each enabled hosted model does, costs and is blocked by.
 *  Spend and refusals are per model across the console; role, candidates and ideas are the
 *  project's. See app/escalation.py external_usage. */
export type ModelUsage = {
  model: string
  provider: string
  provider_label: string
  role: {
    allowed: boolean
    /** False when the provider has no API key. */
    loaded: boolean
    search: { agents: number; why: string | null } | null
    /** `rung` is 0-based, as in EscalationStatus. */
    ideas: { rung: number; of: number; why: string | null } | null
    /** `budget_paused`: out of the search until midnight because the search budget is spent. */
    reserved: { why: string | null; budget_paused: boolean } | null
  }
  calls_total: number
  usd_total: number
  last_call_ts: number | null
  calls_today: number
  usd_today: number
  prompt_tokens_today: number
  completion_tokens_today: number
  ideas_calls_today: number
  /** Today's budget refusals and provider errors (in memory; reset at midnight / restart). */
  refusals: { count: number; short: string | null; reason: string | null; kind: 'budget' | 'provider' | null; ts: number | null }
  candidates: { total: number; by_status: Record<string, number>; lookahead_pass: number; champions: number; last_ts: number | null } | null
  ideas: { count: number; last_ts: number | null } | null
}

export type EscalationError = { ts: number; detail: string; model?: string | null }

export type ExternalUsage = {
  today_usd: number
  limit_usd: number
  ideas_reserve_usd: number
  search_limit_usd: number
  search_budget_left: number
  /** Why hosted models are out of the search right now, or null. */
  search_paused: string | null
  escalation_errors: (EscalationError & { objective_id: string; title: string })[]
  models: ModelUsage[]
}

export type ExternalModel = {
  id: string
  name: string
  provider: string
  /** The name used everywhere else: `<id>@groq` / `<id>@openrouter`. */
  model: string
  context: number | null
  max_output: number | null
  price_in: number | null
  price_out: number | null
  price_blended: number | null
  priced: boolean
  /** Output tokens/s (Groq publishes it; OpenRouter does not). */
  speed: number | null
  swe: number | null
  aa: number | null
  rating_label: string | null
  tier: string
  available?: boolean
  reasoning?: boolean
  enabled: boolean
}

export type Catalog = {
  provider: string
  rows: ExternalModel[]
  note: string | null
  key_set: boolean
  prices_as_of: string
}

export type SwarmPlanEntry = {
  model: string
  kind: 'local' | 'network' | 'external'
  swe: number | null
  aa: number | null
  price_blended: number | null
  provider: string | null
  why: string
}

export type ModelRole = 'auto' | 'search' | 'ideas' | 'both'

export type SwarmPlan = {
  /** Every model the project allows, with its role and what it effectively does. */
  models: (SwarmPlanEntry & { role: ModelRole; searching: boolean; ideas: boolean })[]
  search: (SwarmPlanEntry & { agents?: number })[]
  reserved: SwarmPlanEntry[]
  ladder: SwarmPlanEntry[]
  not_in_ladder: SwarmPlanEntry[]
  swe_range: [number | null, number | null]
  swe_margin: number
  aa_range: [number | null, number | null]
  aa_margin: number
}

export type Idea = { id: number; ts: number; model: string; rung: number; text: string; trigger: string }

export type EscalationStatus = {
  config: Escalation
  stuck: boolean
  due: boolean
  candidates_since_improvement: number
  minutes_since_improvement: number
  ladder: SwarmPlanEntry[]
  next_rung: number | null
  next_model: string | null
  ideas: Idea[]
  /** Why the last due escalation produced no idea (budget refused, provider error...). */
  last_error: EscalationError | null
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const b = await res.json()
      if (b?.detail) detail = typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail)
    } catch {
      /* non-JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const external = {
  overview: () => req<Overview>('/api/external'),
  models: (provider: string, refresh = false) =>
    req<Catalog>(`/api/external/${provider}/models${refresh ? '?refresh=true' : ''}`),
  setEnabled: (model: string, enabled: boolean) =>
    req<{ enabled: string[] }>('/api/external/enabled', { method: 'PUT', body: JSON.stringify({ model, enabled }) }),
  save: (patch: {
    groq_api_key?: string
    openrouter_api_key?: string
    daily_limit_usd?: number
    ideas_reserve_usd?: number
    parallel_agents?: number
    parallel_local_agents?: number
    escalation?: Partial<Escalation>
  }) => req<Overview>('/api/external/config', { method: 'PUT', body: JSON.stringify(patch) }),
  test: (provider: string) => req<{ ok: boolean; detail: string }>(`/api/external/${provider}/test`, { method: 'POST' }),
  plan: (projectId: string) => req<SwarmPlan>(`/api/projects/${encodeURIComponent(projectId)}/swarm/plan`),
  setRole: (projectId: string, model: string, role: ModelRole) =>
    req<unknown>(`/api/projects/${encodeURIComponent(projectId)}/model-role`, { method: 'POST', body: JSON.stringify({ model, role }) }),
  usage: (projectId: string) =>
    req<ExternalUsage>(`/api/external/usage?project_id=${encodeURIComponent(projectId)}`),
  escalation: (oid: string) => req<EscalationStatus>(`/api/objectives/${encodeURIComponent(oid)}/escalation`),
  escalateNow: (oid: string) =>
    req<{ model: string; rung: number; text: string }>(`/api/objectives/${encodeURIComponent(oid)}/escalation/run`, { method: 'POST' }),
}

export const isExternal = (name: string | null | undefined) => !!name && /@(groq|openrouter)$/.test(name)

export const usd = (v: number | null | undefined, digits = 2) =>
  v == null ? '—' : `$${v < 0.01 && v > 0 ? v.toFixed(4) : v.toFixed(digits)}`
