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
  by_model: { provider: string; model: string; calls: number; prompt_tokens: number; completion_tokens: number; usd: number }[]
  days: { day: string; usd: number }[]
}

export type Overview = {
  providers: Provider[]
  enabled: string[]
  daily_limit_usd: number
  parallel_agents: number
  parallel_local_agents: number
  escalation: Escalation
  spend: Spend
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
    parallel_agents?: number
    parallel_local_agents?: number
    escalation?: Partial<Escalation>
  }) => req<Overview>('/api/external/config', { method: 'PUT', body: JSON.stringify(patch) }),
  test: (provider: string) => req<{ ok: boolean; detail: string }>(`/api/external/${provider}/test`, { method: 'POST' }),
  plan: (projectId: string) => req<SwarmPlan>(`/api/projects/${encodeURIComponent(projectId)}/swarm/plan`),
  setRole: (projectId: string, model: string, role: ModelRole) =>
    req<unknown>(`/api/projects/${encodeURIComponent(projectId)}/model-role`, { method: 'POST', body: JSON.stringify({ model, role }) }),
  escalation: (oid: string) => req<EscalationStatus>(`/api/objectives/${encodeURIComponent(oid)}/escalation`),
  escalateNow: (oid: string) =>
    req<{ model: string; rung: number; text: string }>(`/api/objectives/${encodeURIComponent(oid)}/escalation/run`, { method: 'POST' }),
}

export const isExternal = (name: string | null | undefined) => !!name && /@(groq|openrouter)$/.test(name)

export const usd = (v: number | null | undefined, digits = 2) =>
  v == null ? '—' : `$${v < 0.01 && v > 0 ? v.toFixed(4) : v.toFixed(digits)}`
