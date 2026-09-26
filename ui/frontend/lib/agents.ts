// The agent inspector (app/agent_activity.py): what each swarm agent is working on, what it
// was asked, and every tool call it made -- and who asked each forecaster for what.

export type AskedPrompt = {
  at: number
  /** The model the prompt went to: a peer model for audits and judging. */
  model: string | null
  system: string
  system_chars: number
  prompt: string
  prompt_chars: number
  /** Tools offered with it. */
  tools: string[]
}

export type ChatStat = {
  at: number
  model: string | null
  seconds: number
  prompt_tokens: number | null
  completion_tokens: number | null
  finish: string | null
  error: string | null
}

export type TimelineEntry =
  | { kind: 'asked'; at: number; index: number; model: string | null }
  | { kind: 'message'; at: number; text: string }
  | (ChatStat & { kind: 'chat'; tool_calls: string[]; said: string | null; reasoning: string | null })
  | { kind: 'tool'; at: number; name: string; args: unknown; ok: boolean; seconds: number; result: unknown }

export type Submission = {
  candidate_id?: string
  seq?: number
  status?: string
  in_sample_score?: number
  lookahead?: string
  rank?: number
  not_ranked?: string
  contender_for_best?: boolean
  judge_score?: number
  error?: string
  idea?: number | string | null
  rationale?: string
  at: number
}

export type ActivityRecord = {
  id: string
  /** explore | improve | build | audit | consolidate | practices | mentor | task | chat */
  mode: string
  /** running | submitted | no submission | done | interrupted */
  status: string
  outcome?: string
  objective: { id: string; title: string } | null
  parent?: { id: string; seq: number; rank: number | null; in_sample_score?: number | null; diagnosis?: string | null } | null
  audit_of?: { id: string; seq: number; model?: string; score?: number | null; is_score?: number | null } | null
  task?: { id: string; title: string } | null
  why?: string
  idea: number | string | null
  ideas_offered?: { id: number; model: string; tried: number; text: string }[]
  context?: Record<string, number | null>
  started_at: number
  ended_at: number | null
  pending:
    | { kind: 'chat'; model: string | null; since: number }
    | { kind: 'tool'; name: string; since: number; args: unknown }
    | null
  asked: AskedPrompt[]
  timeline: TimelineEntry[]
  timeline_dropped?: number
  chats: ChatStat[]
  submissions: Submission[]
  tokens: { prompt: number; completion: number; chats: number }
  received_at?: number
}

export type AgentActivity = {
  agent: string
  model: string
  role: string
  slot: number
  project_id: string | null
  updated_at: number
  seconds_since_update: number
  records: ActivityRecord[]
}

export type PeerCall = {
  agent: string
  record_id: string
  mode: string
  objective: { id: string; title: string } | null
  started_at: number
  chats: number
  prompt_tokens: number
  completion_tokens: number
  asked: AskedPrompt[]
}

export type ForecastRequest = {
  rid: string | number
  model: string
  /** The agent that asked (from the runner's X-FreeSwarm-Agent header); null = operator / API. */
  agent: string | null
  via: string
  path: string | null
  objective_id: string | null
  /** The recipe the caller asked for (columns, covariates, horizon...), when the endpoint names one. */
  request: Record<string, unknown> | null
  /** What reached the forecaster: batch size, context length, covariate names. */
  shape: {
    kind?: 'series' | 'covariates' | 'candles'
    horizon?: number
    anchors?: number
    context?: number
    targets?: number
    quantiles?: number
    past_covariates?: string[]
    future_covariates?: string[]
    samples?: number
    freq_seconds?: number
  }
  calls: number
  anchors: number
  seconds: number
  first_at: number
  last_at: number
  errors: number
  error: string | null
}

/** A forecast path over the horizon: timestamps (epoch seconds) with median and 10-90% band. */
export type ForecastBand = {
  t: number[]
  median: (number | null)[]
  q10: (number | null)[]
  q90: (number | null)[]
}

/** The numbers behind a request's sample anchors (backend: forecast_values.ValueCapture, served
 *  by /api/agents/forecast-values/{id}). Timestamps are epoch seconds (UTC). Nothing observed at
 *  or after the objective's split date is ever included. */
export type ForecastValues = {
  version: number
  horizon: number
  context: number
  step_seconds: number
  split: string | null
  /** Points per array after decimation (shrunk below 300 when the payload had to fit). */
  max_points: number
  bytes?: number
  anchors: {
    /** Index into the request's `samples` (the timeline's sample-anchor lines). */
    sample: number
    as_of: string
    as_of_t: number
    /** Set when the sample sat in the holdout and the last in-sample forecast is shown instead. */
    note: string | null
    streams: {
      /** Index into the request's `streams` (the timeline's blocks). */
      stream: number | null
      name: string
      role: string
      t: number[]
      lines: { label: string; v: (number | null)[] }[]
      /** Points actually sent, before decimation. */
      sent: number
    }[]
    forecasts: { target: string; with_inputs: ForecastBand; without_inputs?: ForecastBand }[]
    realized: { target: string; t: number[]; v: (number | null)[] }[]
  }[]
}

/** On a request's input detail: which anchors have values, and the id to fetch them by. */
export type ForecastValuesSummary = {
  id: string
  bytes: number
  anchors: { sample: number; as_of: string | null; note: string | null }[]
}

export type ModelActivity = {
  model: string
  now: number
  agents: AgentActivity[]
  peer_calls: PeerCall[]
  forecasts: ForecastRequest[]
}

async function req<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' } })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

// Model names may contain '/' (amazon/chronos-2): the route takes the rest of the path.
const modelPath = (m: string) => m.split('/').map(encodeURIComponent).join('/')

export const agentActivity = {
  forModel: (model: string, projectId?: string | null) =>
    req<ModelActivity>(
      `/api/agents/activity/${modelPath(model)}${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ''}`,
    ),
  /** Fetched on click, not polled: up to ~200 KB each. */
  forecastValues: (id: string) => req<ForecastValues>(`/api/agents/forecast-values/${encodeURIComponent(id)}`),
}
