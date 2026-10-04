/** The work log (backend: app/work.py): every iteration the swarm ran and every candidate it
 *  submitted, kept for reading a long (overnight) session after the fact. */

import type { ActivityRecord } from '@/lib/agents'

/** How an iteration (or candidate) ended. `done` is a chore: audit, mentor, practices... */
export type WorkOutcome = 'ok' | 'error' | 'no_submission' | 'interrupted' | 'running' | 'done'

/** A policy refusal: the runner said no on purpose (not an error). `experiment_budget`: run_python
 *  over the iteration's experiment budget; `truncated`: a call whose code arrived cut off. */
export type RefusalKind = 'experiment_budget' | 'truncated'

export const REFUSAL_LABEL: Record<RefusalKind, string> = {
  experiment_budget: 'experiment budget',
  truncated: 'truncated',
}

export function refusalLabel(kind: string | null | undefined): string {
  return kind ? (REFUSAL_LABEL[kind as RefusalKind] ?? kind.replace(/_/g, ' ')) : 'refused'
}

/** What a side request (a model call the runner made for the agent) was for. */
export function purposeLabel(purpose: string | null | undefined): string {
  if (!purpose) return 'side request'
  if (purpose === 'auto_repair') return 'auto-repair'
  if (purpose === 'answer_feedback') return 'answer to feedback'
  return purpose.replace(/_/g, ' ')
}

/** "2 refused (not errors): 1 experiment budget, 1 truncated · 1 resent ok" */
export function refusalsTitle(n: number, byKind: Partial<Record<string, number>> | undefined, resent: number): string {
  const kinds = Object.entries(byKind ?? {})
    .filter(([, v]) => (v ?? 0) > 0)
    .map(([k, v]) => `${v} ${refusalLabel(k)}`)
    .join(', ')
  return (
    `${n} call${n === 1 ? '' : 's'} refused by the runner's policy (not errors)${kinds ? `: ${kinds}` : ''}` +
    (resent ? ` · ${resent} truncated call${resent === 1 ? ' was' : 's were'} resent ok` : '')
  )
}

/** `state`: failed (stayed broken), recovered (a later call of the tool in the iteration fixed
 *  it), auto_repaired (the runner repaired the call), refused (a policy refusal, not an error;
 *  `recovered`: a truncated call resent ok), soft (a side request that failed, model busy; not an
 *  error). Older summaries have none: failed. */
export type WorkError = {
  tool: string
  line: string
  at: number | null
  state?: 'failed' | 'recovered' | 'auto_repaired' | 'refused' | 'soft'
  refusal?: RefusalKind
  recovered?: boolean
  /** For a soft one: what the side request was for (auto_repair, answer_feedback, ...). */
  purpose?: string
}

/** Policy refusals and failed side requests: counted apart from errors. */
type NotErrors = {
  /** run_python calls refused by policy (not run, not counted as experiments). */
  experiments_refused?: number
  /** Policy refusals (not errors), by kind; of them, truncated calls resent ok. */
  refusals?: number
  refusals_by_kind?: Partial<Record<RefusalKind, number>>
  refusals_recovered?: number
  /** Side requests (auto-repair, answer to feedback) that failed with the model busy: skipped, not errors. */
  chat_errors_soft?: number
}

export type IterationRow = {
  kind: 'iteration'
  id: string
  at: number
  ended: number | null
  agent: string
  model: string
  role: string | null
  objective_id: string | null
  objective_title: string | null
  mode: string
  status: string
  outcome: WorkOutcome
  /** The runner's own words, e.g. "#132 ok". */
  outcome_text: string | null
  /** Why it ended early (retired, restarted, repeat calls...). */
  reason: string | null
  tool_calls: number
  tools: Record<string, number>
  /** run_python calls; how many failed and stayed broken; how many failed and were fixed by a
   *  later run in the iteration or auto-repaired. */
  experiments: number
  experiments_failed: number
  experiments_recovered?: number
  /** Failed tool calls not fixed later in the iteration (still open while it runs). */
  tool_errors: number
  /** Failed calls a later call of the same tool fixed, plus calls the runner auto-repaired. */
  tool_errors_recovered?: number
  /** ... of which auto-repaired by the runner. */
  tool_auto_repaired?: number
  chats: number
  chat_errors: number
  /** Replies cut off at max_tokens (finish "length"). */
  truncations: number
  /** Runner -> model follow-ups: nudges, final-round prompts, the reflection request. */
  followups: number
  tokens: { prompt: number; completion: number; chats: number } | null
  submissions: {
    candidate_id: string | null
    seq: number | null
    status: string | null
    in_sample_score: number | null
    lookahead: string | null
    error: string | null
  }[]
  parent_seq: number | null
  idea: number | string | null
  hypothesis: string
  hypothesis_from: 'submission' | 'said' | 'reasoning' | null
  timeline_dropped: number
  /** The latest few that stayed broken; `error_count` is how many there were, `recovered_count`
   *  how many more the iteration fixed (recovered or auto-repaired; shown in its detail),
   *  `refused_count` / `soft_count` its refusals and skipped side requests (not errors). */
  errors: WorkError[]
  error_count: number
  recovered_count?: number
  refused_count?: number
  soft_count?: number
} & NotErrors

export type CandidateRow = {
  kind: 'candidate'
  id: string
  at: number
  objective_id: string
  objective_title: string | null
  seq: number
  model: string | null
  /** The agent whose iteration submitted it, when the work log has that iteration. */
  agent: string | null
  mode: string | null
  parent_seq: number | null
  status: 'ok' | 'error'
  outcome: 'ok' | 'error'
  score: number | null
  is_score: number | null
  holdout: number | null
  higher_is_better: boolean
  score_note: string
  lookahead: string | null
  eval_seconds: number | null
  idea: number | null
  rationale: string
  /** The last line of the traceback, for a failed one. */
  error_line: string
  /** A failed one whose iteration went on to submit one that ran. */
  recovered?: boolean
}

export type WorkRow = IterationRow | CandidateRow

export type ErrorGroup = { line: string; count: number; sources: Record<string, number>; last_at: number }

export type BestScore = {
  objective_id: string
  objective_title: string | null
  id: string
  seq: number
  model: string | null
  score: number
  is_score: number | null
  holdout: number | null
  higher_is_better: boolean
  at: number
}

export type WorkSummary = {
  iterations: number
  by_outcome: Partial<Record<WorkOutcome, number>>
  candidates: number
  candidates_ok: number
  candidates_error: number
  /** Failed candidates whose iteration went on to submit one that ran. */
  candidates_error_recovered?: number
  tool_calls: number
  experiments: number
  experiments_failed: number
  experiments_recovered?: number
  /** Only failures that stayed broken; `errors` groups only those too. */
  tool_errors: number
  /** Failures fixed later in their iteration, plus auto-repaired calls ... */
  tool_errors_recovered?: number
  /** ... of which auto-repaired. */
  tool_auto_repaired?: number
  chat_errors: number
  truncations: number
  /** Per objective, look-ahead failures left out. */
  best: BestScore[]
  errors: ErrorGroup[]
  agents: string[]
} & NotErrors

export type WorkDoc = {
  now: number
  /** Where this view starts: the window's start, or the clear marker when that is later and hidden. */
  since: number
  /** The window's own start. */
  window_since: number
  /** When the page was last cleared for this project (a view marker; nothing is deleted). */
  cleared_at: number | null
  /** The marker narrowed this view: work from before it is left out of the rows and the summary. */
  hiding_cleared: boolean
  include_cleared: boolean
  summary: WorkSummary
  /** Rows that pass the filters; `items` is the newest `limit` of them. */
  total: number
  items: WorkRow[]
  counts: { all: number; iteration: number; candidate: number }
}

/** What the iteration endpoint adds to a tool call on the timeline: whether it failed, and why. */
export type ToolMarks = {
  failed?: boolean
  /** A failure a later call of the same tool in the iteration fixed. */
  recovered?: boolean
  error_line?: string
  error_tail?: string
  /** The runner repaired this call (it is recorded ok); what it repaired. */
  auto_repaired?: boolean
  repair_attempts?: number | null
  repair_errors?: string[]
  /** A policy refusal (no `failed`): its kind; `recovered` then means a truncated call resent ok. */
  refused?: boolean
  refusal?: RefusalKind
  /** A runner step (answer_feedback) that failed softly: the iteration went on without it. */
  soft?: boolean
  soft_error?: string
}

/** What the iteration endpoint adds to a failed side request (a chat the runner made while a
 *  tool call ran): soft, its error moved from `error` to `soft_error`. */
export type ChatMarks = {
  soft?: boolean
  soft_error?: string
  purpose?: string
  /** "skipped (model busy)" */
  soft_note?: string
}

export type IterationDoc = {
  id: string
  project_id: string | null
  agent: string
  model: string
  role: string | null
  summary: Omit<IterationRow, 'kind' | 'id' | 'at' | 'ended' | 'agent' | 'model' | 'role' | 'error_count' | 'recovered_count' | 'refused_count' | 'soft_count'>
  record: ActivityRecord & { reason?: string; end_reason?: string }
}

export type CandidateDoc = {
  id: string
  objective_id: string
  objective_title: string | null
  project_id: string | null
  seq: number
  created_at: number
  model: string | null
  mode: string | null
  parent_id: string | null
  parent_seq: number | null
  rationale: string
  code: string
  status: 'ok' | 'error'
  score: number | null
  is_score: number | null
  holdout: number | null
  higher_is_better: boolean
  score_note: string
  metrics: Record<string, unknown>
  lookahead: string | null
  lookahead_detail: string
  audit: string | null
  audit_notes: string
  eval_seconds: number | null
  idea: number | null
  error_line: string
  stderr: string
  stdout: string
  /** Iterations in the work log that submitted it. */
  iterations: string[]
}

export type WorkFilters = {
  hours: number
  limit: number
  agent?: string
  outcome?: '' | WorkOutcome | 'tool_errors'
  kind?: '' | 'iteration' | 'candidate'
  q?: string
  /** Show the work from before the clear marker too. */
  includeCleared?: boolean
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

const e = encodeURIComponent

export const workApi = {
  list: (projectId: string, f: WorkFilters) => {
    const qs = new URLSearchParams({ hours: String(f.hours), limit: String(f.limit) })
    if (f.agent) qs.set('agent', f.agent)
    if (f.outcome) qs.set('outcome', f.outcome)
    if (f.kind) qs.set('kind', f.kind)
    if (f.q) qs.set('q', f.q)
    if (f.includeCleared) qs.set('include_cleared', '1')
    return req<WorkDoc>(`/api/projects/${e(projectId)}/work?${qs}`)
  },
  /** Hide the work so far (sets the project's clear marker to now; nothing is deleted). */
  clear: (projectId: string) => req<{ cleared_at: number }>(`/api/projects/${e(projectId)}/work/clear`, { method: 'POST' }),
  /** Undo the clear. */
  unclear: (projectId: string) => req<{ cleared_at: null }>(`/api/projects/${e(projectId)}/work/clear`, { method: 'DELETE' }),
  iteration: (id: string) => req<IterationDoc>(`/api/work/iterations/${e(id)}`),
  candidate: (id: string) => req<CandidateDoc>(`/api/work/candidates/${e(id)}`),
}
