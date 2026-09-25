// Standing objectives: a goal the swarm improves on continuously, scored by a trusted harness.
// Backend: ui/backend/app/objectives.py (routes under /api).

export type MetricKind =
  | 'sharpe'
  | 'sortino'
  | 'calmar'
  | 'total_return'
  | 'cagr'
  | 'max_drawdown'
  | 'reported'
  | 'judge'

export const RETURN_METRICS: MetricKind[] = ['sharpe', 'sortino', 'calmar', 'total_return', 'cagr', 'max_drawdown']

export const METRIC_OPTIONS: { kind: MetricKind; label: string; hint: string }[] = [
  { kind: 'sharpe', label: 'Sharpe ratio', hint: 'risk-adjusted return of the daily return stream' },
  { kind: 'sortino', label: 'Sortino ratio', hint: 'like Sharpe, but only downside volatility counts' },
  { kind: 'calmar', label: 'Calmar ratio', hint: 'CAGR divided by the worst drawdown' },
  { kind: 'total_return', label: 'Total return', hint: 'compounded return over the scored period' },
  { kind: 'cagr', label: 'CAGR', hint: 'annualised compounded return' },
  { kind: 'max_drawdown', label: 'Max drawdown', hint: 'smallest peak-to-trough loss wins' },
  { kind: 'reported', label: 'Reported score', hint: 'the script reports a number with ft.report_score()' },
  { kind: 'judge', label: 'Judge (0-10)', hint: 'a second model scores each answer against a rubric' },
]

export type MetricSpec = {
  kind: MetricKind
  higher_is_better: boolean
  periods_per_year: number
  min_active_days: number
  rubric: string
  price_column: string | null
  cost_bps: number
  max_leverage: number
  mid_cut?: string
}

export type SegmentStats = {
  days: number
  active_days: number
  sharpe?: number | null
  sortino?: number | null
  total_return?: number
  cagr?: number | null
  max_drawdown?: number
  calmar?: number | null
  volatility?: number
  win_rate?: number | null
}

export type CandidateMetrics = {
  in_sample?: SegmentStats
  holdout?: SegmentStats
  full?: SegmentStats
  warning?: string
  extra?: Record<string, number | string>
  execution?: {
    positions: number
    position_changes: number
    bars: number
    cost_bps: number
    max_leverage: number
    price_column: string
  }
  source?: string
}

export type CandidateStatus = 'evaluating' | 'ok' | 'error'
export type Verdict = 'pass' | 'fail' | 'error' | 'skipped'
export type AuditState = 'none' | 'pending' | 'pass' | 'fail'

export type Candidate = {
  id: string
  objective_id: string
  seq: number
  created_at: number
  model: string | null
  mode: string | null
  parent_id: string | null
  rationale: string
  status: CandidateStatus
  score: number | null
  is_score: number | null
  score_note: string
  metrics: CandidateMetrics
  lookahead: Verdict
  lookahead_detail: string
  audit: AuditState
  audit_notes: string
  eval_seconds: number | null
  champion_at: number | null
}

export type CandidateFull = Candidate & {
  code: string
  answer: string
  returns: [string, number][]
  stdout: string
  stderr: string
  run_id: string | null
}

export type Objective = {
  id: string
  project_id: string
  title: string
  description: string
  metric: MetricSpec
  metric_label: string
  status: 'running' | 'paused' | 'stopped'
  dataset: string | null
  time_column: string | null
  split_date: string | null
  lookahead_check: boolean
  require_audit: boolean
  eval_timeout_s: number
  cooldown_s: number
  best_id: string | null
  created_at: number
  updated_at: number
  candidates: number
  candidates_ok: number
  candidates_error: number
  evaluating: number
  improvements: number
  last_improvement_at: number | null
  last_candidate_at: number | null
  best: Candidate | null
}

export type Point = {
  id: string
  seq: number
  created_at: number
  status: CandidateStatus
  score: number | null
  lookahead: Verdict
  audit: AuditState
  model: string | null
  champion_at: number | null
}

export type ObjectiveDetail = Objective & {
  notes: { id: number; ts: number; author: string; text: string }[]
  lessons: { id: number; ts: number; model: string | null; candidate_id: string | null; text: string }[]
  champions: { id: string; seq: number; score: number; is_score: number | null; model: string; champion_at: number }[]
  points: Point[]
}

export type Probe = {
  dataset: string
  columns: { name: string; type: string }[]
  numeric_columns: string[]
  price_column: string | null
  time_column: string | null
  first_date?: string
  last_date?: string
  days?: number
  split_date?: string | null
  mid_cut?: string | null
  note?: string
}

export type NewObjective = {
  title: string
  description: string
  metric: Partial<MetricSpec> & { kind: MetricKind }
  dataset?: string | null
  time_column?: string | null
  split_date?: string | null
  lookahead_check?: boolean
  require_audit?: boolean
  eval_timeout_s?: number
  cooldown_s?: number
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON body */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

const e = encodeURIComponent

export type DemoteResult = {
  demoted: number
  audit: string
  lesson: string
  was_champion: boolean
  /** Library modules the demoted result was built on, now marked do-not-use. */
  quarantined: string[]
  /** Other candidates that imported those modules, disqualified with it. */
  cascaded: number[]
  recrowned: { id: string; seq: number } | null
  descendants: { id: string; seq: number; model: string | null; score: number | null; audit: string; rationale: string }[]
}

export const objectives = {
  list: (projectId: string) => req<{ objectives: Objective[] }>(`/api/projects/${e(projectId)}/objectives`),
  get: (id: string) => req<ObjectiveDetail>(`/api/objectives/${e(id)}`),
  create: (projectId: string, body: NewObjective) =>
    req<Objective>(`/api/projects/${e(projectId)}/objectives`, { method: 'POST', body: JSON.stringify(body) }),
  probe: (projectId: string, dataset: string, holdout: number) =>
    req<Probe>(`/api/projects/${e(projectId)}/objectives/probe?dataset=${e(dataset)}&holdout=${holdout}`),
  setStatus: (id: string, status: Objective['status']) =>
    req<Objective>(`/api/objectives/${e(id)}`, { method: 'PATCH', body: JSON.stringify({ status }) }),
  update: (id: string, body: { title?: string; description?: string; cooldown_s?: number }) =>
    req<Objective>(`/api/objectives/${e(id)}`, { method: 'PATCH', body: JSON.stringify(body) }),
  remove: (id: string) => req<{ ok: boolean }>(`/api/objectives/${e(id)}`, { method: 'DELETE' }),
  steer: (id: string, text: string) =>
    req<{ id: number }>(`/api/objectives/${e(id)}/notes`, { method: 'POST', body: JSON.stringify({ text }) }),
  candidates: (id: string, order: 'rank' | 'recent' = 'rank', limit = 50) =>
    req<{ candidates: Candidate[]; disqualified?: Candidate[] }>(
      `/api/objectives/${e(id)}/candidates?order=${order}&limit=${limit}`,
    ),
  candidate: (id: string, cid: string) => req<CandidateFull>(`/api/objectives/${e(id)}/candidates/${e(cid)}`),
  /** Disqualify a result the automated checks passed, and teach the team why. */
  demote: (id: string, cid: string, body: { finding: string; lesson?: string; reviewer?: string; to_playbook?: boolean }) =>
    req<DemoteResult>(`/api/objectives/${e(id)}/candidates/${e(cid)}/demote`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  dataCatalog: (projectId: string) =>
    req<{ files: { view: string; path: string; format: string; bytes: number }[] }>(
      `/api/projects/${e(projectId)}/data/catalog`,
    ),
  /** Delete candidates for good: a selection by id, or a whole scope ("delete all ranked"). */
  deleteCandidates: (id: string, body: { ids?: string[]; scope?: DeleteScope }) =>
    req<DeleteCandidatesResult>(`/api/objectives/${e(id)}/candidates/delete`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  /** Delete lessons, steering notes or escalation ideas: by id, or all of them. */
  deleteRows: (id: string, kind: 'lessons' | 'notes' | 'ideas', body: { ids?: number[]; all?: boolean }) =>
    req<{ deleted: number }>(`/api/objectives/${e(id)}/${kind}/delete`, { method: 'POST', body: JSON.stringify(body) }),
}

export type DeleteScope = 'ranked' | 'disqualified' | 'all'

export type DeleteCandidatesResult = {
  deleted: number[]
  /** Left in place: still evaluating, queued in a running look-ahead re-test, or not found. */
  skipped: { id: string; seq: number | null; reason: string }[]
  /** Seq of the candidate crowned because the best was deleted. */
  recrowned: number | null
  lost_best: boolean
}

/** Whether a point counts as ranked / disqualified -- the same filters as the backend's
 *  `_ranked` / `_disqualified`, used to say how many a "delete all" will remove. */
export function isRanked(p: Pick<Point, 'status' | 'score' | 'lookahead' | 'audit'>): boolean {
  return p.status === 'ok' && p.score !== null && p.lookahead !== 'fail' && p.lookahead !== 'error' && p.audit !== 'fail'
}

export function isDisqualified(p: Pick<Point, 'status' | 'score' | 'lookahead' | 'audit'>): boolean {
  return p.status === 'ok' && p.score !== null && (p.audit === 'fail' || p.lookahead === 'fail')
}

/** Format a metric value the way people read it: ratios as numbers, returns as percents. */
export function fmtMetric(kind: MetricKind | string, v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  if (kind === 'total_return' || kind === 'cagr' || kind === 'max_drawdown') return `${(v * 100).toFixed(1)}%`
  return v.toFixed(Math.abs(v) >= 100 ? 0 : 2)
}
