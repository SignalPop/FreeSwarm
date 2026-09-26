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
  /** What the leaderboard ranks on when there is a holdout (absent = robust). */
  rank?: 'robust' | 'holdout'
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
  /** R^2 of log equity against time, signed by the slope: 1 is a steady climb. */
  smoothness?: number | null
}

type RegimeSegStats = { days?: number; sharpe?: number | null; total_return?: number | null }
export type RegimeInfo = {
  name: string
  /** Regime label -> the signal traded in it. */
  routes: Record<string, string>
  /** [date, label]: the label the day spent most bars in. */
  days: [string, string][]
  /** [date, daily mean of the series the regime was derived from]. */
  signal?: [string, number][]
  by_label: Record<string, { in_sample?: RegimeSegStats; holdout?: RegimeSegStats }>
}

export type CandidateMetrics = {
  in_sample?: SegmentStats
  holdout?: SegmentStats
  full?: SegmentStats
  /** Which regime each day was in and what the regime was derived from (ft.route / ft.report_regime). */
  regime?: RegimeInfo
  /** How the robust ranking score was built (see _robust in objectives.py). Empty ({}) when
   *  the candidate could not be scored (e.g. too few active days) -- read it via robustRank(). */
  rank?: RobustRank | { method?: undefined }
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
  /** The metric before costs and with every position flipped (same costs), per segment. */
  costs?: {
    metric: string
    cost_bps: number | null
    changes_per_day: number
    in_sample?: CostSegment
    holdout?: CostSegment
    /** In-sample only: what the submitting agent was told to fix. */
    verdict?: string | null
  }
  /** Present on an ensemble (mode 'ensemble'): verified candidates combined into one portfolio. */
  ensemble?: EnsembleInfo
}

/** An ensemble: its daily return is sum_i w[i,t] * r[i,t] over the members' stored net returns
 *  (see ui/backend/app/ensembles.py). Members are in fixed order (by seq); every per-member
 *  array below follows that order. */
export type EnsembleInfo = {
  members: { id: string; seq: number; model: string | null; rationale: string }[]
  weighting: 'equal' | 'inverse_vol'
  lookback_days: number
  /** [date, [w1..wn]] per day; each row sums to 1 and reads only returns before that day. */
  weights: [string, number[]][]
  avg_weights: number[]
  avg_weights_in_sample?: number[]
  /** seq -> [date, net daily return] as each member was scored. */
  member_returns: Record<string, [string, number][]>
  member_stats?: { seq: number; in_sample_sharpe: number | null; holdout_sharpe: number | null; in_sample_score: number | null }[]
  /** n x n Pearson correlation of member daily returns, dates before the split only. */
  correlation_in_sample: (number | null)[][]
  members_note: string
}

export type CorrelationReport = {
  period: string
  days: number
  seqs: number[]
  matrix: (number | null)[][]
  candidates: {
    id: string
    seq: number
    model: string | null
    rationale: string
    in_sample_score: number | null
    in_sample_sharpe: number | null
    eligible: boolean
    why_not?: string
  }[]
  suggestions: {
    seqs: number[]
    avg_abs_rho: number | null
    members_in_sample_sharpe: Record<string, number | null>
    equal_weight_in_sample_sharpe: number | null
    equal_weight_in_sample_smoothness: number | null
  }[]
  note: string
}

export type CombineBody = {
  members: (number | string)[]
  weighting: 'equal' | 'inverse_vol'
  lookback_days?: number
  rationale?: string
  model?: string
}

/** Can this candidate join an ensemble? The same rule as the backend's ensembles.ineligible. */
export function ensembleEligible(c: Pick<Candidate, 'status' | 'lookahead' | 'audit' | 'mode'>): boolean {
  return c.status === 'ok' && c.lookahead === 'pass' && c.audit !== 'fail' && c.mode !== 'ensemble'
}

export type IdeaScore = {
  id: number
  model: string
  trigger: string
  minutes_ago: number
  idea: string
  tried: number
  ran: number
  failed: number
  best_in_sample: number | null
  best_seq: number | null
  median_in_sample: number | null
  best_diagnosis: string | null
  champions: number
}

export type ForecastScore = {
  view: string
  model: string | null
  series: string[] | null
  inputs: string[]
  horizon: number | null
  every: number | null
  skill: Record<string, { skill?: number | null; direction?: number | null; lift_from_inputs?: number | null }>
  used_by: number
  median_in_sample_users: number | null
  median_in_sample_no_forecast: number | null
  helped: number | null
  auto: boolean
  requested_by: string | null
}

export type Scoreboards = {
  ideas: IdeaScore[]
  forecasts: ForecastScore[]
  habits: { candidates: number; failed_to_run: number; results: Record<string, number>; changes_vs_parent: Record<string, number> }
  mentor_active: boolean
  mentor_due: string
}

export type CostSegment = {
  net: number | null
  gross: number | null
  inverted: number | null
  return_net: number | null
  return_gross: number | null
  return_inverted: number | null
}

export type CandidateStatus = 'evaluating' | 'ok' | 'error'
export type Verdict = 'pass' | 'fail' | 'error' | 'skipped' | 'pending'
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
  /** When the operator flagged this run as the shape they want (null = not flagged). */
  liked?: number | null
  liked_note?: string
}

export type CandidateFull = Candidate & {
  /** When an auditor (an agent, or "run audit now") took the pending audit; 0 = nobody yet. */
  audit_started?: number
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
  /** Scored candidates whose look-ahead test is still running in the background. */
  lookahead_pending?: number
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
  /** Score a failed candidate again, in place (same number, same code). */
  rerun: (id: string, cid: string) =>
    req<{ seq: number; status: string }>(`/api/objectives/${e(id)}/candidates/${e(cid)}/rerun`, { method: 'POST' }),
  /** The operator's own verdict: pass (or reinstate a failed audit) / fail, with the reason. */
  setAudit: (id: string, cid: string, passed: boolean, notes: string) =>
    req<{ audit: 'pass' | 'fail'; champion: boolean }>(`/api/objectives/${e(id)}/candidates/${e(cid)}/audit`, {
      method: 'POST',
      body: JSON.stringify({ passed, notes, model: 'operator' }),
    }),
  /** Audit a pending candidate now instead of waiting for an agent to take the chore. */
  runAudit: (id: string, cid: string, model?: string) =>
    req<{ audit: 'pass' | 'fail'; champion: boolean; model: string; notes: string }>(
      `/api/objectives/${e(id)}/candidates/${e(cid)}/audit/run`,
      { method: 'POST', body: JSON.stringify({ model: model ?? null }) },
    ),
  /** The team's memory: what each idea's and each forecast's candidates scored, and the team's habits. */
  scoreboards: (id: string) => req<Scoreboards>(`/api/objectives/${e(id)}/scoreboards`),
  /** Re-run a candidate in the scoring harness (ft, data, features, library). Nothing is recorded. */
  runCandidate: (id: string, cid: string) =>
    req<{ ok: boolean; stdout: string; stderr: string; duration_s: number }>(
      `/api/objectives/${e(id)}/candidates/${e(cid)}/run`,
      { method: 'POST' },
    ),
  /** Flag a run as the shape the operator wants (agents build on liked runs), or unflag it. */
  like: (id: string, cid: string, liked: boolean, note = '') =>
    req<{ liked: boolean }>(`/api/objectives/${e(id)}/candidates/${e(cid)}/like`, {
      method: 'POST',
      body: JSON.stringify({ liked, note }),
    }),
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
  /** Combine verified candidates into one weighted ensemble candidate; returns its in-sample view. */
  combine: (id: string, body: CombineBody) =>
    req<{ id: string; candidate_id: string; seq: number; status: string }>(`/api/objectives/${e(id)}/ensembles`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  /** In-sample daily-return correlations among candidates (or the top ranked), with low-|rho| sets. */
  correlations: (id: string, seqs?: number[], top = 12) =>
    req<CorrelationReport>(
      `/api/objectives/${e(id)}/correlations?top=${top}${seqs?.length ? `&seqs=${e(seqs.join(','))}` : ''}`,
    ),
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
  return p.status === 'ok' && p.score !== null && p.lookahead !== 'fail' && p.lookahead !== 'error' && p.lookahead !== 'pending' && p.audit !== 'fail'
}

export function isDisqualified(p: Pick<Point, 'status' | 'score' | 'lookahead' | 'audit'>): boolean {
  return p.status === 'ok' && p.score !== null && (p.audit === 'fail' || p.lookahead === 'fail')
}

/** Format a metric value the way people read it: ratios as numbers, returns as percents. */
/** Does the leaderboard rank on the robust score (weaker period x curve smoothness)? */
export function robustRanking(o: { split_date: string | null; metric: MetricSpec }): boolean {
  return !!o.split_date && (o.metric.rank ?? 'robust') === 'robust'
}

export type RobustRank = { method: 'robust'; base: number; smoothness: number; weaker: 'in_sample' | 'holdout'; holdout: number | null }

/** The robust-score breakdown, or null when there is none (the backend stores {} for an unscorable candidate). */
export function robustRank(m: CandidateMetrics | undefined): RobustRank | null {
  return m?.rank?.method ? m.rank : null
}

/** The candidate's metric on the holdout alone -- `score` is the ranking score. */
export function holdoutScore(c: { score: number | null; metrics?: CandidateMetrics }, kind: MetricKind): number | null {
  const m = c.metrics
  const r = robustRank(m)
  if (r) return r.holdout
  const v = m?.holdout?.[kind as keyof SegmentStats]
  return typeof v === 'number' ? v : c.score
}

export function fmtMetric(kind: MetricKind | string, v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  if (kind === 'total_return' || kind === 'cagr' || kind === 'max_drawdown') return `${(v * 100).toFixed(1)}%`
  return v.toFixed(Math.abs(v) >= 100 ? 0 : 2)
}
