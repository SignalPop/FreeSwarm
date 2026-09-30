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
  | 'task'

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
  { kind: 'task', label: 'Task server', hint: 'a data/action MCP serves the rows and scores the actions -- any table, any problem' },
]

/** One task a task server offers (GET /api/task-servers). */
export type TaskSummary = {
  name: string
  title: string
  target?: string
  score?: { name: string; higher_is_better: boolean }
  rows?: number
  first?: string
  holdout_from?: string | null
  action?: string
  error?: string
}

export type TaskServers = {
  servers: { server: string; tasks: TaskSummary[]; errors: string[] }[]
  errors: string[]
}

/** The operator's data-timing check of a task: columns whose change predicts the NEXT row's
 *  target move better than the current one (probably filed before they were known). */
export type LeakScan = {
  columns: { column: string; change_vs_current_move: number; change_vs_next_move: number; suspect: boolean; declared_ahead?: boolean }[]
  suspects: string[]
  declared_ahead?: string[]
  how?: string
}

/** A window of a task candidate's result, as its server manages it: the target as candles or a
 *  line, the managed state (a position, a battery's charge) and what the actions did. */
export type TaskDrill = {
  bars?: { kind: 'ohlc' | 'line'; columns: string[]; rows: (string | number | null)[][]; tz?: string; every?: string | null } | null
  state?: [string, number][]
  state_kind?: string
  events?: Record<string, unknown>[]
  problem?: string
}

/** A trade's class: a BIG WINNER, a BIG LOSER, or SCRATCH (anything between). */
export type TradeClass = 'win' | 'loss' | 'scratch'

/** One distinct trade of the trade leaderboard, with every candidate that took it. */
export type TradeRow = {
  entry: string
  exit: string
  side: 'long' | 'short'
  /** Result per unit of size, after costs (a return, or target units for a signed target). */
  unit: number
  net: number
  size: number
  bars: number
  holdout: boolean
  open: boolean
  cls: TradeClass
  takers: { seq: number; id: string }[]
}

export type TradeBook = {
  threshold: number
  threshold_source: 'set' | 'auto' | 'floor'
  additive: boolean
  split_date: string | null
  display_tz: string
  counts: Partial<Record<'in_sample' | 'holdout', { win: number; loss: number; scratch: number; win_total: number; loss_total: number; scratch_total: number }>>
  rows: TradeRow[]
  indexed: number
  indexed_trades: number
  index_errors: number
  candidates: number
  building: { running?: boolean; done?: number; total?: number } | null
}

/** What a task objective keeps of its task's description (metric.task_info). */
export type TaskInfo = {
  title?: string
  description?: string
  brief?: string
  target?: string
  action?: { kind?: string; min?: number | null; max?: number | null; initial?: number; description?: string }
  score?: { name: string; higher_is_better: boolean }
  rows?: number
  in_sample_rows?: number
  first?: string
  last_in_sample?: string
  holdout_from?: string | null
  version?: string
  /** The action rule the server manages under, e.g. "intraday+max_3_trades". */
  action_rule?: string
  columns?: { name: string; dtype?: string; role?: string; description?: string }[]
}

export type Direction = 'both' | 'long' | 'short'

export type MetricSpec = {
  kind: MetricKind
  higher_is_better: boolean
  periods_per_year: number
  min_active_days: number
  rubric: string
  price_column: string | null
  cost_bps: number
  max_leverage: number
  /** Which sides a position may take (absent = both); the other side is held as flat. */
  direction?: Direction
  /** Flat at each day's last bar: every trade opens and closes the same day (absent = may hold overnight). */
  intraday?: boolean
  /** With both sides allowed: longs and shorts must each be at least this share of in-sample trades to rank (0/absent = off). */
  min_side_share?: number
  /** kind 'task': candidates with fewer in-sample trades in all than this are not ranked (0/absent = off). */
  min_trades?: number
  /** kind 'task': the older daily quota -- fewer in-sample trades per day than this are not ranked (0/absent = off). */
  min_trades_per_day?: number
  mid_cut?: string
  /** What the leaderboard ranks on when there is a holdout (absent = robust). */
  rank?: 'robust' | 'holdout'
  /** kind 'task': the registered task server and task that score the candidates. */
  task_server?: string
  task?: string
  /** kind 'task': the MCP's choices this objective runs under (absent = the project's settings). */
  target?: string
  value_function?: string
  action_rule?: string
  task_info?: TaskInfo
}

export type SwingStats = {
  legs: number
  up_legs: number
  down_legs: number
  up_caught_long: number
  down_caught_short: number
  up_while_short: number
  down_while_long: number
  hits: number
  misses: number
  net: number
  net_per_leg: number
  capture: number | null
  legs_per_day: number
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

/** One day of the dataset's bars with a candidate's positions over them (GET .../day). Times are
 *  epoch seconds, UTC -- the same calendar day the daily returns are dated by. */
export type CandidateDay = {
  day: string
  /** Bars the dataset has that day; 0 for a day with none (a weekend, a holiday). */
  bars: number
  bar_s?: number
  /** Candle width: the dataset's bars merged so a day stays readable. */
  bucket_s?: number
  /** Open/high/low come from the dataset; false means only the price column exists. */
  ohlc?: boolean
  price?: string
  max_leverage?: number
  /** [t, open, high, low, close] per candle. */
  candles?: [number, number | null, number | null, number | null, number | null][]
  /** A run of one position: held from `from` to `to`; `ret` is the position times the price move, before costs. */
  spans?: { from: number; to: number; pos: number; ret: number }[]
  changes?: number
  /** Trades open at any point that day, including one carried in from the day before. */
  trades?: DayTrade[]
  cost_bps?: number
  /** How positions were recovered for a candidate scored before they were kept, or null. */
  recovered?: string | null
}

/** A run of one direction: opens leaving flat (or flipping), closes back to flat (or flipping).
 *  `net` also pays cost_bps on every unit traded -- entry, resizes, exit. */
export type DayTrade = {
  entry_t: number
  exit_t: number
  side: 1 | -1
  /** The largest position held during the trade. */
  size: number
  gross: number
  net: number
  /** Opened on an earlier day. */
  carried: boolean
  /** Still open when the data ends. */
  open: boolean
}

/** Trade statistics per UTC day; a trade counts on the day it opened. */
export type CalendarDay = {
  trades: number
  wins: number
  long: number
  short: number
  /** Of the long and short trades, how many made money net of costs. */
  long_wins?: number
  short_wins?: number
  /** Share of the day's bars spent holding a position. */
  exposure: number
  best?: number
  worst?: number
  hold_s?: number
}

export type TradeCalendar = {
  days: Record<string, CalendarDay>
  summary: {
    trades: number
    win_rate: number | null
    long: number
    short: number
    long_win_rate: number | null
    short_win_rate: number | null
    avg_win: number | null
    avg_loss: number | null
    profit_factor: number | null
    expectancy: number | null
    avg_hold_s: number | null
  }
  cost_bps: number
  recovered?: string | null
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
    intraday?: boolean
    /** Trades opened on each side in-sample (a flip opens one). */
    sides?: { long: number; short: number }
    price_column: string
  }
  source?: string
  /** A task objective's evaluation, as its task server returned it. */
  task?: {
    segments?: Record<string, Record<string, unknown>>
    diagnostics?: Record<string, Record<string, unknown>>
    notes?: string
    unranked?: string | null
    actions?: { reported: number; changes: number }
    score_name?: string
    higher_is_better?: boolean
    curve_kind?: 'returns' | 'additive'
  }
  /** Swing legs (zigzag of the price, per day) the positions sat on the right or wrong side of, per segment. */
  swings?: { swing_pct?: number } & Partial<Record<'in_sample' | 'holdout' | 'full', SwingStats>>
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
  setDirection: (id: string, direction: Direction) =>
    req<Objective>(`/api/objectives/${e(id)}/direction`, { method: 'POST', body: JSON.stringify({ direction }) }),
  taskServers: () => req<TaskServers>('/api/task-servers'),
  /** A task candidate's drill-down for a window, from its data/action MCP (harness_actions). */
  candidateActions: (id: string, cid: string, start: string, end: string) =>
    req<TaskDrill>(
      `/api/objectives/${e(id)}/candidates/${e(cid)}/actions?start=${e(start)}&end=${e(end)}&limit=500`,
    ),
  /** The trade leaderboard of a task objective: distinct trades of one class, best first. */
  trades: (id: string, cls: TradeClass | 'all', side: 'all' | 'long' | 'short', segment: 'all' | 'in_sample' | 'holdout', limit = 100, offset = 0) =>
    req<TradeBook>(
      `/api/objectives/${e(id)}/trades?cls=${e(cls)}&side=${e(side)}&segment=${e(segment)}&limit=${limit}&offset=${offset}`,
    ),
  /** What the big winners have in common: the whole book's (in-sample), or one candidate's. */
  tradeReview: (id: string, candidate?: string, segment: 'in_sample' | 'holdout' = 'in_sample') =>
    req<{ text: string; additive: boolean }>(
      `/api/objectives/${e(id)}/trades/review?segment=${segment}${candidate ? `&candidate=${e(candidate)}` : ''}`,
    ),
  /** The per-unit result that makes a trade BIG (bps, or target units for a signed target); null = automatic. */
  setTradeThreshold: (id: string, value: number | null) =>
    req<{ threshold: number; threshold_source: string }>(`/api/objectives/${e(id)}/trades/threshold`, {
      method: 'POST',
      body: JSON.stringify({ value }),
    }),
  reindexTrades: (id: string) => req<{ ok: boolean }>(`/api/objectives/${e(id)}/trades/reindex`, { method: 'POST' }),
  taskLeakScan: (server: string, task: string) =>
    req<LeakScan>(`/api/task-servers/${e(server)}/tasks/${e(task)}/leak-scan`),
  setIntraday: (id: string, intraday: boolean) =>
    req<Objective>(`/api/objectives/${e(id)}/intraday`, { method: 'POST', body: JSON.stringify({ intraday }) }),
  setSideShare: (id: string, min_side_share: number) =>
    req<Objective>(`/api/objectives/${e(id)}/sides`, { method: 'POST', body: JSON.stringify({ min_side_share }) }),
  /** A task objective's daily trade limit (0 = none); every candidate is re-scored from its kept actions. */
  setTradeLimit: (id: string, max_trades_per_day: number) =>
    req<Objective>(`/api/objectives/${e(id)}/trade-limit`, { method: 'POST', body: JSON.stringify({ max_trades_per_day }) }),
  /** A task objective's trade floor: in-sample trades in all, and/or the older daily quota (0 = off, absent = kept);
   *  candidates are ranked again from stored counts. */
  setMinTrades: (id: string, floor: { min_trades?: number; min_trades_per_day?: number }) =>
    req<Objective>(`/api/objectives/${e(id)}/min-trades`, { method: 'POST', body: JSON.stringify(floor) }),
  /** Re-score every task candidate from its kept actions under the server's current valuation. */
  rescore: (id: string) => req<Objective>(`/api/objectives/${e(id)}/rescore`, { method: 'POST' }),
  remarkProgress: (id: string) =>
    req<{ done?: number; failed?: number; total?: number; running?: boolean }>(`/api/objectives/${e(id)}/remark`),
  remove: (id: string) => req<{ ok: boolean }>(`/api/objectives/${e(id)}`, { method: 'DELETE' }),
  steer: (id: string, text: string) =>
    req<{ id: number }>(`/api/objectives/${e(id)}/notes`, { method: 'POST', body: JSON.stringify({ text }) }),
  candidates: (id: string, order: 'rank' | 'recent' = 'rank', limit = 50) =>
    req<{ candidates: Candidate[]; disqualified?: Candidate[] }>(
      `/api/objectives/${e(id)}/candidates?order=${order}&limit=${limit}`,
    ),
  candidate: (id: string, cid: string) => req<CandidateFull>(`/api/objectives/${e(id)}/candidates/${e(cid)}`),
  /** One day's bars and the candidate's positions over them, for the equity chart's click-through. */
  candidateDay: (id: string, cid: string, day: string) =>
    req<CandidateDay>(`/api/objectives/${e(id)}/candidates/${e(cid)}/day?day=${e(day)}`),
  /** Trades, wins, long/short and time in the market per day, for the P&L calendar. */
  candidateCalendar: (id: string, cid: string) =>
    req<TradeCalendar>(`/api/objectives/${e(id)}/candidates/${e(cid)}/calendar`),
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

/** `smoothness` is absent for a task objective: its score is the weaker segment alone. */
export type RobustRank = { method: 'robust'; base: number; smoothness?: number | null; weaker: 'in_sample' | 'holdout'; holdout: number | null }

/** The robust-score breakdown, or null when there is none (the backend stores {} for an unscorable candidate). */
export function robustRank(m: CandidateMetrics | undefined): RobustRank | null {
  return m?.rank?.method ? m.rank : null
}

/** The candidate's metric on the holdout alone -- `score` is the ranking score. */
export function holdoutScore(c: { score: number | null; metrics?: CandidateMetrics }, kind: MetricKind): number | null {
  const m = c.metrics
  const r = robustRank(m)
  if (r) return r.holdout
  // A task segment's score is `score`; a return stream's is the metric's own column.
  const v = (m?.holdout as Record<string, unknown> | undefined)?.[kind === 'task' ? 'score' : kind]
  return typeof v === 'number' ? v : c.score
}

export function fmtMetric(kind: MetricKind | string, v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  if (kind === 'total_return' || kind === 'cagr' || kind === 'max_drawdown') return `${(v * 100).toFixed(1)}%`
  return v.toFixed(Math.abs(v) >= 100 ? 0 : 2)
}
