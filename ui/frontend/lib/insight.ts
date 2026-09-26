// What the team has learned about its forecasts and signals: the forecast report (every forecast
// feature, what it read, whether it helped), explored Chronos-2 input combinations, and decile
// studies ("deci-plots"). Backend: ui/backend/app/tslab.py and ui/backend/app/deciplot.py.

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } })
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

// ----------------------------------------------------------------------------------------
// Forecast report
// ----------------------------------------------------------------------------------------
export type Paired = { gain: number | null; se: number | null; significant: boolean }

export type SeriesSkill = {
  skill?: number | null
  direction?: number | null
  coverage?: number | null
  points?: number | null
  /** Chronos-2 with inputs: the same forecast without them, at the same points. */
  without_inputs?: { skill?: number | null; direction?: number | null; coverage?: number | null } | null
  lift?: Paired | null
  lift_skill?: number | null
  lift_direction?: number | null
}

export type FeatureReport = {
  view: string
  created_at: number | null
  rows: number | null
  seconds: number | null
  auto: boolean
  requested_by: string | null
  inputs_sent: {
    model: string | null
    dataset: string | null
    target: string[] | null
    covariates: string[]
    calendar: boolean
    horizon: number | null
    every: number | null
    context: number | null
    bar: string | null
    samples: number | null
  }
  request: Record<string, unknown> | null
  columns: Record<string, string> | null
  skill: Record<string, SeriesSkill>
  used_by: number
  used_by_seqs: number[]
  verdict: 'helped' | 'hurt' | 'unclear' | 'unused'
  basis: 'vs parent' | 'vs median' | 'none'
  n: number
  effect: number | null
  pairs: { seq: number; parent_seq: number; delta: number }[]
  median_users: number | null
  median_no_forecast: number | null
}

/** A stored forecast feature drawn at a few in-sample anchors (backend: app/forecast_view.py).
 *  Times are epoch seconds (UTC); nothing observed at or after the split date is included. */
export type FeatureView = {
  view: string
  model: string | null
  dataset: string | null
  bar: string | null
  horizon: number
  context: number
  step_seconds: number
  split: string | null
  max_points: number
  bytes: number
  targets: string[]
  covariates: string[]
  calendar: boolean
  /** Inputs past the 8 drawable colours, not drawn. */
  dropped_series: number
  /** "re-run of the feature's model on the same inputs" (full paths) or "stored value at the horizon". */
  path_source: string
  note: string | null
  anchors: {
    as_of: string
    as_of_t: number
    /** Shared timestamps of the context window (decimated). */
    t: number[]
    /** Rows actually in the context window. */
    sent: number
    near_split: boolean
    /** Raw values over the window, with the window's mean and std for z-scoring. */
    inputs: { name: string; role: 'target' | 'covariate'; mean: number | null; std: number | null; v: (number | null)[] }[]
    forecasts: {
      target: string
      last: number | null
      /** The value the feature file stores: the forecast at the horizon. */
      end: { t: number; median: number | null; q10: number | null; q90: number | null; inside?: boolean }
      path?: { t: number[]; median: (number | null)[]; q10: (number | null)[]; q90: (number | null)[] }
      actual: { t: number[]; v: (number | null)[] }
    }[]
  }[]
  coverage: {
    endpoint: { n: number; inside: number; share: number | null }
    path: { n: number; inside: number; share: number | null }
  }
}

// ----------------------------------------------------------------------------------------
// Explored input combinations
// ----------------------------------------------------------------------------------------
export type Combo = {
  inputs: string[]
  ts: number
  author: string | null
  kind: string | null
  skill: number | null
  direction: number | null
  coverage: number | null
  qloss_rel: number | null
  points: number
  gain: number | null
  se: number | null
  significant: boolean
}

export type ComboGroup = {
  dataset: string
  target: string
  horizon: number
  bar: string | null
  model: string
  context: number
  points: number
  end: string | null
  last_ts: number
  baseline: Combo | null
  combos: Combo[]
  tested: number
  helpful: string[]
  hurts: string[]
  useless: string[]
  best: Combo | null
}

export type ExploreJob = {
  id: string
  phase: string
  done: number
  total: number
  current?: string
  cache_hits?: number
  error: string | null
  params: { target: string; horizon: number; bar: string | null }
  already_running?: boolean
}

// ----------------------------------------------------------------------------------------
// Decile studies
// ----------------------------------------------------------------------------------------
export type DeciVerdict = 'monotone' | 'weak' | 'extremes' | 'unstable' | 'flat' | 'no data'

export type DeciCellSummary = {
  timeframe?: string
  horizon: number
  spread_bps: number | null
  t_spread: number | null
  spearman: number | null
  consistency: number | null
  verdict: DeciVerdict
  n?: number
}

export type DeciSummary = {
  best: DeciCellSummary | null
  by_timeframe: Record<string, DeciCellSummary>
  verdict: DeciVerdict
  direction: string | null
}

export type DeciStudyRow = {
  id: number
  objective_id: string | null
  key: string
  ts: number
  author: string | null
  signal: string
  kind: 'dataset' | 'feature'
  timeframes: string[]
  horizons: number[]
  window: { days?: number; bars?: number }
  cut: string | null
  summary: DeciSummary
  runs?: number
}

export type DeciDecile = { decile: number; n: number; mean_bps: number | null; median_bps: number | null; hit: number | null; t: number | null }

export type DeciCell = {
  deciles: DeciDecile[]
  n: number
  spearman: number | null
  spread_bps: number | null
  t_spread: number | null
  mean_all_bps: number | null
  consistency: number | null
  verdict: DeciVerdict
  periods: { from: string; to: string; spread_bps: number | null; t_spread: number | null; spearman: number | null; mean_by_decile: (number | null)[] }[]
}

export type DeciStudy = DeciStudyRow & {
  cached?: boolean
  result: {
    rows: number
    cut: string | null
    coverage: number
    timeframes: Record<string, { bars: number; bucketed: number; warmup_until: string | null; from: string | null; to: string | null; horizons: Record<string, DeciCell> }>
    summary: DeciSummary
  }
}

export type DeciBatch = {
  phase: 'running' | 'done' | 'error' | 'cancelled'
  total: number
  done: number
  already_studied: number
  failed: { signal: string; error: string }[]
  current: string | null
  error: string | null
  started_at: number
  finished_at?: number
}

export const insight = {
  forecastReport: (oid: string) =>
    req<{ features: FeatureReport[]; split_date: string | null; note: string }>(`/api/tslab/forecast-report/${e(oid)}`),
  /** May re-run the feature's model at the drawn anchors (a handful of forecasts) for full paths. */
  forecastView: (oid: string, view: string, anchors = 5) =>
    req<FeatureView>(`/api/tslab/forecast-view/${e(oid)}/${e(view)}?anchors=${anchors}`),
  combos: (projectId: string, oid: string, target?: string) =>
    req<{ groups: ComboGroup[]; running: ExploreJob[] }>(
      `/api/tslab/combos?project_id=${e(projectId)}&objective_id=${e(oid)}${target ? `&target=${e(target)}` : ''}`,
    ),
  explore: (body: {
    project_id: string
    objective_id: string
    target: string
    inputs: string[]
    horizon: number
    bar?: string | null
    budget: number
    author?: string
  }) => req<ExploreJob>(`/api/tslab/explore`, { method: 'POST', body: JSON.stringify(body) }),

  deciList: (oid: string) =>
    req<{ studies: DeciStudyRow[]; batch: DeciBatch | null; split_date: string | null; defaults: { timeframes: string[]; horizons: number[]; window_days: number } }>(
      `/api/objectives/${e(oid)}/deci-plots`,
    ),
  deciStudy: (id: number) => req<DeciStudy>(`/api/deci-plots/${id}`),
  deciRun: (oid: string, body: { signal: string; timeframes?: string[]; horizons?: number[]; window_days?: number; author?: string; force?: boolean }) =>
    req<DeciStudy>(`/api/objectives/${e(oid)}/deci-plots`, { method: 'POST', body: JSON.stringify(body) }),
  deciSignals: (oid: string) =>
    req<{ columns: string[]; features: { view: string; columns: string[] }[] }>(`/api/objectives/${e(oid)}/deci-plots/signals`),
  deciBatch: (oid: string, body: { columns?: string[]; author?: string }) =>
    req<DeciBatch>(`/api/objectives/${e(oid)}/deci-plots/batch`, { method: 'POST', body: JSON.stringify(body) }),
  deciBatchCancel: (oid: string) =>
    req<{ batch: DeciBatch | null }>(`/api/objectives/${e(oid)}/deci-plots/batch/cancel`, { method: 'POST' }),
}

export const VERDICT_TONE: Record<string, string> = {
  monotone: 'text-good',
  weak: 'text-ink-dim',
  extremes: 'text-warn',
  unstable: 'text-warn',
  flat: 'text-ink-faint',
  'no data': 'text-ink-faint',
  helped: 'text-good',
  hurt: 'text-bad',
  unclear: 'text-ink-dim',
  unused: 'text-ink-faint',
}

export function num(v: number | null | undefined, d = 3): string {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—'
}

export function signed(v: number | null | undefined, d = 3): string {
  return typeof v === 'number' && Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(d)}` : '—'
}
