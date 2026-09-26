// The Regime Lab: verified candidates measured inside market regimes, and the router that trades
// each regime with the candidate that works there. Backend: ui/backend/app/regimes.py.

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

export type FieldSplit = { field: string; n: number }
export type Split =
  | { kind: 'fields'; fields: FieldSplit[]; smooth: number; window_days: number }
  | { kind: 'module'; module: string }

/** One segment's numbers: daily Sharpe (annualised) over the days the regime was in force. */
export type Seg = { days: number; sharpe: number | null; pnl: number | null; hit: number | null }
export type Segs = { is: Seg; a: Seg; b: Seg; ho: Seg }

export type Regime = {
  label: string
  parts: [string, string][]
  share: { is: number | null; ho: number | null }
  days: { is: number; ho: number }
  dwell_min: number | null
}

export type Member = {
  seq: number
  id: string
  model: string
  score: number | null
  is_score: number | null
  eval_seconds: number | null
  rationale: string
}

export type Routes = Record<string, number | null>

export type LabResult = {
  version: number
  spec: Split
  describe: string
  split_date: string | null
  mid_date: string | null
  cost_bps: number
  dates: string[]
  regimes: Regime[]
  axes: { field: string; buckets: string[] }[]
  switches_per_day: number | null
  bar_seconds: number | null
  cells: Record<string, Record<string, Segs>>
  member_stats: Record<string, Segs>
  suggested: Routes
  routes: Routes
  routes_source: 'suggested' | 'custom'
  router: Segs
  best_single: { seq: number | null } & Partial<Segs>
  members: Member[]
  errors: Record<string, string>
  code?: string | null
  daily: {
    regime: number[]
    router: number[]
    contrib: Record<string, number[]>
    members: Record<string, number[]>
    cells: Record<string, Record<string, number[]>>
    bars: Record<string, number[]>
  }
}

export type RunSummary = {
  describe: string
  members: number[]
  regimes: number
  routed: number
  router: { is: number | null; a: number | null; b: number | null; ho: number | null }
  best_single: { seq: number | null; is: number | null; ho: number | null }
  routes_source: string
}

export type Run = {
  id: number
  objective_id: string
  ts: number
  author: string
  candidate_seq: number | null
  spec: Split
  members: number[]
  routes: Routes
  summary: RunSummary
  result?: LabResult
}

export type Job = {
  phase: 'running' | 'done' | 'error'
  step: string | null
  done: number
  total: number
  errors: Record<string, string>
  started_at: number
  finished_at: number | null
  run_id: number | null
  error: string | null
  author: string
  describe: string
}

export type Eligible = {
  seq: number
  id: string
  model: string
  mode: string
  score: number | null
  is_score: number | null
  eval_seconds: number | null
  rationale: string
}

export type Overview = {
  runs: Run[]
  job: Job | null
  fields: string[]
  default_fields: string[]
  modules: string[]
  eligible: Eligible[]
  default_members: number[]
  split_date: string | null
  eval_timeout_s: number | null
}

export const regimeLab = {
  overview: (oid: string) => req<Overview>(`/api/objectives/${encodeURIComponent(oid)}/regime-lab`),
  run: (rid: number) => req<Run>(`/api/regime-lab/${rid}`),
  start: (oid: string, body: { split: Split; members?: number[] | null; routes?: Routes | null }) =>
    req<{ job: Job; note?: string }>(`/api/objectives/${encodeURIComponent(oid)}/regime-lab`, {
      method: 'POST',
      body: JSON.stringify({ ...body, author: 'operator' }),
    }),
  submit: (rid: number, routes: Routes | null) =>
    req<{ ok: boolean; seq: number | null; code: string }>(`/api/regime-lab/${rid}/submit`, {
      method: 'POST',
      body: JSON.stringify({ routes }),
    }),
}

// ----------------------------------------------------------------------------------------
// Client-side preview of edited routes, from the per-(regime, member) daily P&L the run stored.
// It charges each member's own trading costs but not the extra trade at a regime change from
// one member's position to another's -- the exact number comes from re-measuring or scoring.
// ----------------------------------------------------------------------------------------
export function sharpe(xs: number[]): number | null {
  if (xs.length < 5) return null
  const m = xs.reduce((s, v) => s + v, 0) / xs.length
  const sd = Math.sqrt(xs.reduce((s, v) => s + (v - m) ** 2, 0) / (xs.length - 1))
  return sd > 0 ? (m / sd) * Math.sqrt(252) : null
}

export function segMasks(r: LabResult): Record<'is' | 'a' | 'b' | 'ho', boolean[]> {
  const split = r.split_date
  const mid = r.mid_date
  const isd = r.dates.map((d) => !split || d < split)
  return {
    is: isd,
    a: r.dates.map((d, i) => isd[i] && (!mid || d < mid)),
    b: r.dates.map((d, i) => isd[i] && !!mid && d >= mid),
    ho: isd.map((v) => !v),
  }
}

export function preview(r: LabResult, routes: Routes) {
  const n = r.dates.length
  const contrib: Record<string, number[]> = {}
  for (const m of r.members) contrib[String(m.seq)] = new Array(n).fill(0)
  const router = new Array(n).fill(0)
  for (const [lab, seq] of Object.entries(routes)) {
    if (seq == null) continue
    const cell = r.daily.cells[lab]?.[String(seq)]
    if (!cell) continue
    const c = contrib[String(seq)]
    for (let i = 0; i < n; i++) {
      c[i] += cell[i]
      router[i] += cell[i]
    }
  }
  const masks = segMasks(r)
  const anyDay = r.dates.map((_, i) => Object.values(r.daily.bars).some((b) => b[i] > 0))
  const seg = (k: 'is' | 'a' | 'b' | 'ho') => {
    const xs = router.filter((_, i) => masks[k][i] && anyDay[i])
    return { days: xs.length, sharpe: sharpe(xs), pnl: xs.reduce((s, v) => s + v, 0), hit: null } as Seg
  }
  return { router, contrib, stats: { is: seg('is'), a: seg('a'), b: seg('b'), ho: seg('ho') } as Segs }
}

export function sameRoutes(a: Routes, b: Routes): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)])
  for (const k of keys) if ((a[k] ?? null) !== (b[k] ?? null)) return false
  return true
}

/** A short form of a regime label for tight headers: "GEX:high|IntrVol:low" -> "GEX hi · IntrVol lo". */
export function shortLabel(label: string): string {
  if (!label.includes(':')) return label
  const abbr: Record<string, string> = { low: 'lo', mid: 'mid', high: 'hi' }
  return label
    .split('|')
    .map((p) => {
      const [f, b] = p.split(':')
      return `${f} ${abbr[b] ?? b}`
    })
    .join(' · ')
}
