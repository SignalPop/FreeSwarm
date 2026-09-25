// The project code library: reusable modules the swarm writes and reviews, plus the
// objective's forecast features. Backend: ui/backend/app/library.py, objectives.py.

export type ModuleKind = 'regime' | 'signal' | 'risk' | 'util'
export type Verdict = 'works' | 'broken' | 'note'

export type Evidence = {
  uses: number
  ok: number
  errors: number
  lookahead_fails: number
  champions: number
  best_in_sample: number | null
  best_holdout: number | null
  recent?: {
    id: string
    seq: number
    objective_id: string
    status: string
    score: number | null
    is_score: number | null
    lookahead: string
    audit: string
    champion_at: number | null
    version: number
  }[]
}

export type LibModule = {
  project_id: string
  name: string
  description: string
  kind: ModuleKind
  version: number
  // 'quarantined': a result built on this module was disqualified. It stays readable, and
  // `warning` says why, but nothing new should be built on it.
  status: 'active' | 'retired' | 'quarantined'
  warning: string
  author: string | null
  created_at: number
  updated_at: number
  test_ok: number | null
  evidence: Evidence
  comments: Partial<Record<Verdict, number>>
}

export type SignalCell = {
  bars: number
  sharpe?: number | null
  sharpe_gross?: number | null
  mean_bps?: number
  trades_per_day?: number
  hit_rate?: number | null
  active?: number
}

export type RegimeMap = {
  id: number
  regime_version: number
  ts: number
  author: string | null
  result: {
    regimes: Record<string, { share: number; signals: Record<string, SignalCell> }>
    errors: Record<string, string>
    bars: number
    from: string
    to: string
    in_sample_only: boolean
  }
}

export type LibComment = {
  id: number
  version: number | null
  ts: number
  author: string
  verdict: Verdict
  text: string
  candidate_id: string | null
}

export type LibModuleFull = Omit<LibModule, 'comments'> & {
  shown_version: number
  code: string
  test_output: string
  version_note: string
  versions: { version: number; author: string | null; ts: number; note: string; test_ok: number | null }[]
  comments: LibComment[]
  regime_map: RegimeMap | null
}

export type Feature = {
  view: string
  file: string
  params: { dataset: string; series?: string[]; column?: string; horizon: number; every: number; context: number; model: string }
  rows: number
  seconds: number
  created_at: number
  columns: Record<string, string>
  skill?: Record<string, { anchors_scored: number; skill_vs_no_change?: number; direction_accuracy?: number | null; band_coverage_10_90?: number }>
}

export type LibDeleteResult = {
  deleted: string[]
  missing: string[]
  /** Per deleted module: candidates whose code reaches it (directly or via another module). */
  importers: Record<string, { candidates: number; ranked: number; ranked_seqs: number[] }>
  /** Per deleted module: modules left in the library that import it (and now break). */
  dependents: Record<string, string[]>
  init_reset?: boolean
  warnings: string[]
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

const e = encodeURIComponent

export const library = {
  list: (projectId: string) => req<{ modules: LibModule[] }>(`/api/projects/${e(projectId)}/library`),
  get: (projectId: string, name: string, version?: number) =>
    req<LibModuleFull>(`/api/projects/${e(projectId)}/library/${e(name)}${version ? `?version=${version}` : ''}`),
  comment: (projectId: string, name: string, verdict: Verdict, text: string) =>
    req<{ id: number }>(`/api/projects/${e(projectId)}/library/${e(name)}/comments`, {
      method: 'POST',
      body: JSON.stringify({ verdict, text, author: 'operator' }),
    }),
  patch: (projectId: string, name: string, body: { status?: 'active' | 'retired'; restore_version?: number; description?: string }) =>
    req<LibModuleFull>(`/api/projects/${e(projectId)}/library/${e(name)}`, { method: 'PATCH', body: JSON.stringify(body) }),
  /** Delete modules outright (every version, comment, usage link, regime map). */
  remove: (projectId: string, names: string[]) =>
    req<LibDeleteResult>(`/api/projects/${e(projectId)}/library/delete`, {
      method: 'POST',
      body: JSON.stringify({ names }),
    }),
  features: (objectiveId: string) => req<{ features: Feature[] }>(`/api/objectives/${e(objectiveId)}/features`),
  buildFeature: (objectiveId: string, body: { columns: string[]; horizon: number; every: number; name?: string }) =>
    req<Feature & { adjusted?: string; cached?: boolean; skill_note?: string }>(
      `/api/objectives/${e(objectiveId)}/features`,
      { method: 'POST', body: JSON.stringify(body) },
    ),
}
