// Forecast Lab: test input combinations for covariate forecasters (Chronos-2) and find which
// inputs improve forecasts. Backend: ui/backend/app/tslab.py.

export type Score = {
  skill: number | null
  direction: number | null
  coverage: number
  qloss_rel: number | null
  points: number
}

export type Paired = { gain: number | null; se: number | null; significant: boolean }

export type LabOptions = {
  datasets: string[]
  dataset: string | null
  numeric: string[]
  families: Record<string, string[]>
  suggested: string[]
  models: { model: string; state: string; family: string | null; covariates: boolean }[]
  split_date: string | null
  objective: { id: string; title: string } | null
}

export type Setup = {
  project_id: string
  objective_id?: string | null
  dataset?: string | null
  target: string
  inputs: string[]
  horizon: number
  context: number
  bar?: string | null
  points: number
  model?: string | null
}

export type Band = { median: number[]; q10: number[]; q90: number[] }

export type TestResult = {
  model: string
  dataset: string
  end: string | null
  target: string
  inputs: string[]
  rows: number
  baseline: Score
  combo: Score
  lift: Paired | null
  examples: { t: string; history: number[]; actual: number[]; baseline: Band; with_inputs: Band }[]
  seconds: number
  note: string | null
}

export type Impact = {
  input: string
  impact_gain: number | null
  impact_se: number | null
  impact_significant: boolean
  impact_skill: number | null
  impact_direction: number | null
}

export type Solo = {
  input: string
  lift_gain: number | null
  lift_se: number | null
  lift_significant: boolean
  lift_skill: number | null
  lift_direction: number | null
}

export type AnalysisResults = {
  model: string
  dataset: string
  end: string | null
  target: string
  anchors: number
  baseline: Score
  all: Score & { vs_baseline_gain: number | null; vs_baseline_se: number | null; vs_baseline_significant: boolean }
  impact: Impact[]
  solo: Solo[]
  greedy_path: { inputs: string[]; skill: number | null; direction: number | null; added: string | null; gain?: number | null; se?: number | null; significant?: boolean }[]
  best: { inputs: string[]; skill: number | null; direction: number | null; coverage: number; qloss_rel: number | null; vs_baseline_gain: number | null; vs_baseline_se: number | null; vs_baseline_significant: boolean }
  runs: (Score & { inputs: string[]; kind: string })[]
}

export type Job = {
  id: string
  phase: string
  done?: number
  total?: number
  current?: string
  error?: string | null
  params?: Setup
  results?: AnalysisResults | { error?: string }
}

export type AnalysisSummary = {
  id: string
  objective_id: string | null
  created_at: number
  status: string
  params: Setup
  best: AnalysisResults['best'] | null
  baseline: Score | null
  error?: string
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

const e = encodeURIComponent

export const tslab = {
  options: (projectId: string, objectiveId?: string | null, dataset?: string | null) => {
    const p = new URLSearchParams({ project_id: projectId })
    if (objectiveId) p.set('objective_id', objectiveId)
    if (dataset) p.set('dataset', dataset)
    return req<LabOptions>(`/api/tslab/options?${p}`)
  },
  test: (s: Setup) => req<TestResult>('/api/tslab/test', { method: 'POST', body: JSON.stringify(s) }),
  analyze: (s: Setup) => req<Job>('/api/tslab/analyze', { method: 'POST', body: JSON.stringify(s) }),
  job: (id: string) => req<Job>(`/api/tslab/jobs/${e(id)}`),
  analyses: (projectId: string) =>
    req<{ analyses: AnalysisSummary[]; running: Job[] }>(`/api/tslab/analyses?project_id=${e(projectId)}`),
  buildFeature: (objectiveId: string, body: Record<string, unknown>) =>
    req<{ view: string; rows: number; skill: Record<string, unknown> }>(`/api/objectives/${e(objectiveId)}/features`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
}
