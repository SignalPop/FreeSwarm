// External review: a frontier Claude model checks a result the local harness passed.
// Backend: ui/backend/app/review.py.

export type ReviewModel = { id: string; label: string; note: string }

export type ReviewConfig = {
  enabled: boolean
  model: string
  effort: string
  /** 'ask' — the operator confirms each demotion; 'auto' — the reviewer demotes on its own. */
  autonomy: 'ask' | 'auto'
  auto_review_champions: boolean
  /** Whether a key is stored. The key itself never leaves the backend. */
  key_set: boolean
  models: ReviewModel[]
}

export type Finding = { severity: string; title: string; detail: string }

export type Verdict = {
  trustworthy: boolean
  look_ahead_found: boolean
  look_ahead_detail: string
  disqualify: boolean
  lesson: string
  findings: Finding[]
  improvements: string[]
  summary: string
}

export type ReviewResult = {
  model: string
  seconds: number
  usage: { input_tokens: number | null; output_tokens: number | null }
  autonomy: 'ask' | 'auto'
  verdict: Verdict
  board_text: string
  demoted: { demoted: number; recrowned: { seq: number } | null; quarantined?: string[] } | null
  modules_reviewed?: string[]
  awaiting_confirmation: boolean
}

/** What the streamed review reports while it runs. */
export type ReviewProgress = {
  type: 'start' | 'progress'
  /** 'start' only: what is being reviewed. */
  model?: string
  seq?: number
  modules?: string[]
  prompt_chars?: number
  /** 'progress' only: reading the code, reasoning about it, or writing the verdict. */
  phase?: 'sending' | 'thinking' | 'writing'
  output_tokens?: number
  /** The tail of the reviewer's own reasoning summary -- what it is working on right now. */
  note?: string
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

export const review = {
  config: () => req<ReviewConfig>('/api/review/config'),
  save: (body: Partial<Omit<ReviewConfig, 'key_set' | 'models'>> & { api_key?: string }) =>
    req<ReviewConfig>('/api/review/config', { method: 'PUT', body: JSON.stringify(body) }),
  test: () => req<{ ok: boolean; model: string; reply: string }>('/api/review/test', { method: 'POST' }),
  run: (oid: string, cid: string, body: { model?: string; autonomy?: 'ask' | 'auto' } = {}) =>
    req<ReviewResult>(`/api/objectives/${e(oid)}/candidates/${e(cid)}/review`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /**
   * The same review, streamed. A review is a minute of silence on one request, so the
   * console follows it instead of freezing: `onProgress` fires as the reviewer thinks and
   * writes, and the promise resolves with the finished verdict.
   */
  runStreamed: async (
    oid: string,
    cid: string,
    onProgress: (p: ReviewProgress) => void,
    body: { model?: string; autonomy?: 'ask' | 'auto' } = {},
    signal?: AbortSignal,
  ): Promise<ReviewResult> => {
    const res = await fetch(`/api/objectives/${e(oid)}/candidates/${e(cid)}/review/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
    if (!res.ok || !res.body) {
      let detail = `${res.status} ${res.statusText}`
      try {
        const j = await res.json()
        if (j?.detail) detail = String(j.detail)
      } catch {
        /* non-JSON error body */
      }
      throw new Error(detail)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let result: ReviewResult | null = null

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop() ?? ''
      for (const event of events) {
        for (const line of event.split('\n')) {
          if (!line.startsWith('data:')) continue
          const raw = line.slice(5).trim()
          if (!raw) continue
          const msg = JSON.parse(raw)
          if (msg.type === 'error') throw new Error(String(msg.error))
          if (msg.type === 'done') result = msg.result as ReviewResult
          else onProgress(msg as ReviewProgress)
        }
      }
    }
    if (!result) throw new Error('the review ended without a verdict')
    return result
  },
}

// ---- Per-module review ------------------------------------------------------------------
// A signal module is imported by many candidates, so a defect in it has already contaminated
// every result that used it. Reviewing the module directly — rather than one candidate that
// happened to import it — is the way to retire a bad signal and everything standing on it.

export type ModuleReviewResult = {
  model: string
  seconds: number
  usage: { input_tokens: number | null; output_tokens: number | null }
  module: string
  autonomy: 'ask' | 'auto'
  verdict: Verdict
  board_text: string
  modules_reviewed: string[]
  /** Set when autonomy was 'auto' and the verdict disqualified: what was actually retired. */
  retired: { quarantined: string[]; disqualified: Record<string, number[]> } | null
  awaiting_confirmation: boolean
}

export const moduleReview = {
  /** Review one library module, streaming progress. */
  run: async (
    projectId: string,
    name: string,
    onProgress: (p: ReviewProgress) => void,
    body: { model?: string; autonomy?: 'ask' | 'auto' } = {},
    signal?: AbortSignal,
  ): Promise<ModuleReviewResult> => {
    const res = await fetch(`/api/projects/${e(projectId)}/library/${e(name)}/review/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
    if (!res.ok || !res.body) {
      let detail = `${res.status} ${res.statusText}`
      try {
        const j = await res.json()
        if (j?.detail) detail = String(j.detail)
      } catch {
        /* non-JSON error body */
      }
      throw new Error(detail)
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let result: ModuleReviewResult | null = null
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop() ?? ''
      for (const event of events) {
        for (const line of event.split('\n')) {
          if (!line.startsWith('data:')) continue
          const raw = line.slice(5).trim()
          if (!raw) continue
          const msg = JSON.parse(raw)
          if (msg.type === 'error') throw new Error(String(msg.error))
          if (msg.type === 'done') result = msg.result as ModuleReviewResult
          else onProgress(msg as ReviewProgress)
        }
      }
    }
    if (!result) throw new Error('the review ended without a verdict')
    return result
  },

  /** Retire a module and disqualify every result built on it, across the project. */
  retire: (projectId: string, name: string, reason: string, reviewer = 'operator') =>
    req<{ quarantined: string[]; disqualified: Record<string, number[]> }>(
      `/api/projects/${e(projectId)}/library/${e(name)}/retire`,
      { method: 'POST', body: JSON.stringify({ reason, reviewer }) },
    ),
}
