/** The bug list and the monitoring agent that fills it (backend: app/bugs.py, app/monitor.py). */

export type BugStatus = 'open' | 'pending' | 'closed'
export type Severity = 'critical' | 'high' | 'medium' | 'low'
export type Priority = 'P1' | 'P2' | 'P3' | 'P4'

export const STATUSES: BugStatus[] = ['open', 'pending', 'closed']
export const SEVERITIES: Severity[] = ['critical', 'high', 'medium', 'low']
export const PRIORITIES: Priority[] = ['P1', 'P2', 'P3', 'P4']

export type BugSummary = {
  id: number
  title: string
  category: string
  severity: Severity
  priority: Priority
  status: BugStatus
  source: 'monitor' | 'manual'
  project_id: string | null
  agent: string | null
  model: string | null
  objective_id: string | null
  objective_title: string | null
  tool: string | null
  occurrences: number
  first_seen: number | null
  last_seen: number | null
  created_at: number
  updated_at: number
  closed_at: number | null
  /** Who closed it: the monitor (it saw the problem stop) or the operator. */
  closed_by: 'monitor' | 'operator' | null
  triaged: number
}

export type Sighting = {
  at: number
  agent: string | null
  model: string | null
  project_id: string | null
  objective_id: string | null
  record_id: string | null
  evidence: string
}

export type Bug = BugSummary & {
  description: string
  script: string
  evidence: string
  context: Record<string, unknown>
  suggestion: string
  notes: string
  sightings: Sighting[]
}

export type BugCounts = Record<BugStatus, number>

export type MonitorConfig = {
  enabled: boolean
  llm_triage: boolean
  model: string
  stall_minutes: number
  slow_reply_s: number
  /** Close a bug once the logs show it fixed (it reopens if it comes back). */
  auto_close: boolean
  /** ... after this many later chances to recur went clean ... */
  fixed_after: number
  /** ... and at least this long without it. */
  fixed_quiet_minutes: number
}

export type Recheck = {
  verdict: 'fixed' | 'not_yet' | 'unknown' | 'closed'
  message: string
  bug: Bug
  chances?: number
  needed?: number
}

export type MonitorDoc = {
  config: MonitorConfig
  state: {
    running: boolean
    last_scan: number | null
    last_error: string | null
    last_findings: number
    last_new: number
    last_closed?: number
    scans: number
    llm_calls: number
    model: string | null
    llm_note: string | null
    /** A model write-up pass is running in the background (it never holds up a scan). */
    triage_running?: boolean
    last_triage?: number | null
  }
  tick_s: number
  models: string[]
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

export const bugsApi = {
  list: (status: BugStatus | 'all', q = '') =>
    req<{ bugs: BugSummary[]; counts: BugCounts }>(
      `/api/bugs?status=${status}${q ? `&q=${encodeURIComponent(q)}` : ''}`,
    ),
  counts: () => req<BugCounts>('/api/bugs/counts'),
  get: (id: number) => req<Bug>(`/api/bugs/${id}`),
  patch: (id: number, changes: Partial<Pick<Bug, 'status' | 'severity' | 'priority' | 'title' | 'description' | 'notes'>>) =>
    req<Bug>(`/api/bugs/${id}`, { method: 'PATCH', body: JSON.stringify(changes) }),
  create: (bug: { title: string; description?: string; severity?: Severity; priority?: Priority; script?: string; evidence?: string }) =>
    req<Bug>('/api/bugs', { method: 'POST', body: JSON.stringify(bug) }),
  remove: (id: number) => req<{ ok: boolean }>(`/api/bugs/${id}`, { method: 'DELETE' }),
  /** Ask the monitor whether this bug is fixed now; it closes it when the logs show so. */
  recheck: (id: number) => req<Recheck>(`/api/bugs/${id}/recheck`, { method: 'POST' }),
  monitor: () => req<MonitorDoc>('/api/monitor'),
  configure: (changes: Partial<MonitorConfig>) =>
    req<MonitorDoc>('/api/monitor', { method: 'PUT', body: JSON.stringify(changes) }),
  scan: () =>
    req<MonitorDoc & { findings: number; new_bugs: number; sightings: number; reopened: number; closed_fixed: number }>(
      '/api/monitor/scan',
      { method: 'POST' },
    ),
}

function when(ts: number | null | undefined): string {
  return ts ? new Date(ts * 1000).toLocaleString(undefined, { hour12: false }) : '—'
}

/** The whole bug as Markdown: paste it into an issue, a chat with a model, or a commit message. */
export function bugAsText(b: Bug): string {
  const fence = (s: string, lang = '') => '```' + lang + '\n' + s.replace(/\n$/, '') + '\n```'
  const where: [string, unknown][] = [
    ['Agent', b.agent],
    ['Model', b.model],
    ['Tool', b.tool],
    ['Objective', b.objective_title],
    ['Project', b.project_id],
  ]
  const ctx = Object.entries(b.context ?? {}).filter(
    ([k, v]) => v !== null && v !== undefined && v !== '' && !['agent', 'model', 'project_id', 'objective_title'].includes(k),
  )
  const parts = [
    `# Bug #${b.id}: ${b.title}`,
    [
      `Status: ${b.status}${b.closed_by ? ` (closed by ${b.closed_by})` : ''}`,
      `Priority: ${b.priority}`,
      `Severity: ${b.severity}`,
      `Category: ${b.category.replace('_', ' ')}`,
      `Source: ${b.source}`,
    ].join(' · '),
    `Seen ${b.occurrences}× · first ${when(b.first_seen)} · last ${when(b.last_seen)}`,
    where
      .filter(([, v]) => v)
      .map(([k, v]) => `${k}: ${v}`)
      .join('\n'),
    `## Description\n${b.description || '—'}`,
    b.suggestion && `## Suggested fix\n${b.suggestion}`,
    b.script && `## Script / inputs (latest)\n${fence(b.script, b.tool === 'run_python' || b.tool === 'submit_candidate' ? 'python' : '')}`,
    b.evidence && `## Evidence (latest)\n${fence(b.evidence)}`,
    ctx.length > 0 && `## Context\n${ctx.map(([k, v]) => `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n')}`,
    b.notes && `## Notes\n${b.notes}`,
    b.sightings.length > 0 &&
      `## Sightings (${b.sightings.length}${b.occurrences > b.sightings.length ? ` of ${b.occurrences}` : ''})\n` +
        b.sightings
          .map((s) => `- ${when(s.at)} · ${s.agent ?? s.model ?? '—'}${s.record_id ? ` · record ${s.record_id}` : ''}`)
          .join('\n'),
  ]
  return parts.filter(Boolean).join('\n\n') + '\n'
}
