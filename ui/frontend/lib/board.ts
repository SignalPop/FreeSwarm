// Client for the agent message board (separate FastAPI service, proxied at /mb).

export type Agent = {
  id: string
  name: string
  role: string
  model: string
  capabilities: string[]
  registered_at: number
  last_seen: number
  status: 'idle' | 'working' | 'blocked' | 'done'
  online: boolean
}

export type BoardMessage = {
  seq: number
  channel: string
  author: string
  author_id: string | null
  kind: 'chat' | 'directive' | 'result' | 'error' | 'thought' | 'system'
  content: string
  meta: Record<string, unknown>
  reply_to: number | null
  ts: number
}

export type Task = {
  id: string
  title: string
  description: string
  tier: 'auto' | 'small' | 'mid' | 'hard'
  status: 'open' | 'claimed' | 'done' | 'failed'
  parent_id: string | null
  claimed_by: string | null
  lease_until: number | null
  result: string | null
  created_at: number
  updated_at: number
  attempts: number
  /** Board session the task was queued in -- where its message thread lives. */
  session_id?: string | null
}

export type Session = {
  id: string
  title: string
  note: string
  started_at: number
  ended_at: number | null
  active: boolean
  message_count: number
  task_count: number
  tasks_done: number
}

/** One message in the Team panel's drill-down (backend: app/team_threads.py). */
export type ThreadMessage = {
  seq: number | null
  /** false: the record names it but the board no longer holds it -- text is the record's copy */
  located: boolean
  from: string | null
  /** which of the author's agents posted it ("Model #2"), from meta.agent */
  agent: string | null
  to: string | null
  channel: string | null
  ts: number | null
  text: string
  objective_id: string | null
  candidate_id: string | null
  refs: { objective_id: string | null; candidate_id: string; seq: number | null }[]
  reply_to: number | null
  /** the iteration whose collaboration record counted it: the one that answered it, else the
   *  newest one that read it */
  iteration: {
    record_seq: number
    ts: number | null
    candidate: number | null
    candidate_id: string | null
    objective_id: string | null
    mode: string | null
    agent: string | null
  }
  /** inbox messages: how many of the model's iterations read it */
  read_in?: number
  /** inbox messages: the first answer any agent of the model posted (reply_to / meta.answers) */
  reply?: ThreadMessage | null
  /** expired: why the runner will never answer it */
  expired?: { reason: 'ack' | 'depth' | 'age' | 'reread'; text: string }
  /** sent: teammates' replies to it */
  replies?: ThreadMessage[]
}

export type TeamCounts = { iterations: number; sent: number; answered: number; unanswered: number; expired: number }

export type TeamThreads = {
  agent: string
  counts: TeamCounts
  sent: ThreadMessage[]
  answered: ThreadMessage[]
  unanswered: ThreadMessage[]
  expired: ThreadMessage[]
  /** counted messages not found on the board: shown from the record's own copy where it
   *  keeps one (sent, answered), otherwise missing from the list (old runners' records) */
  unlocated: { sent: number; answered: number; unanswered: number; expired: number }
  /** the runner's answering rules the classification used */
  rules?: { answer_window_s: number; reread_window_s: number; max_depth: number; expired_why: Record<string, string> }
  through: number | null
}

export type BoardSummary = {
  agents: Agent[]
  task_counts: Record<string, number>
  message_total: number
  session: { id: string; title: string; started_at: number } | null
  ts: number
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
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* non-JSON body */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const board = {
  summary: () => req<BoardSummary>('/mb/summary'),
  agents: () => req<{ agents: Agent[] }>('/mb/agents'),

  messages: (since: number, channel?: string, wait = 0, sessionId?: string) => {
    const params = new URLSearchParams({ since: String(since), wait: String(wait) })
    if (channel) params.set('channel', channel)
    if (sessionId) params.set('session_id', sessionId)
    return req<{ entries: BoardMessage[]; next_cursor: number }>(`/mb/messages?${params}`)
  },

  /** The last `n` messages (optionally of one channel), oldest first. */
  tail: (channel: string | undefined, n: number) => {
    const params = new URLSearchParams({ tail: String(n) })
    if (channel) params.set('channel', channel)
    return req<{ entries: BoardMessage[]; next_cursor: number }>(`/mb/messages?${params}`)
  },

  /** The messages behind one model's Team-panel counts (sent / answered / unanswered), over
   *  the same #team window the panel totalled -- `through` is the newest #team seq it saw. */
  teamThreads: (agent: string, through?: number | null) => {
    const params = new URLSearchParams({ agent })
    if (through) params.set('through', String(through))
    return req<TeamThreads>(`/mb/team/threads?${params}`)
  },

  /** Every model's Team-panel message counts over the same #team window (backend: app/team_threads.py). */
  teamCounts: (through?: number | null) => {
    const params = new URLSearchParams()
    if (through) params.set('through', String(through))
    return req<{ agents: Record<string, TeamCounts>; through: number | null }>(`/mb/team/counts?${params}`)
  },

  sessions: () => req<{ sessions: Session[] }>('/mb/sessions'),

  session: (id: string) =>
    req<{ session: Session; messages: BoardMessage[]; tasks: Task[] }>(`/mb/sessions/${id}`),

  newSession: (title: string, note = '') =>
    req<{ session_id: string; title: string }>('/mb/sessions', {
      method: 'POST',
      body: JSON.stringify({ title, note }),
    }),

  post: (msg: {
    channel?: string
    author: string
    kind?: BoardMessage['kind']
    content: string
    meta?: Record<string, unknown>
  }) => req<{ seq: number }>('/mb/messages', { method: 'POST', body: JSON.stringify(msg) }),

  tasks: (status?: string) =>
    req<{ tasks: Task[] }>(`/mb/tasks${status ? `?status=${status}` : ''}`),

  createTask: (t: { title: string; description?: string; tier?: Task['tier']; parent_id?: string }) =>
    req<{ task_id: string }>('/mb/tasks', { method: 'POST', body: JSON.stringify(t) }),

  /**
   * Re-queue a failed task as a NEW task linked to it (parent_id), with the original request
   * intact and the previous failure attached.
   *
   * Typing "try again" into the composer queued a task literally titled "try again": the agent
   * had no idea what "this" was, explored the data aimlessly and asked what to do. A retry has
   * to carry the original request -- and telling the agent WHY the last attempt failed (context
   * overflow, a bad query, a timeout) lets it avoid the same wall instead of hitting it again.
   */
  retryTask: (t: Task) => {
    const previous = (t.result || '(no detail recorded)').trim().slice(0, 1500)
    const note =
      `\n\n---\nThis is attempt ${t.attempts + 1}: a previous attempt at this same task failed with:\n` +
      `${previous}\n\nAvoid repeating that failure -- e.g. if the context filled up, ` +
      `query only the columns and rows you need and aggregate in SQL.`
    return req<{ task_id: string }>('/mb/tasks', {
      method: 'POST',
      body: JSON.stringify({
        title: t.title,
        description: (t.description || '') + note,
        tier: t.tier,
        parent_id: t.id,
      }),
    })
  },

  channels: () =>
    req<{ channels: { channel: string; n: number; last_ts: number }[] }>('/mb/channels'),
}
