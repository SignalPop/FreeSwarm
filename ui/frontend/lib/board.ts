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
