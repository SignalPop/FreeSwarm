'use client'

/** Live progress (backend: app/events.py) and steering (app/steering.py).
 *
 *  The event stream is Server-Sent Events read with fetch, not EventSource: the console's fetch
 *  carries the bearer token (lib/auth.ts), EventSource cannot send headers. Events are hints --
 *  each one makes the page re-read the real data -- so a dropped connection loses nothing; it
 *  reconnects with backoff and the page falls back to polling meanwhile. */

import { useEffect, useRef, useState } from 'react'

/** One agent as the activity event carries it (agent_activity._summary). */
export type LiveAgent = {
  agent: string
  model: string
  role: string | null
  slot: number | null
  project_id: string | null
  updated_at: number
  current?: {
    id: string
    mode: string
    status: string
    objective: { id: string; title: string | null } | null
    started_at: number
    ended_at: number | null
    pending: { kind: 'chat' | 'tool'; name?: string; model?: string; since: number } | null
    tool_calls: number
    last_tool: string | null
    tokens: { prompt: number; completion: number; chats: number } | null
    submissions: { seq: number | null; status: string | null }[]
  }
}

export type SteerKind = 'steer' | 'wrap_up' | 'abort'

export type Steer = {
  id: number
  project_id: string
  objective_id: string | null
  agent: string
  kind: SteerKind
  text: string
  source: 'operator' | 'mentor' | 'harness' | string
  created_at: number
  expires_at: number
  cancelled_at: number | null
  deliveries?: { agent: string; record_id: string | null; round: number | null; delivered_at: number }[]
}

export type LiveEvent =
  | { kind: 'hello'; at: number }
  | { kind: 'activity'; at: number; agent: LiveAgent }
  | { kind: 'candidate'; at: number; objective_id: string; candidate_id: string; seq: number | null; status: string | null }
  | { kind: 'champion'; at: number; objective_id: string; candidate_id: string }
  | { kind: 'steer'; at: number; steer: Steer }
  | { kind: 'steer_delivered'; at: number; steer_id: number; agent: string; round: number | null }

/** Subscribe to a project's live events while mounted. Returns whether the stream is up. */
export function useProjectEvents(projectId: string | null | undefined, onEvent: (e: LiveEvent) => void): boolean {
  const [connected, setConnected] = useState(false)
  const handler = useRef(onEvent)
  handler.current = onEvent

  useEffect(() => {
    if (!projectId) return
    let stopped = false
    let ctrl: AbortController | null = null
    let retry: ReturnType<typeof setTimeout> | undefined
    let backoff = 1000

    async function connect() {
      ctrl = new AbortController()
      try {
        const res = await fetch(`/api/projects/${encodeURIComponent(projectId!)}/events`, {
          signal: ctrl.signal,
          headers: { Accept: 'text/event-stream' },
        })
        if (!res.ok || !res.body) throw new Error(`${res.status}`)
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        while (!stopped) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const frames = buffer.split('\n\n')
          buffer = frames.pop() ?? ''
          for (const frame of frames) {
            let kind = 'message'
            let data = ''
            for (const line of frame.split('\n')) {
              if (line.startsWith('event:')) kind = line.slice(6).trim()
              else if (line.startsWith('data:')) data += line.slice(5).trim()
            }
            if (!data) continue // a keepalive comment
            try {
              const parsed = JSON.parse(data)
              if (kind === 'hello') {
                setConnected(true)
                backoff = 1000
              }
              handler.current({ ...parsed, kind } as LiveEvent)
            } catch {
              /* a frame we do not model */
            }
          }
        }
      } catch {
        /* aborted, or the control plane is restarting */
      }
      setConnected(false)
      if (!stopped) {
        retry = setTimeout(connect, backoff)
        backoff = Math.min(backoff * 2, 30_000)
      }
    }

    connect()
    return () => {
      stopped = true
      ctrl?.abort()
      if (retry) clearTimeout(retry)
      setConnected(false)
    }
  }, [projectId])

  return connected
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

const e = encodeURIComponent

export const steerApi = {
  list: (projectId: string, limit = 30) => req<{ now: number; steers: Steer[] }>(`/api/projects/${e(projectId)}/steer?limit=${limit}`),
  /** `agent`: an agent's board name, or '*' for every agent on the project. */
  send: (projectId: string, body: { agent: string; kind: SteerKind; text?: string; objective_id?: string | null }) =>
    req<Steer>(`/api/projects/${e(projectId)}/steer`, { method: 'POST', body: JSON.stringify({ text: '', ...body }) }),
  cancel: (id: number) => req<Steer>(`/api/steer/${id}`, { method: 'DELETE' }),
}

export const agentsApi = {
  list: (projectId: string) => req<{ now: number; agents: LiveAgent[] }>(`/api/agents/activity?project_id=${e(projectId)}`),
}
