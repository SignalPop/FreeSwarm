'use client'

// The Work page's live strip: what each agent is doing this second (pushed by app/events.py),
// and the way to talk to it mid-iteration (app/steering.py): a message it reads at its next
// tool round, "wrap up" (that round is its last), or "abort".

import { useState } from 'react'
import { Button, Panel } from '@/components/ui'
import { duration } from '@/lib/format'
import { steerApi, type LiveAgent, type Steer, type SteerKind } from '@/lib/live'

const inputCls =
  'min-w-0 flex-1 rounded-lg border border-seam bg-panel-hi px-2.5 py-1 font-mono text-[12px] text-ink outline-none focus:border-accent'

const SOURCE: Record<string, string> = { operator: 'you', mentor: 'mentor', harness: 'harness' }

function doing(a: LiveAgent, now: number): string {
  const c = a.current
  if (!c || c.status !== 'running') return 'idle'
  const p = c.pending
  if (!p) return 'between steps'
  const t = duration(now - p.since)
  return p.kind === 'chat' ? `thinking · ${t}` : `${p.name ?? 'tool'} · ${t}`
}

function AgentRow({ projectId, a, now, onSent }: { projectId: string; a: LiveAgent; now: number; onSent: () => void }) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const c = a.current
  const running = c?.status === 'running'

  async function send(kind: SteerKind) {
    if (kind === 'abort' && !window.confirm(`Abort ${a.agent}'s iteration? Nothing it has not submitted yet is kept.`)) return
    setBusy(true)
    setErr(null)
    try {
      await steerApi.send(projectId, { agent: a.agent, kind, text: kind === 'steer' ? text.trim() : '', objective_id: c?.objective?.id })
      setText('')
      setOpen(false)
      onSent()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <li className="border-b border-seam/50 py-1.5 last:border-0">
      <div className="grid grid-cols-[minmax(0,1.1fr)_minmax(0,1.4fr)_150px_auto] items-center gap-3 font-mono text-[11.5px]">
        <span className="truncate text-ink" title={a.model}>
          {running && <span className="mr-1.5 inline-block h-1.5 w-1.5 animate-dot rounded-full bg-accent" />}
          {a.agent}
        </span>
        <span className="truncate text-ink-dim" title={c?.objective?.title ?? ''}>
          {running ? (
            <>
              <span className="text-accent">{c?.mode}</span>
              {c?.objective?.title && <span className="text-ink-faint"> · {c.objective.title}</span>}
              <span className="text-ink-faint">
                {' '}
                · {c?.tool_calls ?? 0} tools
                {(c?.submissions?.length ?? 0) > 0 && ` · ${c!.submissions.map((s) => `#${s.seq} ${s.status}`).join(', ')}`}
              </span>
            </>
          ) : (
            <span className="text-ink-faint">{c ? `${c.status} ${duration(now - (c.ended_at ?? c.started_at))} ago` : '—'}</span>
          )}
        </span>
        <span className={`truncate ${running ? 'text-ink-dim' : 'text-ink-faint'}`}>
          {running && c ? `${doing(a, now)} · ${duration(now - c.started_at)}` : doing(a, now)}
        </span>
        <span className="flex items-center gap-1">
          <button
            onClick={() => setOpen((v) => !v)}
            className="rounded px-1.5 py-0.5 text-ink-dim hover:bg-panel-hi hover:text-ink"
            title="Send this agent a message it reads at its next tool round"
          >
            message
          </button>
          {running && (
            <>
              <button
                onClick={() => send('wrap_up')}
                disabled={busy}
                className="rounded px-1.5 py-0.5 text-warn hover:bg-panel-hi disabled:opacity-40"
                title="Its next tool round is its last: it submits what it has"
              >
                wrap up
              </button>
              <button
                onClick={() => send('abort')}
                disabled={busy}
                className="rounded px-1.5 py-0.5 text-bad hover:bg-panel-hi disabled:opacity-40"
                title="End the iteration at its next tool round, nothing submitted"
              >
                abort
              </button>
            </>
          )}
        </span>
      </div>
      {open && (
        <form
          className="mt-1.5 flex items-center gap-2"
          onSubmit={(ev) => {
            ev.preventDefault()
            if (text.trim()) send('steer')
          }}
        >
          <input
            autoFocus
            className={inputCls}
            placeholder={running ? 'e.g. build on #212 instead of tuning thresholds' : 'read at the start of its next iteration'}
            value={text}
            onChange={(ev) => setText(ev.target.value)}
            maxLength={4000}
          />
          <Button type="submit" tone="ghost" className="px-3 py-1 text-[12px]" disabled={busy || !text.trim()}>
            Send
          </Button>
        </form>
      )}
      {err && <div className="mt-1 text-[11.5px] text-bad">could not send: {err}</div>}
    </li>
  )
}

export default function LiveAgents({
  projectId,
  agents,
  steers,
  connected,
  onSent,
}: {
  projectId: string
  agents: LiveAgent[]
  steers: Steer[]
  connected: boolean
  onSent: () => void
}) {
  const [all, setAll] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const now = Date.now() / 1000
  const running = agents.filter((a) => a.current?.status === 'running').length

  async function sendAll() {
    if (!all.trim()) return
    setBusy(true)
    setErr(null)
    try {
      await steerApi.send(projectId, { agent: '*', kind: 'steer', text: all.trim() })
      setAll('')
      onSent()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel className="mb-3 px-4 py-3">
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <h3 className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-ink-faint">
          Agents now · {running} running
          <span
            className={`ml-2 normal-case tracking-normal ${connected ? 'text-good' : 'text-ink-faint'}`}
            title={connected ? 'Updates are pushed as they happen' : 'Live updates are reconnecting; the page polls meanwhile'}
          >
            {connected ? '● live' : '○ polling'}
          </span>
        </h3>
      </div>
      <ul>
        {agents.map((a) => (
          <AgentRow key={a.agent} projectId={projectId} a={a} now={now} onSent={onSent} />
        ))}
      </ul>
      <form
        className="mt-2 flex items-center gap-2"
        onSubmit={(ev) => {
          ev.preventDefault()
          sendAll()
        }}
      >
        <input
          className={inputCls}
          placeholder="Message every agent running on this project (read at each one's next tool round)…"
          value={all}
          onChange={(ev) => setAll(ev.target.value)}
          maxLength={4000}
        />
        <Button type="submit" tone="ghost" className="px-3 py-1 text-[12px]" disabled={busy || !all.trim()}>
          Send to all
        </Button>
      </form>
      {err && <div className="mt-1 text-[11.5px] text-bad">could not send: {err}</div>}
      {steers.length > 0 && (
        <ol className="mt-2 space-y-0.5 border-t border-seam/50 pt-2 font-mono text-[11px] text-ink-faint">
          {steers.slice(0, 6).map((s) => {
            const got = s.deliveries ?? []
            return (
              <li key={s.id} className="flex min-w-0 items-baseline gap-2">
                <span className="shrink-0">{duration(now - s.created_at)} ago</span>
                <span className="shrink-0 text-ink-dim">
                  {SOURCE[s.source] ?? s.source} → {s.agent === '*' ? 'all' : s.agent}
                </span>
                <span className={`min-w-0 truncate ${s.kind === 'abort' ? 'text-bad' : s.kind === 'wrap_up' ? 'text-warn' : 'text-ink-dim'}`} title={s.text}>
                  {s.kind === 'steer' ? s.text : s.kind.replace('_', ' ') + (s.text ? `: ${s.text}` : '')}
                </span>
                <span className="ml-auto shrink-0">
                  {s.cancelled_at
                    ? 'withdrawn'
                    : got.length
                      ? `read by ${got.map((d) => `${d.agent}${d.round ? ` (round ${d.round})` : ''}`).join(', ')}`
                      : now > s.expires_at
                        ? 'expired unread'
                        : 'waiting'}
                </span>
                {!s.cancelled_at && now < s.expires_at && (s.agent === '*' || !got.length) && (
                  <button
                    className="shrink-0 text-ink-dim hover:text-ink"
                    onClick={() => steerApi.cancel(s.id).then(onSent, () => undefined)}
                  >
                    withdraw
                  </button>
                )}
              </li>
            )
          })}
        </ol>
      )}
    </Panel>
  )
}
