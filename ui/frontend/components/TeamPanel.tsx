'use client'

import { useEffect, useMemo, useState } from 'react'
import { board, type BoardMessage } from '@/lib/board'
import { duration } from '@/lib/format'
import { Panel } from '@/components/ui'
import type { ObjectiveDetail } from '@/lib/objectives'
import TeamMessages, { type ThreadKind } from '@/components/TeamMessages'

type Collab = {
  agent: string
  mode: string
  candidate: number | null
  built_on: { seq: number; by: string | null } | null
  reused: { module: string; by: string }[]
  contributed: string[]
  messages_sent: { to: string; reply_to?: number | null }[]
  answered: { seq: number; from: string }[]
  inbox: number
  note?: string
}

const short = (m: string | null | undefined) => (m ?? '?').split('/').pop() ?? '?'

/**
 * How the agents work together, totalled from the collaboration record each one posts to
 * #team after every iteration: who builds on whose candidates, whose library modules get
 * reused, what each contributed, and whether messages between them get answered.
 */
export default function TeamPanel({
  objective,
  onCandidateChanged,
}: {
  /** the objective open in the console, so candidates named in messages can be opened */
  objective?: ObjectiveDetail | null
  onCandidateChanged?: () => void
} = {}) {
  const [msgs, setMsgs] = useState<BoardMessage[]>([])
  const [err, setErr] = useState<string | null>(null)
  // Drill-down: which model's messages, which list, and the #team window the counts came from.
  const [open, setOpen] = useState<{ agent: string; kind: ThreadKind; through: number | null } | null>(null)
  const through = msgs.length ? msgs[msgs.length - 1].seq : null
  const drill = (agent: string, kind: ThreadKind, label: string, cls = '') => (
    <button
      type="button"
      onClick={() => setOpen({ agent, kind, through })}
      className={`rounded-sm underline decoration-dotted underline-offset-2 focus-visible:outline focus-visible:outline-1 focus-visible:outline-accent ${cls || 'hover:text-ink'}`}
      title="Show these messages"
    >
      {label}
    </button>
  )

  useEffect(() => {
    let alive = true
    const load = () =>
      board
        .tail('team', 300)
        .then((d) => {
          if (!alive) return
          setMsgs(d.entries)
          setErr(null)
        })
        .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    void load()
    const t = setInterval(load, 10_000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [])

  const { agents, links } = useMemo(() => {
    const agents = new Map<
      string,
      { iterations: number; builtOnOthers: number; reusedOthers: number; contributed: Set<string>; sent: number; answered: number; unanswered: number; note: string; noteTs: number }
    >()
    const links = new Map<string, number>() // "a -> b" : times a built on / reused b's work
    for (const m of msgs) {
      const c = (m.meta as { collab?: Collab }).collab
      if (!c) continue
      const a = agents.get(c.agent) ?? {
        iterations: 0, builtOnOthers: 0, reusedOthers: 0, contributed: new Set<string>(), sent: 0, answered: 0, unanswered: 0, note: '', noteTs: 0,
      }
      a.iterations += 1
      if (c.built_on?.by && c.built_on.by !== c.agent) {
        a.builtOnOthers += 1
        links.set(`${c.agent}→${c.built_on.by}`, (links.get(`${c.agent}→${c.built_on.by}`) ?? 0) + 1)
      }
      for (const r of c.reused ?? []) {
        if (r.by && r.by !== c.agent) {
          a.reusedOthers += 1
          links.set(`${c.agent}→${r.by}`, (links.get(`${c.agent}→${r.by}`) ?? 0) + 1)
        }
      }
      for (const n of c.contributed ?? []) a.contributed.add(n)
      a.sent += (c.messages_sent ?? []).length
      a.answered += (c.answered ?? []).length
      a.unanswered += Math.max(0, (c.inbox ?? 0) - (c.answered ?? []).length)
      if (c.note && m.ts >= a.noteTs) {
        a.note = c.note
        a.noteTs = m.ts
      }
      agents.set(c.agent, a)
    }
    return { agents, links }
  }, [msgs])

  return (
    <Panel className="p-4">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-[14px] font-medium text-ink">Team collaboration</span>
        <span className="ml-auto font-mono text-[10.5px] text-ink-faint">from #team</span>
      </div>
      {err && <div className="text-[12px] text-bad">{err}</div>}
      {agents.size === 0 ? (
        <div className="text-[12px] text-ink-faint">
          After every iteration each agent records how it worked with the others — whose work it built
          on, what it reused and contributed, who it messaged. Totals appear here.
        </div>
      ) : (
        <div className="space-y-2.5">
          {[...agents.entries()].map(([name, a]) => (
            <div key={name} className="rounded-xl border border-seam bg-panel-hi/40 p-2.5">
              <div className="flex items-center gap-2 font-mono text-[11.5px]">
                <span className="truncate text-ink">{short(name)}</span>
                <span className="ml-auto shrink-0 text-ink-faint">{a.iterations} iterations</span>
              </div>
              <div className="mt-1 grid grid-cols-2 gap-x-3 font-mono text-[10.5px] text-ink-dim">
                <span>built on others: {a.builtOnOthers}</span>
                <span>reused others&apos; code: {a.reusedOthers}</span>
                <span>modules contributed: {a.contributed.size}</span>
                <span>
                  messages: {drill(name, 'sent', `${a.sent} sent`)} · {drill(name, 'answered', `${a.answered} answered`)}
                  {a.unanswered > 0 && (
                    <span className="text-warn">
                      {' · '}
                      {drill(name, 'unanswered', `${a.unanswered} unanswered`, 'hover:text-warn')}
                    </span>
                  )}
                </span>
              </div>
              {a.note && (
                <div className="mt-1.5 text-[11.5px] italic leading-snug text-ink-faint">
                  “{a.note}” <span className="not-italic">· {duration(Date.now() / 1000 - a.noteTs)} ago</span>
                </div>
              )}
            </div>
          ))}
          {links.size > 0 && (
            <div>
              <div className="mb-0.5 text-[10.5px] uppercase tracking-wide text-ink-faint">Who builds on whom</div>
              {[...links.entries()]
                .sort((x, y) => y[1] - x[1])
                .map(([k, n]) => {
                  const [from, to] = k.split('→')
                  return (
                    <div key={k} className="font-mono text-[10.5px] text-ink-dim">
                      {short(from)} → {short(to)} <span className="text-ink-faint">×{n}</span>
                    </div>
                  )
                })}
            </div>
          )}
        </div>
      )}
      {open && (
        <TeamMessages
          agent={open.agent}
          initial={open.kind}
          through={open.through}
          objective={objective}
          onDemoted={onCandidateChanged}
          onClose={() => setOpen(null)}
        />
      )}
    </Panel>
  )
}
