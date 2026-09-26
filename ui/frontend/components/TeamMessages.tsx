'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { board, type TeamThreads, type ThreadMessage } from '@/lib/board'
import { clockTime, duration } from '@/lib/format'
import type { ObjectiveDetail } from '@/lib/objectives'
import CandidateView from '@/components/objective/CandidateView'
import { Pill } from '@/components/ui'

export type ThreadKind = 'unanswered' | 'answered' | 'sent'

const short = (m: string | null | undefined) => (m ?? '?').split('/').pop() ?? '?'

const TABS: { key: ThreadKind; label: string }[] = [
  { key: 'unanswered', label: 'unanswered' },
  { key: 'answered', label: 'answered' },
  { key: 'sent', label: 'sent' },
]

function explain(kind: ThreadKind, who: string): string {
  if (kind === 'unanswered')
    return (
      `Messages teammates addressed to ${who} (sent to it, or @${who} in the text) that it read at the start of an ` +
      `iteration and did not reply to in that iteration. Not ${who}'s own questions — those are under “sent”.`
    )
  if (kind === 'answered')
    return `Messages teammates addressed to ${who} that it replied to (reply_to = the message number) in the iteration it read them. The reply is shown under each.`
  return `What ${who} posted with team_post during its iterations — plans to everyone and direct messages — with any replies teammates sent back.`
}

/**
 * The messages behind one model's Team-panel counts, opened from "N unanswered" / "N answered"
 * / "N sent". Same window and arithmetic as the panel (backend: app/team_threads.py), so the
 * list lengths match the numbers that were clicked.
 */
export default function TeamMessages({
  agent,
  initial,
  through,
  objective,
  onClose,
  onDemoted,
}: {
  agent: string
  initial: ThreadKind
  /** newest #team seq the panel counted: the drill-down uses the same window */
  through: number | null
  /** the objective open in the console: its candidates can be opened from here */
  objective?: ObjectiveDetail | null
  onClose: () => void
  onDemoted?: () => void
}) {
  const router = useRouter()
  const [kind, setKind] = useState<ThreadKind>(initial)
  const [doc, setDoc] = useState<TeamThreads | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [openCand, setOpenCand] = useState<string | null>(null)
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    let alive = true
    board
      .teamThreads(agent, through)
      .then((d) => {
        if (!alive) return
        setDoc(d)
        setErr(null)
      })
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [agent, through])

  // Escape closes the candidate first (it handles its own key), then this drawer.
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose
  const candRef = useRef(openCand)
  candRef.current = openCand
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !candRef.current) onCloseRef.current()
    }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      opener?.focus?.()
    }
  }, [])

  const who = short(agent)
  const list = doc ? doc[kind] : []
  const count = doc ? doc.counts[kind] : 0
  const missing = doc ? doc.unlocated[kind] : 0
  const now = Date.now() / 1000

  const openable = (oid: string | null | undefined) => !!objective && !!oid && oid === objective.id
  const cand = (oid: string | null | undefined, cid: string | null | undefined, seq: number | null | undefined, key: string) => {
    if (!cid) return null
    const label = `candidate #${seq ?? '?'}`
    return openable(oid) ? (
      <button
        key={key}
        onClick={() => setOpenCand(cid)}
        className="rounded-md border border-accent/40 bg-accent/10 px-1.5 py-0.5 font-mono text-[10.5px] text-accent hover:border-accent"
        title="Open this candidate"
      >
        {label}
      </button>
    ) : (
      <span
        key={key}
        className="rounded-md border border-seam px-1.5 py-0.5 font-mono text-[10.5px] text-ink-faint"
        title={oid ? `objective ${oid} — select it in the objective panel to open its candidates` : undefined}
      >
        {label}
        {oid && objective && oid !== objective.id ? ' · other objective' : ''}
      </span>
    )
  }

  const card = (m: ThreadMessage, depth = 0, badge?: string) => (
    <div
      key={`${badge ?? 'm'}-${m.seq ?? m.text.slice(0, 40)}-${m.iteration?.record_seq ?? ''}`}
      className={depth ? 'mt-2 border-l-2 border-accent/30 pl-3' : 'rounded-xl border border-seam bg-panel-hi/40 p-3'}
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[10.5px] text-ink-faint">
        {badge && <span className="uppercase tracking-wider text-accent">{badge}</span>}
        <span className="text-ink">{short(m.from)}</span>
        <span>→ {m.to ? short(m.to) : 'everyone'}</span>
        {m.channel && <span>#{m.channel}</span>}
        {m.ts != null && (
          <span title={new Date(m.ts * 1000).toLocaleString()}>
            {clockTime(m.ts)} · {duration(now - m.ts)} ago
          </span>
        )}
        {m.seq != null && <span className="ml-auto">msg #{m.seq}</span>}
        {!m.located && <span className="text-warn">not on the board any more — the record&apos;s copy</span>}
      </div>
      <pre className="mt-1.5 whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed text-ink-dim">
        {m.text || (m.located ? '(empty)' : '(text not kept)')}
      </pre>
      {(m.refs.length > 0 || (depth === 0 && m.iteration)) && (
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          {m.refs.length > 0 && <span className="font-mono text-[10px] text-ink-faint">refers to</span>}
          {m.refs.map((r) => cand(r.objective_id, r.candidate_id, r.seq, `ref-${r.candidate_id}`))}
          {depth === 0 && m.iteration && (
            <>
              <span className="ml-2 font-mono text-[10px] text-ink-faint">
                counted in {short(agent)}&apos;s {m.iteration.mode ?? ''} iteration
                {m.iteration.ts != null ? ` at ${clockTime(m.iteration.ts)}` : ''}
                {m.iteration.candidate_id ? ' →' : ''}
              </span>
              {cand(m.iteration.objective_id, m.iteration.candidate_id, m.iteration.candidate, 'iter')}
            </>
          )}
        </div>
      )}
      {m.reply && card(m.reply, depth + 1, 'reply')}
      {kind === 'unanswered' && depth === 0 && (
        m.later_reply ? card(m.later_reply, depth + 1, 'replied later') : (
          <div className="mt-1.5 font-mono text-[10.5px] text-warn">no reply from {who} on the board</div>
        )
      )}
      {(m.replies ?? []).map((r) => card(r, depth + 1, 'reply'))}
    </div>
  )

  return (
    <>
      <div className="fixed inset-0 z-50 flex justify-end bg-black/60" onClick={onClose}>
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`${who}'s team messages`}
          className="flex h-full w-full max-w-[760px] flex-col overflow-hidden border-l border-seam bg-panel shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-start gap-3 border-b border-seam px-5 py-4">
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="truncate text-[15px] font-medium text-ink">{agent}</span>
                <Pill tone="neutral">team messages</Pill>
              </div>
              <div className="mt-1.5 font-mono text-[10.5px] text-ink-faint">
                from the collaboration records in #team
                {doc && ` · ${doc.counts.iterations} iterations`}
                {doc?.through ? ` · through #${doc.through}` : ''}
              </div>
            </div>
            <button
              ref={closeRef}
              onClick={onClose}
              className="shrink-0 rounded-md border border-seam px-2.5 py-1 font-mono text-[11px] text-ink-dim hover:border-accent/50 hover:text-ink"
            >
              close
            </button>
          </div>

          <div className="flex-1 space-y-3 overflow-y-auto px-5 py-4">
            <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Which messages">
              {TABS.map((t) => (
                <button
                  key={t.key}
                  role="tab"
                  aria-selected={t.key === kind}
                  onClick={() => setKind(t.key)}
                  className={`rounded-md border px-2.5 py-1 font-mono text-[11px] ${
                    t.key === kind ? 'border-accent/50 bg-accent/10 text-accent' : 'border-seam text-ink-dim hover:text-ink'
                  }`}
                >
                  {doc ? `${doc.counts[t.key]} ` : ''}
                  {t.label}
                </button>
              ))}
            </div>

            <div className="text-[12px] leading-relaxed text-ink-dim">{explain(kind, who)}</div>

            {err && <div className="rounded-lg border border-bad/35 bg-bad/10 p-3 text-[12px] text-bad">{err}</div>}
            {!doc && !err && <div className="text-[12px] text-ink-faint">Loading…</div>}

            {doc && (
              <div className="font-mono text-[10.5px] text-ink-faint">
                {list.length === count ? `${count} ${kind}` : `${list.length} of ${count} ${kind} shown`} · newest first
                {missing > 0 &&
                  (list.length === count
                    ? ` · ${missing} no longer on the board (shown from the record's copy)`
                    : ` · ${count - list.length} could not be located on the board (records from an older runner keep only a count)`)}
              </div>
            )}

            {doc && list.length === 0 && (
              <div className="rounded-lg border border-seam bg-panel-hi p-3 text-[12px] text-ink-faint">Nothing here.</div>
            )}
            {list.map((m) => card(m))}
          </div>
        </div>
      </div>

      {openCand && objective && (
        <CandidateView
          objective={objective}
          candidateId={openCand}
          onClose={() => setOpenCand(null)}
          onDemoted={onDemoted}
          onOpenChat={() => {
            setOpenCand(null)
            onClose()
            router.push('/chat')
          }}
        />
      )}
    </>
  )
}
