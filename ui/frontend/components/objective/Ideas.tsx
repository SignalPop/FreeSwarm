'use client'

import { useCallback, useEffect, useState } from 'react'
import { external, type EscalationStatus, type Idea, usd } from '@/lib/external'
import { objectives } from '@/lib/objectives'
import { Button, Pill } from '@/components/ui'
import { DangerLink, RowDelete } from './Prune'

/** Where an idea came from, as the list shows it. */
function origin(i: Idea): string {
  if (i.trigger === 'scheduled') return 'scheduled'
  if (i.trigger === 'mentor') return 'mentor'
  return `${i.trigger === 'operator' ? 'asked by you' : 'stuck'} · step ${i.rung + 1}`
}

/**
 * Every idea handed to the agents (app/escalation.py). When the search stops improving,
 * stronger models are asked for new directions, one step up the ladder at a time; besides
 * that, the first rung is asked on a schedule and the mentor posts directions, stuck or not.
 * This tab shows how long the objective has gone without a new best, who is asked next, the
 * ideas since the last improvement and the regular ideas still reaching agents -- with a
 * button to ask now instead of waiting for the thresholds.
 */
export default function IdeasTab({ objectiveId }: { objectiveId: string }) {
  const [st, setSt] = useState<EscalationStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setSt(await external.escalation(objectiveId))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [objectiveId])
  useEffect(() => {
    void load()
    const t = setInterval(load, 15_000)
    return () => clearInterval(t)
  }, [load])

  async function askNow() {
    setBusy(true)
    setErr(null)
    try {
      await external.escalateNow(objectiveId)
      await load()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  /** Delete one idea (its ×) or every idea on record for the objective ("Delete all…"). */
  async function drop(ids: number[] | 'all', question: string) {
    if (!window.confirm(question)) return
    setErr(null)
    try {
      await objectives.deleteRows(objectiveId, 'ideas', ids === 'all' ? { all: true } : { ids })
      await load()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  if (!st) return <div className="text-[12px] text-ink-faint">{err ?? 'Loading…'}</div>
  const c = st.config
  const next = st.next_rung != null ? st.ladder[st.next_rung] : null
  const sched = st.scheduled
  const regular = st.regular_ideas ?? []

  const item = (i: Idea) => (
    <li key={i.id} className="group rounded-lg border border-seam p-2">
      <div className="mb-1 flex items-start font-mono text-[10.5px] text-ink-faint">
        <span>
          <span className={i.trigger === 'scheduled' || i.trigger === 'mentor' ? 'text-accent' : ''}>{origin(i)}</span>
          {' '}· {i.model} · {new Date(i.ts * 1000).toLocaleString()}
          {i.tried != null ? ` · tested by ${i.tried} candidate${i.tried === 1 ? '' : 's'}` : ''}
        </span>
        <span className="ml-auto">
          <RowDelete
            title="Delete this idea"
            onClick={() =>
              drop([i.id], `Delete this idea from ${i.model}?\n\nAgents stop reading it from their next iteration.`)
            }
          />
        </span>
      </div>
      <div className="whitespace-pre-wrap text-[12px] leading-snug text-ink-dim">{i.text}</div>
    </li>
  )

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-[12px] text-ink-dim">
        <Pill tone={st.stuck ? 'warn' : 'good'}>{st.stuck ? 'stuck' : 'improving'}</Pill>
        <span>
          {st.candidates_since_improvement} candidates and {Math.round(st.minutes_since_improvement)} min since the last new best
          {c.enabled
            ? ` · stuck after ${c.stuck_candidates} candidates and ${c.stuck_minutes} min`
            : ' · automatic asking is off (Settings → External models)'}
        </span>
        <span className="ml-auto">
          <Button tone="ghost" disabled={busy || !next} onClick={askNow}>
            {busy ? `Asking ${next?.model ?? ''}…` : 'Ask for ideas now'}
          </Button>
        </span>
      </div>
      {sched && (
        <div className="text-[12px] text-ink-faint">
          {sched.enabled
            ? `New ideas on a schedule: ${sched.model ?? 'no model'} is asked every ${c.scheduled_candidates} candidates or ` +
              `${c.scheduled_minutes} min, whichever comes first — ${sched.candidates_since} candidates and ` +
              `${Math.round(sched.minutes_since)} min since the last ask${sched.due ? ' (due at the next check)' : ''}.`
            : 'New ideas on a schedule are off (Settings → External models).'}
        </div>
      )}
      {busy && <div className="text-[12px] text-ink-dim">Waiting for {next?.model} — a strong model can take a few minutes to think.</div>}
      {err && <div className="text-[12px] text-bad">✗ {err}</div>}

      <div>
        <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">Ladder — one step per stuck period</div>
        {st.ladder.length ? (
          <div className="flex flex-wrap items-center gap-1.5">
            {st.ladder.map((m, i) => (
              <span key={m.model} className="flex items-center gap-1.5">
                {i > 0 && <span className="text-ink-faint">→</span>}
                <span title={m.why}
                  className={`rounded-md border px-2 py-0.5 font-mono text-[11px] ${
                    i === st.next_rung ? 'border-accent/60 bg-accent/10 text-accent' : i < (st.next_rung ?? 0) ? 'border-seam text-ink-faint line-through' : 'border-seam text-ink-dim'}`}>
                  {i + 1}. {m.model}
                  <span className="ml-1 text-ink-faint">
                    {m.aa != null ? `AA ${m.aa}` : ''} {m.price_blended != null ? `${usd(m.price_blended)}/M` : 'free'}
                  </span>
                </span>
              </span>
            ))}
          </div>
        ) : (
          <div className="text-[12px] text-ink-faint">
            No model to ask. Load a rated model, or enable an external one and tick it for this project.
          </div>
        )}
      </div>

      <div>
        <div className="mb-1 flex items-center gap-2">
          <span className="font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">
            Ideas since the last new best ({st.ideas.length}) — every agent reads the newest three
          </span>
          {st.ideas.length > 0 && (
            <span className="ml-auto">
              <DangerLink
                onClick={() =>
                  drop(
                    'all',
                    `Delete every idea on record for this objective (${st.ideas.length} shown here, plus any from before the last new best)?\n\n` +
                      'Agents stop reading them from their next iteration. The ladder steps back down to the first rung, so if the ' +
                      'search is still stuck it is asked again soon. This cannot be undone.',
                  )
                }
              >
                Delete all…
              </DangerLink>
            </span>
          )}
        </div>
        {st.ideas.length ? (
          <ul className="space-y-2">{st.ideas.map(item)}</ul>
        ) : (
          <div className="text-[12px] text-ink-faint">None yet — the climb resets whenever a new best is found.</div>
        )}
      </div>

      <div>
        <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">
          Regular ideas — scheduled and mentor, last {st.fresh_hours ?? 6} h ({regular.length})
        </div>
        {regular.length ? (
          <ul className="space-y-2">{regular.map(item)}</ul>
        ) : (
          <div className="text-[12px] text-ink-faint">
            None in the last {st.fresh_hours ?? 6} hours — they come on the schedule above and from the mentor, stuck or not.
          </div>
        )}
      </div>
    </div>
  )
}
