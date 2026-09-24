'use client'

import { useCallback, useEffect, useState } from 'react'
import { external, type EscalationStatus, usd } from '@/lib/external'
import { Button, Pill } from '@/components/ui'

/**
 * When the search stops improving, stronger models are asked for new directions, one step up
 * the ladder at a time (app/escalation.py). This tab shows how long the objective has gone
 * without a new best, who is asked next, and every idea handed to the agents since the last
 * improvement -- with a button to ask now instead of waiting for the thresholds.
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

  if (!st) return <div className="text-[12px] text-ink-faint">{err ?? 'Loading…'}</div>
  const c = st.config
  const next = st.next_rung != null ? st.ladder[st.next_rung] : null

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
        <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">
          Ideas since the last new best ({st.ideas.length}) — every agent reads the newest three
        </div>
        {st.ideas.length ? (
          <ul className="space-y-2">
            {st.ideas.map((i) => (
              <li key={i.id} className="rounded-lg border border-seam p-2">
                <div className="mb-1 font-mono text-[10.5px] text-ink-faint">
                  step {i.rung + 1} · {i.model} · {new Date(i.ts * 1000).toLocaleString()}
                  {i.trigger === 'operator' ? ' · asked by you' : ''}
                </div>
                <div className="whitespace-pre-wrap text-[12px] leading-snug text-ink-dim">{i.text}</div>
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-[12px] text-ink-faint">None yet — the climb resets whenever a new best is found.</div>
        )}
      </div>
    </div>
  )
}
