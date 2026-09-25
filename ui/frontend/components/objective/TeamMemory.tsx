'use client'

import { useEffect, useState } from 'react'
import { objectives, type Scoreboards } from '@/lib/objectives'

/**
 * The team's memory at a glance: which mentor ideas were tested and how they scored, which
 * forecasts have skill and whether candidates that used them did better, and the habits of the
 * last candidates (costs, wrong direction, parameter-only tweaks). In-sample numbers only --
 * the same evidence the mentor and the agents read.
 */
export default function TeamMemory({ objectiveId }: { objectiveId: string }) {
  const [sb, setSb] = useState<Scoreboards | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    const load = () =>
      objectives
        .scoreboards(objectiveId)
        .then((d) => alive && (setSb(d), setErr(null)))
        .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    void load()
    const t = setInterval(load, 15000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [objectiveId])

  if (err) return <div className="text-[12px] text-bad">✗ {err}</div>
  if (!sb) return <div className="text-[12px] text-ink-faint">loading…</div>
  const num = (v: number | null | undefined, d = 2) => (typeof v === 'number' ? v.toFixed(d) : '—')
  const h = sb.habits

  return (
    <div className="space-y-5">
      <div className="text-[11.5px] text-ink-dim">
        Mentor {sb.mentor_active ? <span className="text-good">on duty</span> : <span className="text-ink-faint">not running</span>}
        {' · '}next notes: {sb.mentor_due}
      </div>

      <section>
        <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">
          Habits · last {h.candidates} candidates
        </div>
        <div className="flex flex-wrap gap-1.5 font-mono text-[11px]">
          {h.failed_to_run > 0 && <span className="rounded border border-bad/40 px-2 py-0.5 text-bad">failed to run {h.failed_to_run}</span>}
          {Object.entries(h.results).map(([k, n]) => (
            <span key={k} className="rounded border border-seam px-2 py-0.5 text-ink-dim">{k} {n}</span>
          ))}
          {Object.entries(h.changes_vs_parent).map(([k, n]) => (
            <span key={k} className={`rounded border px-2 py-0.5 ${k === 'parameters only' ? 'border-warn/50 text-warn' : 'border-seam text-ink-dim'}`}>
              {k} {n}
            </span>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">Ideas · what testing them scored (in-sample)</div>
        {sb.ideas.length === 0 ? (
          <div className="text-[12px] text-ink-faint">No ideas yet — the mentor posts its first ones once it runs.</div>
        ) : (
          <table className="w-full font-mono text-[11px]">
            <thead>
              <tr className="text-ink-faint">
                <th className="py-1 text-left font-normal">idea</th>
                <th className="py-1 text-right font-normal">tried</th>
                <th className="py-1 text-right font-normal">best</th>
                <th className="py-1 text-right font-normal">median</th>
                <th className="py-1 pl-3 text-left font-normal">best result</th>
              </tr>
            </thead>
            <tbody>
              {sb.ideas.map((i) => (
                <tr key={i.id} className="border-t border-seam/60 align-top">
                  <td className="py-1 pr-3 text-ink-dim" title={i.idea}>
                    <span className="text-accent">[{i.id}]</span> {i.idea.split('\n')[0].replace(/^IDEA:\s*/, '').slice(0, 140)}
                    <div className="text-[10px] text-ink-faint">{i.model} · {i.minutes_ago} min ago{i.champions ? ` · ${i.champions} champion` : ''}</div>
                  </td>
                  <td className="py-1 text-right text-ink">{i.tried}{i.failed ? <span className="text-bad"> ({i.failed}✗)</span> : ''}</td>
                  <td className="py-1 text-right text-ink">{num(i.best_in_sample)}{i.best_seq ? <span className="text-ink-faint"> #{i.best_seq}</span> : ''}</td>
                  <td className="py-1 text-right text-ink-dim">{num(i.median_in_sample)}</td>
                  <td className="py-1 pl-3 text-ink-faint">{i.best_diagnosis ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section>
        <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">
          Forecasts · skill vs “no change”, and did using them help?
        </div>
        {sb.forecasts.length === 0 ? (
          <div className="text-[12px] text-ink-faint">No forecasts built yet.</div>
        ) : (
          <table className="w-full font-mono text-[11px]">
            <thead>
              <tr className="text-ink-faint">
                <th className="py-1 text-left font-normal">forecast</th>
                <th className="py-1 text-right font-normal">skill</th>
                <th className="py-1 text-right font-normal">inputs lift</th>
                <th className="py-1 text-right font-normal">used by</th>
                <th className="py-1 text-right font-normal" title="median in-sample of candidates using it, minus those using no forecast">helped</th>
              </tr>
            </thead>
            <tbody>
              {sb.forecasts.map((f) => {
                const s = Object.values(f.skill)[0] ?? {}
                const good = (v: number | null | undefined) => (typeof v === 'number' ? (v > 0 ? 'text-good' : 'text-bad') : 'text-ink-faint')
                return (
                  <tr key={f.view} className="border-t border-seam/60">
                    <td className="py-1 pr-3 text-ink-dim" title={f.requested_by ?? undefined}>
                      {f.view}
                      <div className="text-[10px] text-ink-faint">
                        {f.model} · {(f.series ?? []).join(', ')}{f.inputs.length ? ` ← ${f.inputs.join(', ')}` : ''} · h{f.horizon}
                      </div>
                    </td>
                    <td className={`py-1 text-right ${good(s.skill)}`}>{num(s.skill, 3)}</td>
                    <td className={`py-1 text-right ${good(s.lift_from_inputs)}`}>{num(s.lift_from_inputs, 3)}</td>
                    <td className="py-1 text-right text-ink">{f.used_by}</td>
                    <td className={`py-1 text-right ${good(f.helped)}`}>{num(f.helped)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
