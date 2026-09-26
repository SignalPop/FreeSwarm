'use client'

import { useEffect, useState } from 'react'
import { insight, num, signed, VERDICT_TONE, type FeatureReport } from '@/lib/insight'

/**
 * The forecasts a candidate actually loaded (recorded by the harness as metrics.features_used):
 * for each, what was sent to the model, its in-sample skill, the lift its inputs gave, and --
 * when this candidate improved a parent that did not use it -- how much the score moved.
 */
export default function CandidateForecasts({
  objectiveId,
  seq,
  metrics,
}: {
  objectiveId: string
  seq: number
  metrics: object | null | undefined
}) {
  const used = (metrics as { features_used?: string[] } | null | undefined)?.features_used
  const [feats, setFeats] = useState<FeatureReport[] | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    if (!used?.length) return
    insight
      .forecastReport(objectiveId)
      .then((d) => setFeats(d.features))
      .catch((x) => setErr(x instanceof Error ? x.message : String(x)))
  }, [objectiveId, used?.length])

  if (!used) return null
  return (
    <div>
      <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-faint">Forecasts used</div>
      {used.length === 0 ? (
        <div className="text-[12px] text-ink-faint">This candidate loaded no forecast feature.</div>
      ) : err ? (
        <div className="text-[12px] text-bad">✗ {err}</div>
      ) : !feats ? (
        <div className="text-[12px] text-ink-faint">loading…</div>
      ) : (
        <div className="space-y-1.5">
          {used.map((v) => {
            const f = feats.find((x) => x.view === v)
            if (!f) {
              return (
                <div key={v} className="font-mono text-[11px] text-ink-faint">
                  {v} — no longer on record
                </div>
              )
            }
            const i = f.inputs_sent
            const mine = f.pairs.find((p) => p.seq === seq)
            return (
              <div key={v} className="rounded-lg border border-seam px-3 py-2 font-mono text-[11px]">
                <div className="flex flex-wrap items-baseline gap-2">
                  <span className="text-ink">{v}</span>
                  <span className="text-ink-faint">
                    {i.model} · target {(i.target ?? []).join(', ')}
                    {i.covariates.length ? ' ← ' : ''}
                    <span className="text-accent">{i.covariates.join(', ')}</span>
                    {i.calendar ? ' + calendar' : ''} · h{i.horizon} every {i.every}
                    {i.bar ? ` · ${i.bar}` : ''} · ctx {i.context}
                  </span>
                </div>
                {Object.entries(f.skill).map(([series, s]) => (
                  <div key={series} className="mt-0.5 text-ink-dim">
                    {series}: skill <span className={(s.skill ?? 0) > 0 ? 'text-good' : 'text-bad'}>{num(s.skill)}</span>, direction{' '}
                    {num(s.direction, 3)}
                    {s.without_inputs && (
                      <>
                        {' '}
                        · without inputs {num(s.without_inputs.skill)} → inputs lift{' '}
                        <span className={s.lift?.significant ? ((s.lift.gain ?? 0) > 0 ? 'text-good' : 'text-bad') : 'text-ink-faint'}>
                          {signed(s.lift?.gain ?? null, 4)} ± {num(2 * (s.lift?.se ?? 0), 4)}
                          {s.lift?.significant ? '' : ' (noise)'}
                        </span>
                      </>
                    )}
                  </div>
                ))}
                <div className="mt-0.5 text-ink-faint">
                  {mine ? (
                    <>
                      added to parent #{mine.parent_seq}:{' '}
                      <span className={mine.delta > 0 ? 'text-good' : 'text-bad'}>{signed(mine.delta, 3)}</span> in-sample ·{' '}
                    </>
                  ) : null}
                  across the team: <span className={VERDICT_TONE[f.verdict]}>{f.verdict}</span> ({f.basis}, n={f.n}
                  {f.effect !== null ? `, ${signed(f.effect, 3)}` : ''}) · used by {f.used_by}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
