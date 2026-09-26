'use client'

import { Fragment, useCallback, useEffect, useState } from 'react'
import { insight, num, signed, VERDICT_TONE, type ComboGroup, type ExploreJob, type FeatureReport } from '@/lib/insight'
import FeatureForecastView from '@/components/objective/FeatureForecastView'

/**
 * Every forecast the team built and what came of it: the exact inputs sent to the model, its
 * in-sample skill, the lift its input columns gave (Chronos-2: the same forecast with and
 * without them, at the same points), who used it -- and whether using it helped, judged
 * against each user's own parent where one exists without the forecast. Below it, the
 * explored Chronos-2 input combinations (never re-tested), and a form to explore more.
 */
export default function ForecastsTab({ projectId, objectiveId }: { projectId: string; objectiveId: string }) {
  const [feats, setFeats] = useState<FeatureReport[] | null>(null)
  const [groups, setGroups] = useState<ComboGroup[]>([])
  const [running, setRunning] = useState<ExploreJob[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)

  const load = useCallback(() => {
    insight
      .forecastReport(objectiveId)
      .then((d) => (setFeats(d.features), setErr(null)))
      .catch((x) => setErr(x instanceof Error ? x.message : String(x)))
    insight
      .combos(projectId, objectiveId)
      .then((d) => (setGroups(d.groups), setRunning(d.running)))
      .catch(() => undefined)
  }, [projectId, objectiveId])

  useEffect(() => {
    load()
    const t = setInterval(load, 12000)
    return () => clearInterval(t)
  }, [load])

  return (
    <div className="space-y-6">
      {err && <div className="text-[12px] text-bad">✗ {err}</div>}
      <section>
        <Title>Forecast features · inputs sent, skill, and did using them help? (in-sample)</Title>
        {!feats ? (
          <div className="text-[12px] text-ink-faint">loading…</div>
        ) : feats.length === 0 ? (
          <div className="text-[12px] text-ink-faint">No forecasts built yet.</div>
        ) : (
          <table className="w-full font-mono text-[11px]">
            <thead>
              <tr className="text-ink-faint">
                <th className="py-1 text-left font-normal">forecast · inputs sent</th>
                <th className="py-1 text-right font-normal" title="1 - MAE / MAE(no change), in-sample">skill</th>
                <th className="py-1 text-right font-normal" title="the same forecast without its input columns">w/o inputs</th>
                <th className="py-1 text-right font-normal" title="paired skill gain from the inputs, ± 2 SE">input lift</th>
                <th className="py-1 text-right font-normal">used by</th>
                <th className="py-1 pl-3 text-left font-normal" title="vs parent: children that added it vs their parent without it; vs median: users vs candidates with no forecast">
                  helped?
                </th>
              </tr>
            </thead>
            <tbody>
              {feats.map((f) => {
                const series = Object.entries(f.skill)
                const s = series[0]?.[1] ?? {}
                const lift = s.lift
                const isOpen = open === f.view
                return (
                  <Fragment key={f.view}>
                    <tr
                      className="cursor-pointer border-t border-seam/60 align-top hover:bg-panel-hi/60"
                      onClick={() => setOpen(isOpen ? null : f.view)}
                    >
                      <td className="py-1 pr-3">
                        <span className="text-ink">{f.view}</span>
                        {f.auto && <span className="ml-1 text-ink-faint">(ft.forecast)</span>}
                        <Recipe f={f} />
                      </td>
                      <td className={`py-1 text-right ${tone(s.skill)}`}>{num(s.skill)}</td>
                      <td className="py-1 text-right text-ink-dim">{s.without_inputs ? num(s.without_inputs.skill) : '—'}</td>
                      <td className={`py-1 text-right ${lift?.significant ? tone(lift.gain) : 'text-ink-faint'}`}>
                        {lift && lift.gain !== null ? `${signed(lift.gain, 4)} ±${num(2 * (lift.se ?? 0), 4)}` : '—'}
                      </td>
                      <td className="py-1 text-right text-ink">{f.used_by}</td>
                      <td className="py-1 pl-3">
                        <span className={VERDICT_TONE[f.verdict]}>{f.verdict}</span>
                        {f.verdict !== 'unused' && (
                          <div className="text-[10px] text-ink-faint">
                            {f.basis} · n={f.n}
                            {f.effect !== null ? ` · ${signed(f.effect, 3)}` : ''}
                          </div>
                        )}
                      </td>
                    </tr>
                    {isOpen && (
                      <tr className="border-t border-seam/30">
                        <td colSpan={6} className="pb-3 pt-1">
                          <FeatureDetail f={f} objectiveId={objectiveId} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        )}
      </section>

      <Explorer projectId={projectId} objectiveId={objectiveId} running={running} onStarted={load} />

      <section>
        <Title>Explored Chronos-2 input combinations · each compared with the target alone at the same points</Title>
        {groups.length === 0 ? (
          <div className="text-[12px] text-ink-faint">
            None yet. An exploration (above, or an agent&apos;s explore_forecast_inputs) screens each input alone, then adds
            inputs greedily while each step is significant, then prunes. Every combination is stored and never forecast again.
          </div>
        ) : (
          <div className="space-y-4">
            {groups.map((g) => (
              <ComboGroupView key={`${g.target}|${g.horizon}|${g.bar}|${g.model}|${g.end}|${g.points}`} g={g} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

function Recipe({ f }: { f: FeatureReport }) {
  const i = f.inputs_sent
  return (
    <div className="text-[10px] leading-snug text-ink-faint">
      {i.model ?? '?'} · target <span className="text-ink-dim">{(i.target ?? ['?']).join(', ')}</span>
      {i.covariates.length > 0 && (
        <>
          {' ← '}
          <span className="text-accent">{i.covariates.join(', ')}</span>
        </>
      )}
      {i.calendar && ' + calendar'}
      {' · '}h{i.horizon} every {i.every}
      {i.bar ? ` · ${i.bar} bars` : ''}
      {i.context ? ` · ctx ${i.context}` : ''}
      {f.requested_by ? ` · by ${f.requested_by}` : ''}
    </div>
  )
}

function FeatureDetail({ f, objectiveId }: { f: FeatureReport; objectiveId: string }) {
  return (
    <div className="space-y-2 rounded-lg border border-seam bg-panel-hi/40 p-3 text-[11px]">
      {/* The inputs it read and the cone it returned, against what followed (in-sample). */}
      <FeatureForecastView objectiveId={objectiveId} view={f.view} />
      <div className="grid gap-2 sm:grid-cols-2">
        {Object.entries(f.skill).map(([series, s]) => (
          <div key={series}>
            <div className="text-ink">{series}</div>
            <div className="text-ink-dim">
              skill {num(s.skill)} · direction {num(s.direction, 3)} · 10–90% coverage {num(s.coverage, 3)}
              {s.points ? ` · ${s.points} in-sample points` : ''}
            </div>
            {s.without_inputs && (
              <div className="text-ink-dim">
                without inputs: skill {num(s.without_inputs.skill)} · direction {num(s.without_inputs.direction)} → lift{' '}
                <span className={s.lift?.significant ? tone(s.lift?.gain) : 'text-ink-faint'}>
                  {signed(s.lift?.gain ?? null, 4)} ± {num(2 * (s.lift?.se ?? 0), 4)} {s.lift?.significant ? '(significant)' : '(within noise)'}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="text-ink-dim">
        users&apos; median in-sample {num(f.median_users, 3)} vs no-forecast median {num(f.median_no_forecast, 3)}
        {f.used_by_seqs.length > 0 && <> · used by #{f.used_by_seqs.join(', #')}</>}
      </div>
      {f.pairs.length > 0 && (
        <div className="text-ink-dim">
          vs parent:{' '}
          {f.pairs.map((p) => (
            <span key={p.seq} className={`mr-2 ${tone(p.delta)}`}>
              #{p.parent_seq}→#{p.seq} {signed(p.delta, 3)}
            </span>
          ))}
        </div>
      )}
      {f.request && (
        <pre className="max-h-[160px] overflow-auto whitespace-pre-wrap rounded bg-panel p-2 text-[10.5px] text-ink-faint">
          {JSON.stringify(f.request, null, 1)}
        </pre>
      )}
    </div>
  )
}

function Explorer({
  projectId,
  objectiveId,
  running,
  onStarted,
}: {
  projectId: string
  objectiveId: string
  running: ExploreJob[]
  onStarted: () => void
}) {
  const [target, setTarget] = useState('Close')
  const [inputs, setInputs] = useState('')
  const [horizon, setHorizon] = useState(30)
  const [bar, setBar] = useState('')
  const [budget, setBudget] = useState(30)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  async function go() {
    setBusy(true)
    setMsg(null)
    try {
      const j = await insight.explore({
        project_id: projectId,
        objective_id: objectiveId,
        target: target.trim(),
        inputs: inputs.split(',').map((s) => s.trim()).filter(Boolean),
        horizon,
        bar: bar.trim() || null,
        budget,
        author: 'operator',
      })
      setMsg(j.already_running ? 'That exploration is already running.' : 'Started — combinations appear below as they are scored.')
      onStarted()
    } catch (x) {
      setMsg(`✗ ${x instanceof Error ? x.message : String(x)}`)
    } finally {
      setBusy(false)
    }
  }

  const field = 'rounded border border-seam bg-panel-hi px-2 py-1 font-mono text-[11px] text-ink outline-none focus:border-accent'
  return (
    <section>
      <Title>Explore Chronos-2 inputs · in-sample, budgeted, cached</Title>
      <div className="flex flex-wrap items-end gap-2">
        <label className="text-[10.5px] text-ink-faint">
          target
          <input value={target} onChange={(e) => setTarget(e.target.value)} className={`${field} ml-1 w-28`} />
        </label>
        <label className="min-w-[240px] flex-1 text-[10.5px] text-ink-faint">
          candidate inputs (comma-separated; empty = field-scan leaders + monotone decile signals)
          <input value={inputs} onChange={(e) => setInputs(e.target.value)} className={`${field} mt-0.5 block w-full`} />
        </label>
        <label className="text-[10.5px] text-ink-faint">
          horizon
          <input type="number" value={horizon} min={1} max={512} onChange={(e) => setHorizon(+e.target.value)} className={`${field} ml-1 w-16`} />
        </label>
        <label className="text-[10.5px] text-ink-faint">
          bar
          <input value={bar} placeholder="native" onChange={(e) => setBar(e.target.value)} className={`${field} ml-1 w-16`} />
        </label>
        <label className="text-[10.5px] text-ink-faint">
          budget
          <input type="number" value={budget} min={3} max={300} onChange={(e) => setBudget(+e.target.value)} className={`${field} ml-1 w-16`} />
        </label>
        <button
          disabled={busy || !target.trim()}
          onClick={go}
          className="rounded-md border border-accent/40 px-2.5 py-1 font-mono text-[11px] text-accent hover:bg-accent/10 disabled:opacity-40"
        >
          {busy ? 'starting…' : 'explore'}
        </button>
      </div>
      {msg && <div className="mt-1 text-[11.5px] text-ink-dim">{msg}</div>}
      {running.map((j) => (
        <div key={j.id} className="mt-1 font-mono text-[11px] text-ink-dim">
          <span className="inline-block size-1.5 animate-dot rounded-full bg-accent" /> {j.params.target} h{j.params.horizon}: {j.phase}
          {' · '}
          {j.done}/{j.total} new{j.cache_hits ? ` · ${j.cache_hits} from the store` : ''}
          {j.current ? ` · ${j.current}` : ''}
        </div>
      ))}
    </section>
  )
}

function ComboGroupView({ g }: { g: ComboGroup }) {
  const [all, setAll] = useState(false)
  const rows = all ? g.combos : g.combos.slice(0, 12)
  return (
    <div className="rounded-lg border border-seam p-3">
      <div className="flex flex-wrap items-baseline gap-2 font-mono text-[11px]">
        <span className="text-ink">{g.target}</span>
        <span className="text-ink-faint">
          h{g.horizon}
          {g.bar ? ` @${g.bar}` : ''} · {g.model} · {g.points} points · ctx {g.context} · data before {g.end ?? 'all'}
        </span>
        <span className="ml-auto text-ink-dim">
          alone {num(g.baseline?.skill ?? null)} · best{' '}
          {g.best ? (
            <span className="text-good">
              {g.best.inputs.join(' + ')} {num(g.best.skill)} ({signed(g.best.gain, 4)})
            </span>
          ) : (
            <span className="text-ink-faint">none beyond noise</span>
          )}{' '}
          · {g.tested} tested
        </span>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1 font-mono text-[10.5px]">
        {g.helpful.map((c) => (
          <span key={c} className="rounded border border-good/40 px-1.5 text-good" title="alone, it beats the target alone beyond ± 2 SE">
            ✓ {c}
          </span>
        ))}
        {g.hurts.map((c) => (
          <span key={c} className="rounded border border-bad/40 px-1.5 text-bad" title="alone, it makes the forecast significantly worse">
            ✗ {c}
          </span>
        ))}
        {g.useless
          .filter((c) => !g.hurts.includes(c))
          .map((c) => (
            <span key={c} className="rounded border border-seam px-1.5 text-ink-faint" title="tested alone: within noise">
              {c}
            </span>
          ))}
      </div>
      <table className="mt-2 w-full font-mono text-[10.5px]">
        <thead>
          <tr className="text-ink-faint">
            <th className="py-0.5 text-left font-normal">inputs</th>
            <th className="py-0.5 text-right font-normal">skill</th>
            <th className="py-0.5 text-right font-normal">direction</th>
            <th className="py-0.5 text-right font-normal">vs alone ± 2 SE</th>
            <th className="py-0.5 pl-3 text-left font-normal">stage · by</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.inputs.join('|') || '(alone)'} className="border-t border-seam/40">
              <td className="py-0.5 pr-2 text-ink-dim">{c.inputs.length ? c.inputs.join(' + ') : <span className="text-ink-faint">(target alone)</span>}</td>
              <td className="py-0.5 text-right text-ink">{num(c.skill)}</td>
              <td className="py-0.5 text-right text-ink-dim">{num(c.direction, 3)}</td>
              <td className={`py-0.5 text-right ${c.significant ? tone(c.gain) : 'text-ink-faint'}`}>
                {c.gain !== null ? `${signed(c.gain, 4)} ±${num(2 * (c.se ?? 0), 4)}` : '—'}
              </td>
              <td className="py-0.5 pl-3 text-ink-faint">
                {c.kind ?? ''} · {c.author ?? ''}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {g.combos.length > 12 && (
        <button onClick={() => setAll(!all)} className="mt-1 font-mono text-[10.5px] text-ink-faint hover:text-accent">
          {all ? 'show top 12' : `show all ${g.combos.length}`}
        </button>
      )}
    </div>
  )
}

function Title({ children }: { children: React.ReactNode }) {
  return <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">{children}</div>
}

function tone(v: number | null | undefined): string {
  return typeof v === 'number' ? (v > 0 ? 'text-good' : v < 0 ? 'text-bad' : 'text-ink-dim') : 'text-ink-faint'
}
