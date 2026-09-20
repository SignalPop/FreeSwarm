'use client'

import { useEffect, useMemo, useState } from 'react'
import { duration } from '@/lib/format'
import { projects } from '@/lib/projects'
import { objectives, type Objective } from '@/lib/objectives'
import { tslab, type AnalysisResults, type AnalysisSummary, type Job, type LabOptions, type Score, type Paired, type TestResult } from '@/lib/tslab'
import { Button, PageHeader, Panel, Pill } from '@/components/ui'
import { FanChart, GreedyPath, ImpactBars } from '@/components/lab/LabCharts'

const field = 'rounded-lg border border-seam bg-panel-hi px-2 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent'
const label = 'mb-1 block text-[10.5px] uppercase tracking-wide text-ink-faint'
const f3 = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(3))
const pct = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`)

function ScoreRow({ name, s, lift }: { name: string; s: Score; lift?: Paired | null }) {
  return (
    <tr className="border-t border-seam/60">
      <td className="py-1 text-ink-dim">{name}</td>
      <td className={`py-1 text-right ${(s.skill ?? 0) > 0 ? 'text-good' : 'text-ink'}`}>{f3(s.skill)}</td>
      <td className="py-1 text-right">{pct(s.direction)}</td>
      <td className="py-1 text-right">{pct(s.coverage)}</td>
      <td className="py-1 text-right">{f3(s.qloss_rel)}</td>
      <td className="py-1 pl-3">
        {lift && lift.gain !== null && (
          <span className={lift.significant ? (lift.gain > 0 ? 'text-good' : 'text-bad') : 'text-ink-faint'}>
            {lift.gain > 0 ? '+' : ''}
            {lift.gain.toFixed(4)} ± {(2 * (lift.se ?? 0)).toFixed(4)} {lift.significant ? '(significant)' : '(within noise)'}
          </span>
        )}
      </td>
    </tr>
  )
}

function ScoreTable({ rows }: { rows: { name: string; s: Score; lift?: Paired | null }[] }) {
  return (
    <table className="w-full font-mono text-[11.5px]">
      <thead>
        <tr className="text-left text-ink-faint">
          <th className="py-1 font-normal">forecast</th>
          <th className="py-1 text-right font-normal" title="1 - error / error of 'no change'; > 0 beats no change">skill</th>
          <th className="py-1 text-right font-normal">direction</th>
          <th className="py-1 text-right font-normal" title="share of outcomes inside the 10-90% band; 80% is calibrated">10–90% band</th>
          <th className="py-1 text-right font-normal" title="quantile loss relative to no change; lower is better">q-loss</th>
          <th className="py-1 pl-3 font-normal">gain vs target only (±2 SE)</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <ScoreRow key={r.name} {...r} />
        ))}
      </tbody>
    </table>
  )
}

/**
 * Forecast Lab: try any combination of inputs with a covariate forecaster (Chronos-2) and see
 * whether they beat forecasting the target alone -- or run the full analysis, which reverses
 * the model: removes each input from the full set (impact), tries each alone (solo lift), and
 * builds the best combination step by step. All scored on the in-sample period, with error bars.
 */
export default function ForecastLabPage() {
  const [projectId, setProjectId] = useState<string | null>(null)
  const [objs, setObjs] = useState<Objective[]>([])
  const [objectiveId, setObjectiveId] = useState<string>('')
  const [opts, setOpts] = useState<LabOptions | null>(null)
  const [model, setModel] = useState('')
  const [target, setTarget] = useState('Close')
  const [inputs, setInputs] = useState<Set<string>>(new Set())
  const [filter, setFilter] = useState('')
  const [horizon, setHorizon] = useState(30)
  const [context, setContext] = useState(1024)
  const [bar, setBar] = useState('1min')
  const [points, setPoints] = useState(400)
  const [busy, setBusy] = useState<'test' | 'analyze' | 'feature' | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [test, setTest] = useState<TestResult | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [history, setHistory] = useState<AnalysisSummary[]>([])
  const [featureMsg, setFeatureMsg] = useState<string | null>(null)

  useEffect(() => {
    projects.list().then((r) => setProjectId(r.active)).catch(() => {})
  }, [])
  useEffect(() => {
    if (!projectId) return
    objectives.list(projectId).then((r) => {
      setObjs(r.objectives)
      setObjectiveId((cur) => cur || r.objectives.find((o) => o.status === 'running')?.id || r.objectives[0]?.id || '')
    })
    tslab.analyses(projectId).then((r) => {
      setHistory(r.analyses)
      if (r.running[0]) setJob(r.running[0])
    })
  }, [projectId])
  useEffect(() => {
    if (!projectId) return
    tslab
      .options(projectId, objectiveId || null)
      .then((o) => {
        setOpts(o)
        setModel((cur) => cur || o.models.find((m) => m.covariates && m.state === 'running')?.model || '')
        if (!o.numeric.includes(target) && o.numeric.length) setTarget(o.numeric.includes('Close') ? 'Close' : o.numeric[0])
      })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, objectiveId])

  // Poll a running analysis.
  useEffect(() => {
    if (!job || ['done', 'error'].includes(job.phase)) return
    const t = setInterval(async () => {
      try {
        const j = await tslab.job(job.id)
        setJob(j)
        if (['done', 'error'].includes(j.phase) && projectId) tslab.analyses(projectId).then((r) => setHistory(r.analyses))
      } catch {
        /* keep polling */
      }
    }, 2000)
    return () => clearInterval(t)
  }, [job, projectId])

  const setup = () => ({
    project_id: projectId as string,
    objective_id: objectiveId || null,
    target,
    inputs: [...inputs].filter((i) => i !== target),
    horizon,
    context,
    bar: bar || null,
    points,
    model: model || null,
  })

  async function run(kind: 'test' | 'analyze') {
    setBusy(kind)
    setErr(null)
    try {
      if (kind === 'test') setTest(await tslab.test(setup()))
      else setJob(await tslab.analyze(setup()))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  async function buildFeature(best: string[]) {
    if (!objectiveId) return
    setBusy('feature')
    setFeatureMsg(null)
    try {
      const f = await tslab.buildFeature(objectiveId, {
        column: target, covariates: best, horizon, context: Math.min(context, 2048), bar: bar || null, model: model || null,
      })
      setFeatureMsg(`Built ${f.view} (${f.rows} forecasts) — agents can load it now; its lift vs the target alone is recorded with it.`)
    } catch (e) {
      setFeatureMsg(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const familyOf = useMemo(() => {
    const m = new Map<string, string>()
    Object.entries(opts?.families ?? {}).forEach(([fam, cols]) => cols.forEach((c) => m.set(c, fam)))
    return (c: string) => m.get(c) ?? ''
  }, [opts])

  const results = job?.results && 'impact' in (job.results as AnalysisResults) ? (job.results as AnalysisResults) : null
  const covModels = (opts?.models ?? []).filter((m) => m.covariates)
  const toggle = (c: string) =>
    setInputs((s) => {
      const n = new Set(s)
      if (n.has(c)) n.delete(c)
      else n.add(c)
      return n
    })

  return (
    <div className="mx-auto max-w-[1180px] px-8 py-8">
      <PageHeader
        title="Forecast Lab"
        subtitle="Which inputs make a time-series forecast better? Test combinations, or let the analysis find out — scored in-sample, with error bars."
      />
      {err && <Panel className="mb-4 border-bad/35 bg-bad/5 p-3 text-[12.5px] text-bad">{err}</Panel>}
      {opts && covModels.filter((m) => m.state === 'running').length === 0 && (
        <Panel className="mb-4 border-warn/40 bg-warn/5 p-3 text-[12.5px] text-warn">
          No model that takes input series is loaded. Load <span className="font-mono">amazon/chronos-2</span> on the Models page
          (download it there first if needed).
        </Panel>
      )}

      <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
        {/* ---- Setup ---- */}
        <Panel className="h-fit space-y-3 p-4">
          <div>
            <label className={label}>Objective (sets dataset and hides its holdout)</label>
            <select className={`${field} w-full`} value={objectiveId} onChange={(e) => setObjectiveId(e.target.value)}>
              <option value="">(none — score on all data)</option>
              {objs.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.title.slice(0, 50)}
                </option>
              ))}
            </select>
            {opts?.split_date && <div className="mt-1 font-mono text-[10.5px] text-ink-faint">scored before {opts.split_date} only</div>}
          </div>
          <div>
            <label className={label}>Model</label>
            <select className={`${field} w-full`} value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="">(first that takes inputs)</option>
              {(opts?.models ?? []).map((m) => (
                <option key={m.model} value={m.model} disabled={!m.covariates}>
                  {m.model}
                  {m.covariates ? '' : ' — single series only'}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className={label}>Target (what to forecast)</label>
            <select className={`${field} w-full`} value={target} onChange={(e) => setTarget(e.target.value)}>
              {(opts?.numeric ?? []).map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className={label}>Horizon (bars)</label>
              <input type="number" min={1} max={512} className={`${field} w-full`} value={horizon} onChange={(e) => setHorizon(Math.max(1, +e.target.value || 1))} />
            </div>
            <div>
              <label className={label}>Bar</label>
              <select className={`${field} w-full`} value={bar} onChange={(e) => setBar(e.target.value)}>
                <option value="">raw (10s)</option>
                {['30s', '1min', '5min', '15min'].map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className={label}>Context (bars)</label>
              <input type="number" min={64} max={8192} step={64} className={`${field} w-full`} value={context} onChange={(e) => setContext(Math.max(64, +e.target.value || 512))} />
            </div>
            <div>
              <label className={label}>Test points</label>
              <input type="number" min={30} max={2000} step={50} className={`${field} w-full`} value={points} onChange={(e) => setPoints(Math.max(30, +e.target.value || 300))} />
            </div>
          </div>

          <div>
            <div className="mb-1 flex items-center gap-2">
              <span className={label + ' mb-0'}>Inputs ({inputs.size})</span>
              {(opts?.suggested.length ?? 0) > 0 && (
                <button onClick={() => setInputs(new Set(opts!.suggested.filter((c) => c !== target)))} className="ml-auto font-mono text-[10.5px] text-accent" title="the latest field scan's strongest fields">
                  top from field scan
                </button>
              )}
              <button onClick={() => setInputs(new Set())} className="font-mono text-[10.5px] text-ink-faint hover:text-ink">
                clear
              </button>
            </div>
            <input className={`${field} mb-1.5 w-full`} placeholder="filter fields…" value={filter} onChange={(e) => setFilter(e.target.value)} />
            <div className="max-h-[320px] space-y-2 overflow-y-auto pr-1">
              {Object.entries(opts?.families ?? {}).map(([fam, cols]) => {
                const shown = cols.filter((c) => c !== target && c.toLowerCase().includes(filter.toLowerCase()))
                if (!shown.length) return null
                const allOn = shown.every((c) => inputs.has(c))
                return (
                  <div key={fam}>
                    <button
                      onClick={() => setInputs((s) => { const n = new Set(s); shown.forEach((c) => (allOn ? n.delete(c) : n.add(c))); return n })}
                      className="font-mono text-[10.5px] text-ink-faint hover:text-ink"
                    >
                      {fam} {allOn ? '−' : '+'}
                    </button>
                    <div className="mt-0.5 flex flex-wrap gap-1">
                      {shown.map((c) => (
                        <button
                          key={c}
                          onClick={() => toggle(c)}
                          className={`rounded border px-1.5 py-0.5 font-mono text-[10.5px] ${
                            inputs.has(c) ? 'border-accent/60 bg-accent/10 text-ink' : 'border-seam text-ink-faint hover:text-ink-dim'
                          }`}
                        >
                          {c}
                        </button>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="flex flex-wrap gap-2 pt-1">
            <Button tone="primary" disabled={!!busy || !projectId} onClick={() => run('test')}>
              {busy === 'test' ? 'Testing…' : inputs.size ? 'Test this combination' : 'Test target alone'}
            </Button>
            <Button tone="ghost" disabled={!!busy || inputs.size < 2 || (!!job && !['done', 'error'].includes(job.phase))} onClick={() => run('analyze')}>
              Run full analysis
            </Button>
          </div>
          <div className="text-[11px] leading-relaxed text-ink-faint">
            Full analysis: target alone, all inputs, each input removed (impact), each input alone (solo lift), then the best
            combination built step by step. About {2 + 2 * inputs.size + 4 * Math.min(10, inputs.size)} combinations × {points} points.
          </div>
        </Panel>

        {/* ---- Results ---- */}
        <div className="min-w-0 space-y-4">
          {test && (
            <Panel className="p-4">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="text-[14px] font-medium text-ink">Test: {test.target}</span>
                <span className="font-mono text-[10.5px] text-ink-faint">
                  {test.inputs.length ? `+ ${test.inputs.join(', ')}` : 'target only'} · {test.baseline.points} points · {test.seconds}s
                  {test.end ? ` · before ${test.end}` : ''}
                </span>
              </div>
              {test.note && <div className="mb-2 text-[11.5px] text-warn">{test.note}</div>}
              <ScoreTable rows={[{ name: 'target only', s: test.baseline }, ...(test.inputs.length ? [{ name: 'with inputs', s: test.combo, lift: test.lift }] : [])]} />
              <div className="mt-3 grid gap-2 md:grid-cols-3">
                {test.examples.map((ex, i) => (
                  <FanChart key={i} title={ex.t.slice(0, 16)} history={ex.history} actual={ex.actual} baseline={ex.baseline} withInputs={ex.with_inputs} />
                ))}
              </div>
              <div className="mt-1 font-mono text-[10px] text-ink-faint">
                white = what happened · grey dashed/band = target only · blue line/band = with inputs
              </div>
            </Panel>
          )}

          {job && (
            <Panel className="p-4">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="text-[14px] font-medium text-ink">Analysis{results ? `: ${results.target}` : ''}</span>
                <Pill tone={job.phase === 'done' ? 'good' : job.phase === 'error' ? 'bad' : 'accent'} pulse={!['done', 'error'].includes(job.phase)}>
                  {job.phase}
                </Pill>
                {!['done', 'error'].includes(job.phase) && (
                  <span className="font-mono text-[10.5px] text-ink-faint">
                    {job.done ?? 0}/{job.total ?? '?'} combinations · {job.current}
                  </span>
                )}
              </div>
              {!['done', 'error'].includes(job.phase) && (
                <div className="h-1.5 overflow-hidden rounded bg-seam">
                  <div className="h-full bg-accent transition-all" style={{ width: `${Math.min(100, ((job.done ?? 0) / Math.max(1, job.total ?? 1)) * 100)}%` }} />
                </div>
              )}
              {job.error && <div className="text-[12px] text-bad">{job.error}</div>}
              {results && (
                <div className="space-y-4">
                  <ScoreTable
                    rows={[
                      { name: 'target only', s: results.baseline },
                      { name: `all ${results.impact.length} inputs`, s: results.all, lift: { gain: results.all.vs_baseline_gain, se: results.all.vs_baseline_se, significant: results.all.vs_baseline_significant } },
                      { name: `best: ${results.best.inputs.join(' + ') || '(none helped)'}`, s: { ...results.best, points: results.anchors }, lift: { gain: results.best.vs_baseline_gain, se: results.best.vs_baseline_se, significant: results.best.vs_baseline_significant } },
                    ]}
                  />
                  <div className={`rounded-lg border px-3 py-2 text-[12px] ${results.best.vs_baseline_significant && (results.best.vs_baseline_gain ?? 0) > 0 ? 'border-good/40 bg-good/5 text-good' : 'border-seam bg-panel-hi/40 text-ink-dim'}`}>
                    {results.best.inputs.length === 0
                      ? 'No input improved the forecast over the target alone.'
                      : results.best.vs_baseline_significant && (results.best.vs_baseline_gain ?? 0) > 0
                        ? `${results.best.inputs.join(' + ')} improves the forecast beyond noise.`
                        : `Best found: ${results.best.inputs.join(' + ')} — but the gain is within noise (±2 SE crosses zero). More test points would tell.`}
                    {objectiveId && results.best.inputs.length > 0 && (
                      <button onClick={() => buildFeature(results.best.inputs)} disabled={!!busy} className="ml-2 font-mono text-[11px] text-accent hover:opacity-80">
                        {busy === 'feature' ? 'building…' : 'build as a forecast feature for the agents →'}
                      </button>
                    )}
                    {featureMsg && <div className="mt-1 text-ink-dim">{featureMsg}</div>}
                  </div>
                  <div>
                    <div className="mb-1 text-[12.5px] font-medium text-ink">Impact — how much worse the forecast gets without each input</div>
                    <ImpactBars familyOf={familyOf}
                      items={results.impact.map((i) => ({ label: i.input, gain: i.impact_gain, se: i.impact_se, significant: i.impact_significant }))} />
                  </div>
                  <div>
                    <div className="mb-1 text-[12.5px] font-medium text-ink">Solo lift — each input alone vs the target alone</div>
                    <ImpactBars items={results.solo.map((s) => ({ label: s.input, gain: s.lift_gain, se: s.lift_se, significant: s.lift_significant }))} />
                  </div>
                  <div>
                    <div className="mb-1 text-[12.5px] font-medium text-ink">Best combination, built one input at a time</div>
                    <GreedyPath path={results.greedy_path} />
                  </div>
                  <div className="text-[11px] leading-relaxed text-ink-faint">
                    Bars are the mean gain in skill (error relative to a no-change forecast) at the same {results.anchors} points; whiskers
                    are ±2 standard errors. Faded bars cross zero — they may be noise. All scores are in-sample
                    {results.end ? ` (before ${results.end})` : ''}: the holdout the agents are ranked on stays unseen.
                  </div>
                </div>
              )}
            </Panel>
          )}

          {!test && !job && (
            <Panel className="p-6 text-[12.5px] leading-relaxed text-ink-dim">
              Pick a target and some inputs on the left. <b>Test this combination</b> scores it against forecasting the target alone
              and plots examples. <b>Run full analysis</b> reverses the model: it removes each input from the full set to measure its
              impact, tries each input alone, and builds the best combination step by step — every number with an error bar, so a
              lucky result is not mistaken for a real one.
            </Panel>
          )}

          {history.length > 0 && (
            <Panel className="p-4">
              <div className="mb-2 text-[13px] font-medium text-ink">Previous analyses</div>
              {history.map((h) => (
                <button key={h.id} onClick={() => tslab.job(h.id).then(setJob)}
                  className="flex w-full items-center gap-2 border-t border-seam/60 py-1.5 text-left font-mono text-[11px] text-ink-dim first:border-t-0 hover:text-ink">
                  <span className="text-ink">{h.params.target}</span>
                  <span className="truncate">{h.best?.inputs?.length ? `best: ${h.best.inputs.join(' + ')}` : h.status}</span>
                  {h.best?.vs_baseline_significant && <Pill tone="good">significant</Pill>}
                  <span className="ml-auto shrink-0 text-ink-faint">{duration(Date.now() / 1000 - h.created_at)} ago</span>
                </button>
              ))}
            </Panel>
          )}
        </div>
      </div>
    </div>
  )
}
