'use client'

import { useEffect, useState } from 'react'
import {
  METRIC_OPTIONS,
  RETURN_METRICS,
  objectives,
  type MetricKind,
  type Objective,
  type Probe,
} from '@/lib/objectives'
import { Button } from '@/components/ui'

const field =
  'w-full rounded-lg border border-seam bg-panel-hi px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-accent'
const label = 'mb-1 block text-[11px] uppercase tracking-wide text-ink-faint'

/**
 * Define an objective: what the swarm pursues, and -- the part that makes "better" mean
 * something -- how a result is measured. For a return metric the form reads the chosen
 * dataset and proposes the time column, the split date (holdout = the last N% of days) and
 * the price column the harness marks positions against.
 */
export default function NewObjective({
  projectId,
  initialText = '',
  onClose,
  onCreated,
}: {
  projectId: string
  initialText?: string
  onClose: () => void
  onCreated: (o: Objective) => void
}) {
  const [firstLine, ...rest] = initialText.split('\n')
  const [title, setTitle] = useState(firstLine.slice(0, 300))
  const [description, setDescription] = useState(rest.join('\n').trim())
  const [kind, setKind] = useState<MetricKind>('sharpe')
  const [higher, setHigher] = useState(true)
  const [rubric, setRubric] = useState('')
  const [datasets, setDatasets] = useState<string[]>([])
  const [dataset, setDataset] = useState('')
  const [holdout, setHoldout] = useState(30)
  const [probe, setProbe] = useState<Probe | null>(null)
  const [probing, setProbing] = useState(false)
  const [split, setSplit] = useState('')
  const [price, setPrice] = useState('')
  const [costBps, setCostBps] = useState(1)
  const [lev, setLev] = useState(1)
  const [ppy, setPpy] = useState(252)
  const [minActive, setMinActive] = useState(20)
  const [lookahead, setLookahead] = useState(true)
  const [audit, setAudit] = useState(true)
  const [timeout, setTimeoutS] = useState(300)
  const [cooldown, setCooldown] = useState(5)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const returnsMetric = RETURN_METRICS.includes(kind)

  useEffect(() => {
    objectives
      .dataCatalog(projectId)
      .then((d) => {
        const views = d.files.map((f) => f.view)
        setDatasets(views)
        setDataset((cur) => cur || views[0] || '')
      })
      .catch(() => setDatasets([]))
  }, [projectId])

  useEffect(() => {
    if (!returnsMetric || !dataset) {
      setProbe(null)
      return
    }
    let alive = true
    setProbing(true)
    objectives
      .probe(projectId, dataset, holdout / 100)
      .then((p) => {
        if (!alive) return
        setProbe(p)
        setSplit(p.split_date ?? '')
        setPrice(p.price_column ?? '')
      })
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setProbing(false))
    return () => {
      alive = false
    }
  }, [projectId, dataset, holdout, returnsMetric])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  async function create() {
    setBusy(true)
    setErr(null)
    try {
      const o = await objectives.create(projectId, {
        title: title.trim(),
        description: description.trim(),
        metric: {
          kind,
          higher_is_better: kind === 'reported' ? higher : true,
          rubric,
          periods_per_year: ppy,
          min_active_days: minActive,
          // '' = deliberately none (the server would otherwise guess one)
          price_column: returnsMetric ? price : null,
          cost_bps: costBps,
          max_leverage: lev,
        },
        dataset: returnsMetric ? dataset || null : null,
        time_column: returnsMetric ? probe?.time_column ?? null : null,
        split_date: returnsMetric && split ? split : null,
        lookahead_check: lookahead,
        require_audit: audit,
        eval_timeout_s: timeout,
        cooldown_s: cooldown,
      })
      onCreated(o)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const canCreate = title.trim().length > 0 && (!returnsMetric || !!dataset) && !busy

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={onClose}>
      <div
        className="flex max-h-[90vh] w-full max-w-[760px] flex-col overflow-hidden rounded-2xl border border-seam bg-panel shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-seam px-5 py-4">
          <div className="text-[15px] font-medium text-ink">New objective</div>
          <div className="mt-1 text-[12px] text-ink-faint">
            The swarm works on it continuously, submitting candidates that a harness scores. Your
            messages then steer it instead of starting new tasks.
          </div>
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
          <div>
            <label className={label}>Objective</label>
            <input className={field} value={title} onChange={(e) => setTitle(e.target.value)}
              placeholder="Create the best intraday trading strategy possible" />
          </div>
          <div>
            <label className={label}>Details for the agents</label>
            <textarea className={`${field} min-h-[90px] font-mono text-[12px]`} value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Use the GEX fields to find predictable intraday patterns. Long/short allowed. Flat overnight preferred." />
          </div>

          <div>
            <label className={label}>How is "better" measured?</label>
            <div className="grid gap-2 sm:grid-cols-2">
              {METRIC_OPTIONS.map((m) => (
                <button
                  key={m.kind}
                  onClick={() => setKind(m.kind)}
                  className={`rounded-lg border px-3 py-2 text-left transition-colors ${
                    kind === m.kind ? 'border-accent bg-accent/10' : 'border-seam hover:border-ink-faint'
                  }`}
                >
                  <div className="text-[12.5px] text-ink">{m.label}</div>
                  <div className="text-[11px] text-ink-faint">{m.hint}</div>
                </button>
              ))}
            </div>
          </div>

          {kind === 'reported' && (
            <label className="flex items-center gap-2 text-[12.5px] text-ink-dim">
              <input type="checkbox" checked={higher} onChange={(e) => setHigher(e.target.checked)} />
              Higher is better (untick if lower wins, e.g. an error)
            </label>
          )}
          {kind === 'judge' && (
            <div>
              <label className={label}>Rubric for the judge model</label>
              <textarea className={`${field} min-h-[70px]`} value={rubric} onChange={(e) => setRubric(e.target.value)}
                placeholder="Correctness first, then clarity; penalise unsupported claims." />
            </div>
          )}

          {returnsMetric && (
            <div className="space-y-3 rounded-xl border border-seam p-3">
              <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_120px]">
                <div>
                  <label className={label}>Dataset</label>
                  <select className={field} value={dataset} onChange={(e) => setDataset(e.target.value)}>
                    {datasets.length === 0 && <option value="">no datasets in the project data folder</option>}
                    {datasets.map((d) => (
                      <option key={d} value={d}>{d}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className={label}>Holdout %</label>
                  <input type="number" min={5} max={70} className={field} value={holdout}
                    onChange={(e) => setHoldout(Math.max(5, Math.min(70, Number(e.target.value) || 30)))} />
                </div>
              </div>
              {probing && <div className="text-[12px] text-ink-faint">Reading the dataset…</div>}
              {probe && !probing && (
                <>
                  {probe.time_column ? (
                    <div className="font-mono text-[11.5px] text-ink-dim">
                      time column <span className="text-ink">{probe.time_column}</span> · {probe.first_date} → {probe.last_date} ·{' '}
                      {probe.days} days
                    </div>
                  ) : (
                    <div className="text-[12px] text-warn">{probe.note}</div>
                  )}
                  <div className="grid gap-3 sm:grid-cols-3">
                    <div>
                      <label className={label}>Split date</label>
                      <input className={field} value={split} onChange={(e) => setSplit(e.target.value)} placeholder="YYYY-MM-DD" />
                    </div>
                    <div>
                      <label className={label}>Price column</label>
                      <select className={field} value={price} onChange={(e) => setPrice(e.target.value)}>
                        <option value="">none — scripts report returns</option>
                        {probe.numeric_columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className={label}>Cost (bps / trade)</label>
                      <input type="number" min={0} step={0.5} className={field} value={costBps}
                        onChange={(e) => setCostBps(Math.max(0, Number(e.target.value) || 0))} />
                    </div>
                    <div>
                      <label className={label}>Max |position|</label>
                      <input type="number" min={0.1} step={0.5} className={field} value={lev}
                        onChange={(e) => setLev(Math.max(0.1, Number(e.target.value) || 1))} />
                    </div>
                    <div>
                      <label className={label}>Periods / year</label>
                      <input type="number" min={1} className={field} value={ppy}
                        onChange={(e) => setPpy(Math.max(1, Number(e.target.value) || 252))} />
                    </div>
                    <div>
                      <label className={label}>Min active days</label>
                      <input type="number" min={0} className={field} value={minActive}
                        onChange={(e) => setMinActive(Math.max(0, Number(e.target.value) || 0))} />
                    </div>
                  </div>
                  <div className="text-[11.5px] leading-relaxed text-ink-faint">
                    {price
                      ? `Candidates report positions per bar; the harness marks them to market on "${price}", charges costs and computes the returns itself — scripts cannot report a return they did not earn.`
                      : 'Without a price column, candidates report their own returns: weaker — returns can be misstated, and the look-ahead test can only compare returns, which misses one-bar peeks.'}{' '}
                    Ranking uses only the period from the split date on; agents never see it.
                  </div>
                  <label className="flex items-center gap-2 text-[12.5px] text-ink-dim">
                    <input type="checkbox" checked={lookahead} onChange={(e) => setLookahead(e.target.checked)} />
                    Look-ahead test (re-run each candidate with future rows removed; reject if past decisions change)
                  </label>
                </>
              )}
            </div>
          )}

          <label className="flex items-center gap-2 text-[12.5px] text-ink-dim">
            <input type="checkbox" checked={audit} onChange={(e) => setAudit(e.target.checked)} />
            Audit would-be champions (a second model reviews the code before it takes the title)
          </label>

          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className={label}>Run time limit per candidate (s)</label>
              <input type="number" min={30} max={600} className={field} value={timeout}
                onChange={(e) => setTimeoutS(Math.max(30, Math.min(600, Number(e.target.value) || 300)))} />
            </div>
            <div>
              <label className={label}>Pause between iterations (s)</label>
              <input type="number" min={0} max={3600} className={field} value={cooldown}
                onChange={(e) => setCooldown(Math.max(0, Math.min(3600, Number(e.target.value) || 0)))} />
            </div>
          </div>
          {err && <div className="font-mono text-[11.5px] text-bad">{err}</div>}
        </div>

        <div className="flex items-center gap-3 border-t border-seam px-5 py-3">
          <span className="text-[11.5px] text-ink-faint">Starts running immediately on this project&apos;s loaded models.</span>
          <Button tone="ghost" className="ml-auto" onClick={onClose}>
            Cancel
          </Button>
          <Button tone="primary" onClick={create} disabled={!canCreate}>
            {busy ? 'Creating…' : 'Start objective'}
          </Button>
        </div>
      </div>
    </div>
  )
}
