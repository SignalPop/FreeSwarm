'use client'

import { useEffect, useState } from 'react'
import {
  METRIC_OPTIONS,
  RETURN_METRICS,
  objectives,
  type Direction,
  type MetricKind,
  type Objective,
  type Probe,
  type LeakScan,
  type TaskServers,
} from '@/lib/objectives'
import { Button } from '@/components/ui'
import { projects as projectsApi } from '@/lib/projects'

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
  // The project's data/action MCP, if it has one: new objectives default to being scored by it.
  const [projectServer, setProjectServer] = useState<string | null>(null)
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
  const [direction, setDirection] = useState<Direction>('both')
  const [intraday, setIntraday] = useState(true)
  const [sideShare, setSideShare] = useState(0.2)
  const [ppy, setPpy] = useState(252)
  const [minActive, setMinActive] = useState(20)
  const [lookahead, setLookahead] = useState(true)
  const [audit, setAudit] = useState(true)
  const [timeout, setTimeoutS] = useState(300)
  const [cooldown, setCooldown] = useState(5)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  // kind 'task': the registered task servers and the chosen "server/task".
  const [taskServers, setTaskServers] = useState<TaskServers | null>(null)
  const [taskErr, setTaskErr] = useState<string | null>(null)
  const [taskRef, setTaskRef] = useState('')
  const [scan, setScan] = useState<{ ref: string; result?: LeakScan; error?: string } | null>(null)

  const returnsMetric = RETURN_METRICS.includes(kind)
  const taskMetric = kind === 'task'

  useEffect(() => {
    if (!taskMetric || taskServers) return
    // Asking each server for its tasks can take a while the first time (a server loads its rows).
    objectives
      .taskServers()
      .then((t) => {
        setTaskServers(t)
        const first = t.servers
          .filter((s) => !projectServer || s.server === projectServer)
          .flatMap((s) => s.tasks.filter((x) => !x.error).map((x) => `${s.server}/${x.name}`))[0]
        setTaskRef((cur) => cur || first || '')
      })
      .catch((e: Error) => setTaskErr(e.message))
  }, [taskMetric, taskServers, projectServer])
  const [taskServer, taskName] = taskRef ? [taskRef.split('/')[0], taskRef.split('/').slice(1).join('/')] : ['', '']
  const chosenTask = taskServers?.servers.find((s) => s.server === taskServer)?.tasks.find((t) => t.name === taskName)

  useEffect(() => {
    projectsApi
      .list()
      .then((r) => {
        const ts = r.projects.find((p) => p.id === projectId)?.task_server ?? null
        setProjectServer(ts)
        if (ts) setKind('task')
      })
      .catch(() => {})
  }, [projectId])

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
          direction,
          intraday,
          min_side_share: direction === 'both' ? sideShare : 0,
          ...(taskMetric ? { task_server: taskServer, task: taskName } : {}),
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

  const canCreate =
    title.trim().length > 0 && (!returnsMetric || !!dataset) && (!taskMetric || !!chosenTask) && !busy

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

          {taskMetric && (
            <div className="space-y-3 rounded-xl border border-seam p-3">
              <div>
                <label className={label}>Task</label>
                {!taskServers && !taskErr && (
                  <div className="text-[12px] text-ink-faint">Asking the registered task servers for their tasks…</div>
                )}
                {taskErr && <div className="text-[12px] text-bad">{taskErr}</div>}
                {taskServers && (
                  <select className={field} value={taskRef} onChange={(e) => setTaskRef(e.target.value)}>
                    {taskServers.servers.every((s) => !s.tasks.length) && (
                      <option value="">no task servers registered -- see docs/task-servers.md</option>
                    )}
                    {taskServers.servers.filter((s) => !projectServer || s.server === projectServer).map((s) => (
                      <optgroup key={s.server} label={s.server}>
                        {s.tasks.map((t) => (
                          <option key={t.name} value={`${s.server}/${t.name}`} disabled={!!t.error}>
                            {t.name} -- {t.error ? `error: ${t.error}` : t.title}
                          </option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                )}
              </div>
              {chosenTask && (
                <div className="font-mono text-[11.5px] leading-relaxed text-ink-dim">
                  target <span className="text-ink">{chosenTask.target}</span> · action{' '}
                  <span className="text-ink">{chosenTask.action}</span> · score{' '}
                  <span className="text-ink">{chosenTask.score?.name}</span>{' '}
                  ({chosenTask.score?.higher_is_better === false ? 'lower' : 'higher'} is better) ·{' '}
                  {chosenTask.rows?.toLocaleString()} rows from {chosenTask.first?.slice(0, 10)} · holdout from{' '}
                  <span className="text-ink">{chosenTask.holdout_from?.slice(0, 10) ?? 'none'}</span>
                </div>
              )}
              {chosenTask && (
                <div className="space-y-1">
                  <button
                    type="button"
                    disabled={scan?.ref === taskRef && !scan.result && !scan.error}
                    onClick={() => {
                      const ref = taskRef
                      setScan({ ref })
                      objectives
                        .taskLeakScan(taskServer, taskName)
                        .then((result) => setScan({ ref, result }))
                        .catch((e: Error) => setScan({ ref, error: e.message }))
                    }}
                    className="rounded-md border border-seam px-2 py-0.5 font-mono text-[11px] text-ink-dim hover:border-ink-faint hover:text-ink disabled:opacity-40"
                    title="Columns whose change predicts the NEXT row's target move better than the current one were probably filed before they were known -- a leak no code test can catch."
                  >
                    {scan?.ref === taskRef && !scan.result && !scan.error ? 'checking data timing…' : 'check data timing'}
                  </button>
                  {scan?.ref === taskRef && scan.error && <div className="text-[11px] text-bad">{scan.error}</div>}
                  {scan?.ref === taskRef && scan.result && (
                    <div className="font-mono text-[11px] leading-relaxed">
                      {scan.result.suspects.length ? (
                        <span className="text-warn">
                          {scan.result.suspects.length} column(s) predict the NEXT move better than the current one --
                          probably filed before they were known; delay them (shift_rows) before trusting results:{' '}
                          {scan.result.suspects.join(', ')}
                        </span>
                      ) : (
                        <span className="text-good">no column looks like it knows the future</span>
                      )}
                      {scan.result.declared_ahead?.length ? (
                        <span className="text-ink-faint">
                          {' '}
                          · declared known in advance: {scan.result.declared_ahead.slice(0, 8).join(', ')}
                          {scan.result.declared_ahead.length > 8 ? '…' : ''}
                        </span>
                      ) : null}
                    </div>
                  )}
                </div>
              )}
              {projectServer && (
                <div className="text-[11px] text-ink-faint">
                  This project&apos;s data/action MCP is <span className="font-mono text-ink">{projectServer}</span>
                  {taskServers && !taskServers.servers.some((s) => s.server === projectServer)
                    ? ' -- it did not answer (is it running and signed in on the Connectors page?)'
                    : ''}
                  .
                </div>
              )}
              {taskServers?.errors.length ? (
                <div className="text-[11px] text-warn">{taskServers.errors.join(' · ')}</div>
              ) : null}
              <div className="text-[11px] leading-relaxed text-ink-faint">
                The task server serves the rows, defines what an action means and scores the actions; candidates read
                ft.rows() and report ft.report_actions(). The split, the look-ahead cuts and the task description come
                from the server.
              </div>
              <label className="flex items-center gap-2 text-[12.5px] text-ink-dim">
                <input type="checkbox" checked={lookahead} onChange={(e) => setLookahead(e.target.checked)} />
                Look-ahead test (re-run on rows cut at the holdout, mid in-sample and after the candidate&apos;s own action
                changes)
              </label>
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
                      <label className={label}>Direction</label>
                      <select className={field} value={direction} onChange={(e) => setDirection(e.target.value as Direction)}>
                        <option value="both">long and short</option>
                        <option value="long">long only</option>
                        <option value="short">short only</option>
                      </select>
                    </div>
                    <div>
                      <label className={label}>Holding</label>
                      <select className={field} value={intraday ? 'intraday' : 'overnight'} onChange={(e) => setIntraday(e.target.value === 'intraday')}>
                        <option value="intraday">intraday only (flat at each day's close)</option>
                        <option value="overnight">may hold overnight</option>
                      </select>
                    </div>
                    {direction === 'both' && (
                      <div>
                        <label className={label} title="Longs and shorts must each be at least this share of a candidate's in-sample trades for it to be ranked.">
                          Min share per side
                        </label>
                        <select className={field} value={sideShare} onChange={(e) => setSideShare(Number(e.target.value))}>
                          <option value={0}>no requirement</option>
                          <option value={0.1}>10% of trades</option>
                          <option value={0.2}>20% of trades</option>
                          <option value={0.3}>30% of trades</option>
                        </select>
                      </div>
                    )}
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
