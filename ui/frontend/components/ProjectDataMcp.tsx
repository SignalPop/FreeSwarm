'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { projectResources as res, projects, type ExportJob, type McpTask, type Project } from '@/lib/projects'
import { Pill } from '@/components/ui'

const fmtNum = (n: number | undefined | null) => (n == null ? '—' : n.toLocaleString())
const fmtStep = (s: number | undefined | null) =>
  s == null ? '—' : s < 60 ? `${s} s` : s < 3600 ? `${+(s / 60).toFixed(1)} min` : s < 86400 ? `${+(s / 3600).toFixed(1)} h` : `${+(s / 86400).toFixed(1)} d`

/**
 * What the project's data/action MCP offers, as the server describes it: for each task the shape
 * of its data, the target column (switchable to the server's target_options -- a project setting
 * every new objective of this task is valued on), what an action means, and the value function.
 */
export default function ProjectDataMcp({ project, onChanged }: { project: Project; onChanged: () => void | Promise<void> }) {
  const [data, setData] = useState<{ tasks: McpTask[]; errors: string[] } | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [saving, setSaving] = useState<string | null>(null)
  const [reload, setReload] = useState(0)          // bumped after a source refresh: describe again
  const ready = project.task_server_status === 'ready'
  const optionsKey = JSON.stringify(project.task_options ?? {})

  useEffect(() => {
    if (!project.task_server || !ready) return
    let live = true
    setData(null)
    setErr(null)
    projects
      .dataMcp(project.id)
      .then((d) => live && setData(d))
      .catch((e: Error) => live && setErr(e.message))
    return () => {
      live = false
    }
  }, [project.id, project.task_server, ready, optionsKey, reload])

  async function setOption(
    task: string,
    change: { target?: string; value_function?: string; action_rule?: string; direction?: string; source?: string },
  ) {
    setSaving(task)
    try {
      // Only the change: the server merges it per task and setting, so quick successive picks
      // (even on different tasks) cannot overwrite each other from a stale copy.
      await projects.update(project.id, { task_options: { [task]: change } })
      await onChanged()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(null)
    }
  }

  if (!project.task_server || !ready) return null
  if (err) return <div className="mt-4 text-[12px] text-bad">✗ {err}</div>
  if (!data) return <div className="mt-4 text-[12px] text-ink-faint">Asking {project.task_server} what it offers…</div>

  return (
    <div className="mt-5 space-y-4">
      {data.errors.length > 0 && <div className="text-[11.5px] text-warn">{data.errors.join(' · ')}</div>}
      {data.tasks.map((t) => {
        const sh = t.shape ?? {}
        const val = t.valuation ?? {}
        const extra = Object.entries(val).filter(
          ([k, v]) => !['summary', 'target_kind'].includes(k) && ['string', 'number', 'boolean'].includes(typeof v),
        )
        return (
          <div key={t.name} className="rounded-xl border border-seam bg-panel-hi/30 p-4">
            <div className="flex flex-wrap items-baseline gap-x-3">
              <span className="font-mono text-[13px] text-ink">{t.name}</span>
              <span className="text-[12.5px] text-ink-dim">{t.title}</span>
            </div>
            {t.description && <p className="mt-1.5 text-[12px] leading-relaxed text-ink-faint">{t.description}</p>}

            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <Block title="Data">
                {fmtNum(sh.rows)} rows × {fmtNum(sh.columns)} columns · every {fmtStep(sh.step_s)}
                {sh.days ? ` · ${fmtNum(sh.days)} days` : ''}
                <br />
                {sh.first?.slice(0, 16).replace('T', ' ')} → {sh.last?.slice(0, 16).replace('T', ' ')} UTC
                <br />
                holdout from {t.holdout_from?.slice(0, 10) ?? '—'} ({fmtNum(t.in_sample_rows)} rows in-sample)
                {sh.delayed_columns?.rows ? (
                  <>
                    <br />
                    <span className="text-ink-faint">
                      columns delayed {sh.delayed_columns.rows} rows (all but {sh.delayed_columns.except.join(', ')}) -- filed
                      before they were known
                    </span>
                  </>
                ) : null}
              </Block>
              <Block title="Target">
                <select
                  className="rounded-md border border-seam bg-panel-hi px-2 py-1 font-mono text-[12px] text-ink outline-none focus:border-accent"
                  value={t.target}
                  disabled={saving === t.name || (t.target_options ?? []).length < 2}
                  onChange={(e) => setOption(t.name, { target: e.target.value })}
                >
                  {(t.target_options?.length ? t.target_options : [t.target]).map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
                <div className="mt-1 text-ink-faint">
                  {(t.target_options ?? []).length < 2
                    ? 'the only target this task offers'
                    : `${t.target_options?.length} fields offered -- what new objectives of this task are valued on (a project setting)`}
                </div>
                {t.valuation?.target_kind ? <div className="mt-1 text-ink-faint">valued as {String(t.valuation.target_kind)}</div> : null}
                {t.columns?.find((c) => c.name === t.target)?.description && (
                  <div className="mt-1 text-ink-faint">{t.columns.find((c) => c.name === t.target)?.description}</div>
                )}
              </Block>
            </div>

            {/* The project's settings for this task, as cards: new objectives are scored under them. */}
            <div className="mt-4 space-y-3">
              {(t.sources ?? []).length > 1 && (
                <Choices
                  title="Data source"
                  hint="which rows new objectives of this task are built and scored on -- existing objectives keep theirs"
                  options={t.sources ?? []}
                  value={t.source ?? t.sources?.[0]?.name}
                  disabled={saving === t.name}
                  onPick={(v) => setOption(t.name, { source: v })}
                />
              )}
              <SourceRefresh project={project} task={t} onRefreshed={() => setReload((n) => n + 1)} />
              {(t.value_functions ?? []).length > 0 && (
                <Choices
                  title="Value function"
                  hint="what ranks this task's objectives -- the weaker of in-sample and holdout counts"
                  options={(t.value_functions ?? []).map((f) => ({
                    ...f,
                    note: `(${f.higher_is_better === false ? 'lower' : 'higher'} is better)`,
                  }))}
                  value={t.value_function ?? t.value_functions?.[0]?.name}
                  disabled={saving === t.name}
                  onPick={(v) => setOption(t.name, { value_function: v })}
                />
              )}
              {(t.action_rules ?? []).length > 0 && (
                <Choices
                  title="Holding"
                  hint="how long a trade may be held"
                  options={t.action_rules ?? []}
                  value={t.action_rule ?? t.action_rules?.[0]?.name}
                  disabled={saving === t.name}
                  onPick={(v) => setOption(t.name, { action_rule: v })}
                />
              )}
              {(t.directions ?? []).length > 0 && (
                <Choices
                  title="Direction"
                  hint="which sides trades may take"
                  options={t.directions ?? []}
                  value={t.direction ?? t.directions?.[0]?.name}
                  disabled={saving === t.name}
                  onPick={(v) => setOption(t.name, { direction: v })}
                />
              )}
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <Block title="Actions">
                {t.action?.description}
                {t.action?.min != null || t.action?.max != null ? (
                  <div className="mt-1 text-ink-faint">
                    bounds {t.action?.min ?? '−∞'} .. {t.action?.max ?? '∞'}
                    {t.action?.mode ? ` · mode ${t.action.mode}` : ''}
                  </div>
                ) : null}
              </Block>
              <Block title="Valuation">
                {val.summary}
                {extra.length > 0 && (
                  <div className="mt-1 text-ink-faint">
                    {extra.map(([k, v]) => `${k.replaceAll('_', ' ')} ${String(v)}`).join(' · ')}
                  </div>
                )}
              </Block>
            </div>
            <Guidance task={t} />
            <Schema task={t} />
          </div>
        )
      })}
    </div>
  )
}

/** The chosen data source's snapshot (what the server's export holds) and, for a source the
 *  server exports from a database, a refresh: the backend exports the table again with the source's
 *  own filter through this project's SQL login and the server swaps it in. The snapshot is the
 *  server's, shared by every project on it, so the backend refuses while an objective uses it. */
function SourceRefresh({ project, task, onRefreshed }: { project: Project; task: McpTask; onRefreshed: () => void }) {
  const src = (task.sources ?? []).find((s) => s.name === (task.source ?? task.sources?.[0]?.name))
  const [job, setJob] = useState<ExportJob | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const onRefreshedRef = useRef(onRefreshed)
  onRefreshedRef.current = onRefreshed
  const prevState = useRef<string | null>(null)

  const poll = useCallback(async () => {
    if (!src) return
    try {
      const r = await res.sqlExports(project.id)
      const j = r.jobs.find((x) => x.kind === 'source_refresh' && x.source === src.name) ?? null
      if (j && j.state === 'done' && prevState.current === 'running') onRefreshedRef.current()
      prevState.current = j?.state ?? null
      setJob(j)
    } catch {
      /* a failed poll is not worth an error banner */
    }
  }, [project.id, src])

  useEffect(() => {
    void poll()
  }, [poll])
  const running = job?.state === 'running'
  useEffect(() => {
    if (!running) return
    const t = setInterval(poll, 2000)
    return () => clearInterval(t)
  }, [running, poll])

  if (!src || (!src.snapshot && !src.refresh)) return null
  const snap = src.snapshot
  const canRefresh = !!src.refresh && !!project.sql

  async function start() {
    if (!src?.refresh) return
    if (
      !window.confirm(
        `Export ${src.refresh.table} where ${src.refresh.where} from ${src.refresh.server}/${src.refresh.database} and ` +
          `replace ${task.name}'s ${src.title ?? src.name} snapshot? Every project on ${project.task_server} uses it; ` +
          'the current copy is kept as the previous one.',
      )
    )
      return
    setErr(null)
    try {
      const j = await res.refreshSource(project.id, src.name, task.name)
      prevState.current = j.state
      setJob(j)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const pct = job?.total ? Math.min(100, (job.rows / job.total) * 100) : null
  return (
    <div className="rounded-lg border border-seam px-3 py-2 font-mono text-[11px] text-ink-dim">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-ink-faint">{src.title ?? src.name} snapshot</span>
        {snap ? (
          <span>
            {snap.rows?.toLocaleString() ?? '?'} rows
            {snap.exported_at ? ` · exported ${new Date(snap.exported_at * 1000).toISOString().slice(0, 16).replace('T', ' ')}` : ''}
            {snap.where ? <span className="text-ink-faint"> · where {snap.where}</span> : null}
          </span>
        ) : (
          <span className="text-warn">no export manifest</span>
        )}
        {src.refresh && (
          <button
            type="button"
            onClick={start}
            disabled={!canRefresh || running}
            title={
              project.sql
                ? `export ${src.refresh.table} where ${src.refresh.where} again and swap it in`
                : 'set up SQL Server for this project first (it exports through the project login)'
            }
            className="ml-auto rounded border border-seam px-2 py-0.5 text-ink hover:border-accent disabled:cursor-not-allowed disabled:opacity-40"
          >
            {running ? `${job?.phase ?? 'working'}…` : 'refresh from database'}
          </button>
        )}
      </div>
      {job && (
        <div className="mt-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <Pill tone={job.state === 'done' ? 'good' : job.state === 'failed' ? 'bad' : 'warn'} pulse={running}>
              {running ? job.phase : job.state}
            </Pill>
            <span className="text-ink-faint">
              {job.state === 'done' && job.result
                ? `${job.result.rows?.toLocaleString()} rows · ${job.result.days} days · ${job.result.first?.slice(0, 10)} → ${job.result.last?.slice(0, 10)}`
                : `${job.rows.toLocaleString()}${job.total != null ? ` / ${job.total.toLocaleString()}` : ''} rows`}
              {job.finished_at ? ` · ${Math.round(job.finished_at - job.started_at)}s` : ''}
            </span>
          </div>
          {running && job.phase === 'exporting' && pct != null && (
            <div className="mt-1.5 h-1 overflow-hidden rounded bg-seam">
              <div className="h-full bg-accent transition-all" style={{ width: `${pct}%` }} />
            </div>
          )}
          {job.audit?.summary && (
            <div className={`mt-1 ${job.audit.status === 'bad' ? 'text-bad' : job.audit.status === 'warn' ? 'text-warn' : 'text-good'}`}>
              audit: {job.audit.summary}
            </div>
          )}
          {job.error && <div className="mt-1 text-[11.5px] text-bad">{job.error}</div>}
        </div>
      )}
      {err && <div className="mt-1 text-[11.5px] text-bad">{err}</div>}
    </div>
  )
}

/** The advice the MCP gives the agents, as it will appear in their brief (it follows the settings above). */
function Guidance({ task }: { task: McpTask }) {
  const [open, setOpen] = useState(false)
  const g = task.guidance ?? []
  if (!g.length) return null
  return (
    <div className="mt-4">
      <button type="button" onClick={() => setOpen(!open)} className="font-mono text-[11.5px] text-accent hover:underline">
        {open ? '▾' : '▸'} agent guidance from the MCP -- {g.length} section{g.length === 1 ? '' : 's'}
      </button>
      {open && (
        <div className="mt-2 space-y-1.5 text-[12px] leading-snug text-ink-dim">
          {g.map((s, k) => (
            <div key={k}>
              <span className="font-mono text-[11px] uppercase text-ink">{s.title}</span> -- {s.text}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/** Every column the MCP serves: name, type, role, what it is -- filterable. */
function Schema({ task }: { task: McpTask }) {
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const cols = task.columns ?? []
  const shown = cols.filter(
    (c) => !q || c.name.toLowerCase().includes(q.toLowerCase()) || (c.description ?? '').toLowerCase().includes(q.toLowerCase()),
  )
  return (
    <div className="mt-4">
      <button type="button" onClick={() => setOpen(!open)} className="font-mono text-[11.5px] text-accent hover:underline">
        {open ? '▾' : '▸'} schema -- {cols.length} columns
      </button>
      {open && (
        <div className="mt-2">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="filter by name or description"
            className="mb-2 w-full rounded-md border border-seam bg-panel-hi px-2 py-1 font-mono text-[12px] text-ink outline-none focus:border-accent"
          />
          <div className="max-h-[420px] overflow-y-auto rounded-lg border border-seam/60">
            <table className="w-full font-mono text-[11px]">
              <thead className="sticky top-0 bg-panel text-ink-faint">
                <tr>
                  <th className="px-2 py-1 text-left font-normal">column</th>
                  <th className="px-2 py-1 text-left font-normal">type</th>
                  <th className="px-2 py-1 text-left font-normal">role</th>
                  <th className="px-2 py-1 text-left font-normal">what it is</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((c) => (
                  <tr key={c.name} className="border-t border-seam/40">
                    <td className={`px-2 py-1 ${c.name === task.target ? 'text-accent' : 'text-ink'}`}>{c.name}</td>
                    <td className="px-2 py-1 text-ink-faint">{c.dtype}</td>
                    <td className="px-2 py-1 text-ink-dim">{c.name === task.target ? 'target' : c.role}</td>
                    <td className="px-2 py-1 text-ink-faint">{c.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

/** One project setting as a row of cards (title, what it means); the chosen one is highlighted and
 *  a click saves the other at once. */
function Choices({
  title,
  hint,
  options,
  value,
  disabled,
  onPick,
}: {
  title: string
  hint: string
  options: { name: string; title?: string; description?: string; note?: string }[]
  value: string | undefined
  disabled: boolean
  onPick: (name: string) => void
}) {
  return (
    <div>
      <div className="mb-1.5 text-[10.5px] uppercase tracking-wide text-ink-faint">
        {title} <span className="normal-case tracking-normal">-- {hint} (a project setting)</span>
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {options.map((o) => (
          <button
            key={o.name}
            type="button"
            disabled={disabled}
            onClick={() => o.name !== value && onPick(o.name)}
            className={`rounded-lg border px-3 py-2 text-left transition-colors disabled:cursor-wait ${
              o.name === value ? 'border-accent bg-accent/10' : 'border-seam hover:border-ink-faint'
            }`}
          >
            <div className="text-[12.5px] text-ink">
              {o.title ?? o.name}
              {o.note && <span className="text-[11px] text-ink-faint"> {o.note}</span>}
            </div>
            {o.description && <div className="mt-0.5 text-[11px] leading-snug text-ink-faint">{o.description}</div>}
          </button>
        ))}
      </div>
    </div>
  )
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 text-[10.5px] uppercase tracking-wide text-ink-faint">{title}</div>
      <div className="font-mono text-[11.5px] leading-relaxed text-ink-dim">{children}</div>
    </div>
  )
}
