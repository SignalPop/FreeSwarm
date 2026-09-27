'use client'

import { useEffect, useState } from 'react'
import { projects, type McpTask, type Project } from '@/lib/projects'

const fmtNum = (n: number | undefined | null) => (n == null ? '—' : n.toLocaleString())
const fmtStep = (s: number | undefined | null) =>
  s == null ? '—' : s < 60 ? `${s} s` : s < 3600 ? `${+(s / 60).toFixed(1)} min` : s < 86400 ? `${+(s / 3600).toFixed(1)} h` : `${+(s / 86400).toFixed(1)} d`

/**
 * What the project's data/action MCP offers, as the server describes it: for each task the shape
 * of its data, the target column (switchable to the server's target_options -- a project setting
 * every new objective of this task is valued on), what an action means, and the value function.
 */
export default function ProjectDataMcp({ project, onChanged }: { project: Project; onChanged: () => void }) {
  const [data, setData] = useState<{ tasks: McpTask[]; errors: string[] } | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [saving, setSaving] = useState<string | null>(null)
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
  }, [project.id, project.task_server, ready, optionsKey])

  async function setTarget(task: string, target: string) {
    setSaving(task)
    try {
      const opts = { ...(project.task_options ?? {}), [task]: { target } }
      await projects.update(project.id, { task_options: opts })
      onChanged()
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
        const extra = Object.entries(val).filter(([k, v]) => k !== 'summary' && ['string', 'number', 'boolean'].includes(typeof v))
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
                  onChange={(e) => setTarget(t.name, e.target.value)}
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
              <Block title="Actions">
                {t.action?.description}
                {t.action?.min != null || t.action?.max != null ? (
                  <div className="mt-1 text-ink-faint">
                    bounds {t.action?.min ?? '−∞'} .. {t.action?.max ?? '∞'}
                    {t.action?.mode ? ` · mode ${t.action.mode}` : ''}
                  </div>
                ) : null}
              </Block>
              <Block title="Value function">
                {val.summary ?? `${t.score?.name} (${t.score?.higher_is_better === false ? 'lower' : 'higher'} is better)`}
                {extra.length > 0 && (
                  <div className="mt-1 text-ink-faint">
                    {extra.map(([k, v]) => `${k.replaceAll('_', ' ')} ${String(v)}`).join(' · ')}
                  </div>
                )}
              </Block>
            </div>
            <Schema task={t} />
          </div>
        )
      })}
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

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 text-[10.5px] uppercase tracking-wide text-ink-faint">{title}</div>
      <div className="font-mono text-[11.5px] leading-relaxed text-ink-dim">{children}</div>
    </div>
  )
}
