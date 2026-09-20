'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { projectResources as res, type DataFile, type Project } from '@/lib/projects'
import { bytesLabel } from '@/lib/format'
import { Panel, Pill } from '@/components/ui'
import QueryBox from '@/components/QueryBox'

/**
 * Which LOADED models this project's swarm may use.
 *
 * Only loaded models are offered. Agents are only ever created for loaded models, so ticking
 * an unloaded one did nothing visible and read as broken. The list follows the Models page
 * live: load a model there and it appears here within a few seconds.
 *
 * "All loaded models" is the default. Unticking narrows the project to an explicit list.
 * Entries for models that are not loaded right now are kept in that list, untouched, rather
 * than silently dropped: unloading a model to free memory must not quietly change what this
 * project may use when it comes back.
 */
export function ProjectModels({ project, onChanged }: { project: Project; onChanged: () => void }) {
  const [llms, setLlms] = useState<string[]>([])
  const [forecasters, setForecasters] = useState<string[]>([])
  const [ready, setReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const [e, t] = await Promise.all([api.engines(), api.tsInstances().catch(() => ({ instances: [] }))])
        if (!alive) return
        setLlms(
          [...new Set(e.loaded.filter((m) => m.ready && m.model).map((m) => m.model as string))].sort(),
        )
        setForecasters(
          [...new Set(t.instances.filter((i) => i.state === 'running').map((i) => i.model_id))].sort(),
        )
        setReady(true)
      } catch {
        if (alive) setReady(true)
      }
    }
    void load()
    const timer = setInterval(load, 5000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])

  const loaded = [...llms, ...forecasters]
  const all = project.models === null || project.models === undefined
  const chosen = new Set(project.models ?? [])
  // Allowed in the saved list but not loaded right now: kept, and mentioned, never lost.
  const dormant = all ? [] : (project.models ?? []).filter((m) => !loaded.includes(m))

  async function save(next: string[] | null) {
    setBusy(true)
    setErr(null)
    try {
      await res.setModels(project.id, next)
      onChanged()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  function toggle(id: string) {
    // From "all": start from every loaded model. From a list: keep its dormant entries.
    const base = all ? new Set(loaded) : new Set(chosen)
    if (base.has(id)) base.delete(id)
    else base.add(id)
    void save([...base])
  }

  const groups: [string, string[]][] = [
    ['LLMs — become agents', llms],
    ['Time series — offered to agents as the forecast tool', forecasters],
  ]

  return (
    <Panel className="p-5">
      <div className="flex items-center gap-3">
        <span className="text-[15px] font-medium text-ink">Models</span>
        <label className="ml-auto flex cursor-pointer items-center gap-2 text-[12px] text-ink-dim">
          <input
            type="checkbox"
            checked={all}
            disabled={busy || loaded.length === 0}
            onChange={() => save(all ? loaded : null)}
          />
          all loaded models
        </label>
      </div>
      <p className="mt-1 text-[12px] text-ink-faint">
        Only loaded models can be chosen. Each ticked LLM becomes an agent; each ticked
        forecaster is offered to the agents as a tool.
      </p>

      {ready && loaded.length === 0 && (
        <div className="mt-3 text-[12px] text-ink-faint">
          No models are loaded.{' '}
          <Link href="/models" className="text-accent hover:opacity-80">
            Load one on the Models page
          </Link>{' '}
          and it will appear here.
        </div>
      )}

      <div className="mt-3 space-y-3">
        {groups.map(([label, list]) =>
          list.length ? (
            <div key={label}>
              <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">{label}</div>
              <div className="flex flex-wrap gap-2">
                {list.map((id) => {
                  const on = all || chosen.has(id)
                  const remote = id.includes('@')
                  return (
                    <button
                      key={id}
                      onClick={() => toggle(id)}
                      disabled={busy}
                      title={remote ? `served by ${id.slice(id.lastIndexOf('@') + 1)} over the network` : undefined}
                      className={`rounded-lg border px-2.5 py-1 font-mono text-[11.5px] transition-colors ${
                        on
                          ? remote
                            ? 'border-remote/60 bg-remote/10 text-remote'
                            : 'border-accent/50 bg-accent/[0.08] text-ink'
                          : remote
                            ? 'border-remote/30 text-remote/60 hover:text-remote'
                            : 'border-seam text-ink-faint hover:text-ink-dim'
                      }`}
                    >
                      {on ? '✓ ' : ''}
                      {id}
                    </button>
                  )
                })}
              </div>
            </div>
          ) : null,
        )}
      </div>

      {dormant.length > 0 && (
        <div className="mt-3 font-mono text-[10.5px] text-ink-faint">
          also allowed once loaded: {dormant.join(', ')}
        </div>
      )}
      {err && <div className="mt-2 text-[12px] text-bad">{err}</div>}
    </Panel>
  )
}

/**
 * The data folder as the agents see it: every parquet/CSV/JSON file becomes a SQL view, and
 * you can query them here with exactly the confinement the agents get.
 */
export function ProjectDataQuery({ project, refreshKey = 0 }: { project: Project; refreshKey?: number }) {
  const [files, setFiles] = useState<DataFile[] | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    setFiles(null)
    res
      .dataCatalog(project.id)
      .then((r) => alive && setFiles(r.files))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [project.id, project.data_dir, refreshKey])

  const first = files?.[0]

  return (
    <Panel className="p-5">
      <div className="flex items-center gap-3">
        <span className="text-[15px] font-medium text-ink">Data for the swarm</span>
        {files && <Pill tone={files.length ? 'good' : 'neutral'}>{files.length} queryable file{files.length === 1 ? '' : 's'}</Pill>}
      </div>
      <p className="mt-1 text-[12px] leading-relaxed text-ink-faint">
        Point the data directory above at a drive or folder of <strong className="text-ink-dim">parquet, CSV or JSON</strong> and
        agents can query it in SQL — in place, nothing is imported. They cannot open anything outside
        this folder, and cannot write to it.
      </p>
      {err && <div className="mt-2 text-[12px] text-bad">{err}</div>}
      {files && files.length > 0 && (
        <>
          <div className="mt-3 max-h-40 overflow-y-auto rounded-lg border border-seam">
            {files.map((f) => (
              <div key={f.path} className="flex items-center gap-3 border-b border-seam/40 px-3 py-1 font-mono text-[11px] last:border-0">
                <span className="text-accent">{f.view}</span>
                <span className="truncate text-ink-faint">
                  {f.dataset ? `${f.source ?? 'SQL export'} · ${(f.rows ?? 0).toLocaleString()} rows` : f.path}
                </span>
                <span className="ml-auto text-ink-faint">{bytesLabel(f.bytes)}</span>
              </div>
            ))}
          </div>
          <div className="mt-3">
            <QueryBox
              run={(sql) => res.dataQuery(project.id, sql)}
              placeholder={first ? `SELECT * FROM ${first.view} LIMIT 10` : 'SELECT ...'}
              initial={first ? `SELECT * FROM ${first.view} LIMIT 10` : ''}
            />
          </div>
        </>
      )}
      {files && files.length === 0 && (
        <div className="mt-3 text-[12px] text-ink-faint">
          No parquet, CSV or JSON files in {project.data_dir}. Change the data directory above, or upload files below.
        </div>
      )}
    </Panel>
  )
}

