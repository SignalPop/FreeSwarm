'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { projectResources as res, type ExportJob, type Project } from '@/lib/projects'
import { Button, Pill } from '@/components/ui'

/**
 * Snapshot an allowed SQL table into parquet inside the project's data folder.
 *
 * Why this is worth having beside live SQL: agents then query a local file set with DuckDB --
 * fast, and without putting load on the SQL Server while a swarm iterates -- and the snapshot
 * stays fixed while the table keeps changing, so an analysis is reproducible.
 *
 * Runs through the project's read-only login, so only tables it may already read can be
 * exported. The result is one dataset (folder of parts) that agents see as one table.
 */
export default function SqlExport({
  project,
  onExported,
}: {
  project: Project
  onExported: () => void
}) {
  const tables = project.sql?.tables ?? []
  const [table, setTable] = useState(tables[0] ?? '')
  const [folder, setFolder] = useState('sql_exports')
  const [where, setWhere] = useState('')
  const [perFile, setPerFile] = useState(1_000_000)
  const [jobs, setJobs] = useState<ExportJob[]>([])
  const [err, setErr] = useState<string | null>(null)
  // Held in a ref, not a dependency: a caller passing an inline function would otherwise make
  // `refresh` a new function every render, re-running the effect below in a fetch loop.
  const onExportedRef = useRef(onExported)
  onExportedRef.current = onExported

  // The previous poll's jobs, to spot a running -> done transition. Kept in a ref and compared
  // HERE rather than inside a setJobs(prev => ...) updater: React may run updaters during
  // render, and notifying the parent from one is a setState on ProjectsPage mid-render.
  const jobsRef = useRef<ExportJob[]>([])

  const refresh = useCallback(async () => {
    try {
      const r = await res.sqlExports(project.id)
      const prev = jobsRef.current
      const finished = r.jobs.some(
        (j) => j.state === 'done' && prev.find((p) => p.id === j.id)?.state === 'running',
      )
      jobsRef.current = r.jobs
      setJobs(r.jobs)
      // Tell the data panel the moment an export finishes, so its table appears there.
      if (finished) onExportedRef.current()
    } catch {
      /* the list is a convenience; a failed poll is not worth an error banner */
    }
  }, [project.id])

  useEffect(() => {
    setTable(project.sql?.tables[0] ?? '')
    void refresh()
  }, [project.id, project.sql, refresh])

  const running = jobs.some((j) => j.state === 'running')
  useEffect(() => {
    if (!running) return
    const t = setInterval(refresh, 1000)
    return () => clearInterval(t)
  }, [running, refresh])

  async function start() {
    setErr(null)
    try {
      await res.sqlExport(project.id, {
        table,
        folder,
        where: where.trim() || undefined,
        rows_per_file: perFile,
      })
      await refresh()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const leaf = table.replace(/\./g, '_')

  return (
    <div className="mt-4 space-y-3 border-t border-seam pt-4">
      <div className="text-[13px] font-medium text-ink">Export a table to parquet</div>
      <p className="text-[12px] leading-relaxed text-ink-faint">
        Snapshots the table into the data folder, where agents query it locally with DuckDB — fast,
        no load on the server, and fixed while the live table keeps changing.
      </p>
      <div className="grid gap-2 sm:grid-cols-[1fr_1fr]">
        <label className="text-[11px] text-ink-faint">
          table
          <select
            value={table}
            onChange={(e) => setTable(e.target.value)}
            className="mt-1 w-full rounded-lg border border-seam bg-panel-hi px-2 py-1.5 font-mono text-[12px] text-ink outline-none"
          >
            {tables.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <label className="text-[11px] text-ink-faint">
          folder in the data directory
          <input
            value={folder}
            onChange={(e) => setFolder(e.target.value)}
            className="mt-1 w-full rounded-lg border border-seam bg-canvas px-2 py-1.5 font-mono text-[12px] text-ink outline-none"
          />
        </label>
      </div>
      <label className="block text-[11px] text-ink-faint">
        optional filter (T-SQL WHERE clause, without the word WHERE)
        <input
          value={where}
          onChange={(e) => setWhere(e.target.value)}
          placeholder="TradeDate >= '2024-01-01'"
          spellCheck={false}
          className="mt-1 w-full rounded-lg border border-seam bg-canvas px-2 py-1.5 font-mono text-[12px] text-ink outline-none"
        />
      </label>
      <div className="flex flex-wrap items-center gap-3">
        <label className="font-mono text-[11px] text-ink-faint">
          rows per file{' '}
          <input
            type="number"
            min={10000}
            step={100000}
            value={perFile}
            onChange={(e) => setPerFile(Number(e.target.value))}
            className="ml-1 w-32 rounded border border-seam bg-canvas px-2 py-1 text-ink"
          />
        </label>
        <Button tone="primary" onClick={start} disabled={!table || !folder.trim() || running}>
          {running ? 'exporting…' : 'Export'}
        </Button>
        {table && (
          <span className="font-mono text-[10.5px] text-ink-faint">
            → {folder.trim() || 'sql_exports'}/{leaf}/part-*.parquet
          </span>
        )}
      </div>
      {err && (
        <div className="rounded-lg border border-bad/40 bg-bad/[0.07] px-3 py-2 text-[12px] text-ink-dim">{err}</div>
      )}

      {jobs.length > 0 && (
        <div className="space-y-2">
          {jobs.slice(0, 5).map((j) => {
            const pct = j.total ? Math.min(100, (j.rows / j.total) * 100) : null
            return (
              <div key={j.id} className="rounded-lg border border-seam bg-panel-hi/40 p-2.5">
                <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
                  <span className="text-ink">{j.table}</span>
                  <span className="text-ink-faint">→ {j.relative}</span>
                  <span className="ml-auto">
                    <Pill tone={j.state === 'done' ? 'good' : j.state === 'failed' ? 'bad' : 'warn'} pulse={j.state === 'running'}>
                      {j.state}
                    </Pill>
                  </span>
                </div>
                <div className="mt-1.5 font-mono text-[10.5px] text-ink-faint">
                  {j.rows.toLocaleString()}
                  {j.total != null ? ` / ${j.total.toLocaleString()}` : ''} rows · {j.files} file
                  {j.files === 1 ? '' : 's'}
                  {j.finished_at ? ` · ${(j.finished_at - j.started_at).toFixed(1)}s` : ''}
                  {j.where ? ` · where ${j.where}` : ''}
                </div>
                {j.state === 'running' && pct != null && (
                  <div className="mt-1.5 h-1 overflow-hidden rounded bg-seam">
                    <div className="h-full bg-accent transition-all" style={{ width: `${pct}%` }} />
                  </div>
                )}
                {j.error && <div className="mt-1 text-[11.5px] text-bad">{j.error}</div>}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
