'use client'

import { useState } from 'react'
import type { QueryResult } from '@/lib/projects'
import { Button } from '@/components/ui'

/**
 * "Try it yourself" for a project data source: run the same kind of query an agent will,
 * with the same restrictions, and see the rows. The point is trust -- you can check what
 * the swarm can and cannot see before it runs on it.
 */
export default function QueryBox({
  run,
  placeholder,
  initial = '',
}: {
  run: (sql: string) => Promise<QueryResult>
  placeholder: string
  initial?: string
}) {
  const [sql, setSql] = useState(initial)
  const [res, setRes] = useState<QueryResult | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function go() {
    if (!sql.trim()) return
    setBusy(true)
    setErr(null)
    try {
      setRes(await run(sql))
    } catch (e) {
      setRes(null)
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-2">
      <textarea
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault()
            void go()
          }
        }}
        placeholder={placeholder}
        spellCheck={false}
        className="h-20 w-full resize-y rounded-lg border border-seam bg-canvas p-2 font-mono text-[12px] text-ink outline-none focus:border-accent/50"
      />
      <div className="flex items-center gap-3">
        <Button tone="default" onClick={go} disabled={busy || !sql.trim()}>
          {busy ? 'running…' : 'Run query'}
        </Button>
        <span className="font-mono text-[10.5px] text-ink-faint">
          Ctrl+Enter · read-only · same limits the agents get
        </span>
        {res && (
          <span className="ml-auto font-mono text-[10.5px] text-ink-faint">
            {res.row_count} row{res.row_count === 1 ? '' : 's'}
            {res.truncated ? ' (truncated)' : ''} · {res.seconds}s
          </span>
        )}
      </div>
      {err && (
        <div className="rounded-lg border border-bad/40 bg-bad/[0.07] px-3 py-2 font-mono text-[11.5px] text-ink-dim">
          {err}
        </div>
      )}
      {res && res.columns.length > 0 && (
        <div className="max-h-72 overflow-auto rounded-lg border border-seam">
          <table className="w-full border-collapse font-mono text-[11px]">
            <thead className="sticky top-0 bg-panel-hi">
              <tr>
                {res.columns.map((c) => (
                  <th key={c} className="border-b border-seam px-2 py-1.5 text-left font-medium text-ink">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {res.rows.map((r, i) => (
                <tr key={i} className="odd:bg-panel-hi/30">
                  {r.map((v, j) => (
                    <td key={j} className="whitespace-nowrap border-b border-seam/40 px-2 py-1 text-ink-dim">
                      {v === null ? <span className="text-ink-faint">NULL</span> : String(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
