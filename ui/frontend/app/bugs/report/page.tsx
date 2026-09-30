'use client'

import { useEffect, useState } from 'react'
import { bugsApi, PRIORITIES, SEVERITIES, STATUSES, type Bug, type BugStatus } from '@/lib/bugs'

/**
 * Every bug as a printable report: the Bugs page's "Export PDF" opens this with ?print=1 and the
 * browser's print dialog saves it as a PDF ("Save as PDF" / "Microsoft Print to PDF"). Light on
 * white whatever the console theme, the sidebar hidden, each bug starting cleanly on the page.
 *
 * Query: status=open|pending|closed|all (default all), print=1 opens the print dialog when ready.
 */

const MAX_BLOCK = 6000 // characters of script / evidence per bug: a PDF is for reading, not a dump

function when(ts: number | null | undefined): string {
  return ts ? new Date(ts * 1000).toLocaleString(undefined, { hour12: false }) : '—'
}

function clip(s: string): string {
  return s.length <= MAX_BLOCK ? s : `${s.slice(0, MAX_BLOCK / 2)}\n… (${s.length - MAX_BLOCK} characters cut) …\n${s.slice(-MAX_BLOCK / 2)}`
}

const SEV_COLOR: Record<string, string> = { critical: '#b42318', high: '#c4320a', medium: '#b54708', low: '#475467' }
const PRI_COLOR: Record<string, string> = { P1: '#b42318', P2: '#b54708', P3: '#344054', P4: '#667085' }
const STATUS_COLOR: Record<string, string> = { open: '#b42318', pending: '#b54708', closed: '#067647' }

function Chip({ text, color }: { text: string; color: string }) {
  return (
    <span
      className="inline-block rounded border px-1.5 py-px font-mono text-[10px] leading-tight"
      style={{ color, borderColor: color }}
    >
      {text}
    </span>
  )
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-3">
      <div className="mb-1 font-mono text-[9.5px] uppercase tracking-[0.14em] text-[#667085]">{title}</div>
      {children}
    </div>
  )
}

function Pre({ text }: { text: string }) {
  return (
    <pre className="whitespace-pre-wrap break-all rounded border border-[#d0d5dd] bg-[#f9fafb] p-2 font-mono text-[9.5px] leading-snug text-[#1d2939]">
      {clip(text)}
    </pre>
  )
}

export default function BugReport() {
  const [bugs, setBugs] = useState<Bug[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [status, setStatus] = useState<BugStatus | 'all'>('all')
  const [loaded, setLoaded] = useState(0)
  const [total, setTotal] = useState(0)

  useEffect(() => {
    const qs = new URLSearchParams(window.location.search)
    const s = (qs.get('status') ?? 'all') as BugStatus | 'all'
    const autoPrint = qs.get('print') === '1'
    setStatus(s)
    let live = true
    ;(async () => {
      try {
        const { bugs: rows } = await bugsApi.list(s)
        setTotal(rows.length)
        // The list has summaries; the report needs each bug in full. A few at a time.
        const full: Bug[] = []
        for (let i = 0; i < rows.length; i += 6) {
          const batch = await Promise.all(rows.slice(i, i + 6).map((r) => bugsApi.get(r.id)))
          full.push(...batch)
          if (live) setLoaded(full.length)
        }
        if (!live) return
        setBugs(full)
        document.title = `FreeSwarm bug report — ${new Date().toLocaleDateString()}`
        if (autoPrint) setTimeout(() => window.print(), 400)
      } catch (e) {
        if (live) setErr(e instanceof Error ? e.message : String(e))
      }
    })()
    return () => {
      live = false
    }
  }, [])

  const count = (key: 'status' | 'priority' | 'severity', v: string) => (bugs ?? []).filter((b) => b[key] === v).length

  return (
    <div className="bug-report min-h-full bg-white px-10 py-8 text-[#101828]">
      {/* Print: hide the console's sidebar and header, and let the page grow past one screen
          (the shell is a fixed-height, scrolling layout). */}
      <style>{`
        @page { size: A4; margin: 14mm 12mm; }
        @media print {
          aside, header { display: none !important; }
          html, body, body > div, .h-screen { height: auto !important; overflow: visible !important; background: #fff !important; }
          main { overflow: visible !important; }
          .no-print { display: none !important; }
          .bug-report { padding: 0 !important; }
          .bug-head { break-after: avoid; break-inside: avoid; }
          .summary, table tr { break-inside: avoid; }
          * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
        }
      `}</style>

      <div className="no-print mb-6 flex items-center gap-3 rounded-lg border border-[#d0d5dd] bg-[#f9fafb] px-4 py-2.5 text-[13px] text-[#344054]">
        <span className="flex-1">
          {bugs
            ? `${bugs.length} bug${bugs.length === 1 ? '' : 's'} ready. In the print dialog pick "Save as PDF" (or "Microsoft Print to PDF").`
            : err
              ? `Could not load the bugs: ${err}`
              : `Loading ${loaded}/${total || '…'} bugs…`}
        </span>
        <button
          onClick={() => window.print()}
          disabled={!bugs}
          className="rounded-md bg-[#101828] px-3 py-1.5 text-[12.5px] font-medium text-white disabled:opacity-40"
        >
          Save as PDF
        </button>
      </div>

      <h1 className="text-[24px] font-semibold tracking-tight">FreeSwarm bug report</h1>
      <div className="mt-1 text-[12px] text-[#475467]">
        {status === 'all' ? 'All bugs' : `${status[0].toUpperCase()}${status.slice(1)} bugs`} · generated{' '}
        {new Date().toLocaleString(undefined, { hour12: false })} · filed by the monitoring agent from the agents&apos;
        logs and the engines, and by hand
      </div>

      {bugs && (
        <>
          <div className="summary mt-5 grid grid-cols-3 gap-4 text-[11.5px]">
            {(
              [
                ['Status', 'status', STATUSES, STATUS_COLOR],
                ['Priority', 'priority', PRIORITIES, PRI_COLOR],
                ['Severity', 'severity', SEVERITIES, SEV_COLOR],
              ] as const
            ).map(([label, key, values, colors]) => (
              <div key={key} className="rounded-lg border border-[#d0d5dd] p-3">
                <div className="mb-1.5 font-mono text-[9.5px] uppercase tracking-[0.14em] text-[#667085]">{label}</div>
                {values.map((v) => (
                  <div key={v} className="flex justify-between">
                    <span style={{ color: (colors as Record<string, string>)[v] }}>{v}</span>
                    <span className="font-mono">{count(key, v)}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>

          <h2 className="mt-6 mb-2 text-[14px] font-semibold">Index</h2>
          <table className="w-full border-collapse text-[10.5px]">
            <thead>
              <tr className="border-b border-[#98a2b3] text-left font-mono text-[9px] uppercase tracking-[0.1em] text-[#667085]">
                <th className="py-1 pr-2">#</th>
                <th className="py-1 pr-2">Pri</th>
                <th className="py-1 pr-2">Severity</th>
                <th className="py-1 pr-2">Status</th>
                <th className="py-1 pr-2">Title</th>
                <th className="py-1 pr-2 text-right">Seen</th>
                <th className="py-1 text-right">Last</th>
              </tr>
            </thead>
            <tbody>
              {bugs.map((b) => (
                <tr key={b.id} className="border-b border-[#eaecf0] align-top">
                  <td className="py-1 pr-2 font-mono">{b.id}</td>
                  <td className="py-1 pr-2 font-mono" style={{ color: PRI_COLOR[b.priority] }}>
                    {b.priority}
                  </td>
                  <td className="py-1 pr-2" style={{ color: SEV_COLOR[b.severity] }}>
                    {b.severity}
                  </td>
                  <td className="py-1 pr-2" style={{ color: STATUS_COLOR[b.status] }}>
                    {b.status}
                    {b.closed_by === 'monitor' ? ' (auto)' : ''}
                  </td>
                  <td className="py-1 pr-2">{b.title}</td>
                  <td className="py-1 pr-2 text-right font-mono">{b.occurrences}×</td>
                  <td className="py-1 text-right font-mono whitespace-nowrap">{when(b.last_seen)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="mt-8">
            {bugs.map((b) => {
              const where = (
                [
                  ['Agent', b.agent],
                  ['Model', b.model],
                  ['Tool', b.tool],
                  ['Objective', b.objective_title],
                  ['Project', b.project_id],
                ] as [string, string | null][]
              ).filter(([, v]) => v)
              return (
                <section key={b.id} className="bug pt-6">
                  <div className="bug-head border-b border-[#98a2b3] pb-2">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="font-mono text-[11px] text-[#667085]">#{b.id}</span>
                      <Chip text={b.priority} color={PRI_COLOR[b.priority]} />
                      <Chip text={b.severity} color={SEV_COLOR[b.severity]} />
                      <Chip
                        text={`${b.status}${b.closed_by ? ` · by ${b.closed_by}` : ''}`}
                        color={STATUS_COLOR[b.status]}
                      />
                      <span className="font-mono text-[10px] text-[#667085]">
                        {b.category.replace('_', ' ')} · {b.source}
                      </span>
                    </div>
                    <h3 className="mt-1 text-[15px] font-semibold leading-snug">{b.title}</h3>
                    <div className="mt-0.5 font-mono text-[10px] text-[#475467]">
                      seen {b.occurrences}× · first {when(b.first_seen)} · last {when(b.last_seen)}
                      {b.closed_at ? ` · closed ${when(b.closed_at)}` : ''}
                    </div>
                  </div>

                  <Block title="Description">
                    <p className="whitespace-pre-wrap text-[11.5px] leading-relaxed">{b.description || '—'}</p>
                  </Block>
                  {b.suggestion && (
                    <Block title="Suggested fix">
                      <p className="whitespace-pre-wrap text-[11.5px] leading-relaxed">{b.suggestion}</p>
                    </Block>
                  )}
                  {where.length > 0 && (
                    <Block title="Where">
                      <div className="grid grid-cols-2 gap-x-6 font-mono text-[10px]">
                        {where.map(([k, v]) => (
                          <div key={k} className="break-all">
                            <span className="text-[#667085]">{k} </span>
                            {v}
                          </div>
                        ))}
                      </div>
                    </Block>
                  )}
                  {b.script && (
                    <Block title="Script / inputs (latest)">
                      <Pre text={b.script} />
                    </Block>
                  )}
                  {b.evidence && (
                    <Block title="Evidence (latest)">
                      <Pre text={b.evidence} />
                    </Block>
                  )}
                  {b.notes && (
                    <Block title="Notes">
                      <p className="whitespace-pre-wrap font-mono text-[10px] leading-relaxed">{b.notes}</p>
                    </Block>
                  )}
                  {b.sightings.length > 0 && (
                    <Block
                      title={`Sightings (${Math.min(b.sightings.length, 10)} of ${b.occurrences}, newest first)`}
                    >
                      <ul className="font-mono text-[10px] leading-relaxed">
                        {b.sightings.slice(0, 10).map((s, i) => (
                          <li key={i}>
                            {when(s.at)} · {s.agent ?? s.model ?? '—'}
                            {s.record_id ? ` · record ${s.record_id}` : ''}
                          </li>
                        ))}
                      </ul>
                    </Block>
                  )}
                </section>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
