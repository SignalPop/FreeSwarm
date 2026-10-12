'use client'

import { useCallback, useEffect, useState } from 'react'
import { objectives, type DataAudit } from '@/lib/objectives'

const TONE = { ok: 'text-good', warn: 'text-warn', bad: 'text-bad' } as const

/** One month of the audit as a small block: green when every day is verified, red when none is. */
function monthTone(m: { days: number; verified: number }): string {
  const share = m.days ? m.verified / m.days : 0
  return share >= 0.95 ? 'bg-good/70' : share >= 0.6 ? 'bg-warn/70' : share > 0 ? 'bg-bad/50' : 'bg-bad/80'
}

/** A task server's data audit (harness_data_audit): the verified date range on one line, and on
 *  demand what is wrong with the rest -- month by month, the data outside the range with the columns
 *  to blame, where columns stop, and the bad days inside it. */
export function DataAuditView({
  audit,
  onRecheck,
  busy,
}: {
  audit: DataAudit
  onRecheck?: () => void
  busy?: boolean
}) {
  const [open, setOpen] = useState(audit.status === 'bad')
  const v = audit.verified
  const h = audit.holdout
  const holdoutShort = !!h && h.days > 0 && h.unverified_days > 0
  return (
    <div className="font-mono text-[10.5px] leading-relaxed">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
        <span className="text-ink-faint">data</span>
        <span className={TONE[audit.status]}>
          {v ? (
            <>
              verified {v.from} → {v.to}{' '}
              <span className="text-ink-faint">
                ({v.good_days} good of {audit.data.days} days{audit.data.last ? `; rows to ${audit.data.last}` : ''})
              </span>
            </>
          ) : (
            'nothing verified'
          )}
        </span>
        {holdoutShort && (
          <span className={h.unverified_days / h.days > 0.25 ? 'text-bad' : 'text-warn'}>
            · holdout from {h.from}: {h.unverified_days} of {h.days} days unverified
            {h.unverified_after ? ` (past ${h.unverified_after})` : ''}
          </span>
        )}
        {audit.split_after_verified && <span className="text-bad">· the whole holdout lies past the verified data</span>}
        <button type="button" onClick={() => setOpen(!open)} className="text-ink-faint underline-offset-2 hover:text-ink hover:underline">
          {open ? 'hide' : 'details'}
        </button>
        {onRecheck && (
          <button
            type="button"
            disabled={busy}
            onClick={onRecheck}
            className="text-ink-faint underline-offset-2 hover:text-ink hover:underline disabled:opacity-40"
            title="Audit the server's rows again (after a re-export, or to bypass its cache)"
          >
            {busy ? 'checking…' : 're-check'}
          </button>
        )}
      </div>
      {open && (
        <div className="mt-1.5 space-y-1.5 rounded-lg border border-seam p-2 text-ink-dim">
          {audit.months?.length ? (
            <div>
              <div className="flex flex-wrap gap-[2px]">
                {audit.months.map((m) => (
                  <span
                    key={m.month}
                    className={`inline-block h-3 w-3 rounded-[2px] ${monthTone(m)}`}
                    title={`${m.month}: ${m.verified} of ${m.days} days verified${m.bad ? `, ${m.bad} bad` : ''}`}
                  />
                ))}
              </div>
              <div className="mt-0.5 text-ink-faint">
                {audit.months[0].month} → {audit.months[audit.months.length - 1].month}, one block a month (hover for its days)
              </div>
            </div>
          ) : null}
          {audit.excluded?.map((x) => (
            <div key={x.from}>
              <span className="text-bad">
                excluded {x.from} → {x.to}
              </span>{' '}
              ({x.days} days): {x.why}
            </div>
          ))}
          {audit.columns_end?.length ? (
            <div>
              <span className="text-ink">columns that stop</span>
              {audit.columns_end.map((g) => (
                <div key={g.day} className="pl-2">
                  last good {g.day}: {g.count} column{g.count > 1 ? 's' : ''} ({g.columns.slice(0, 8).join(', ')}
                  {g.count > 8 ? ', …' : ''})
                </div>
              ))}
            </div>
          ) : null}
          {audit.columns_start?.length ? (
            <div>
              <span className="text-ink">columns that start late</span>
              {audit.columns_start.map((g) => (
                <div key={g.day} className="pl-2">
                  first good {g.day}: {g.columns.slice(0, 8).join(', ')}
                  {g.count > 8 ? ', …' : ''}
                </div>
              ))}
            </div>
          ) : null}
          {audit.exceptions?.length ? (
            <div>
              <span className="text-ink">
                bad days inside the range ({audit.exceptions_total ?? audit.exceptions.length})
              </span>
              {audit.exceptions.slice(0, 12).map((x) => (
                <div key={x.day} className="pl-2">
                  {x.day}: {x.why}
                </div>
              ))}
              {(audit.exceptions_total ?? 0) > 12 && <div className="pl-2 text-ink-faint">and {(audit.exceptions_total ?? 0) - 12} more</div>}
            </div>
          ) : null}
          {audit.gaps?.length ? (
            <div>
              <span className="text-ink">gaps</span>{' '}
              {audit.gaps.map((g) => `${g.after} → ${g.before} (${g.calendar_days} days)`).join(' · ')}
            </div>
          ) : null}
          {audit.intermittent_columns?.length ? (
            <div className="text-ink-faint">
              patchy by nature (reported, not failing days):{' '}
              {audit.intermittent_columns.map((c) => `${c.column} ${Math.round(c.bad_share * 100)}%`).join(', ')}
            </div>
          ) : null}
          {audit.status === 'bad' && v && (
            <div className="text-warn">
              Scores past {v.to} are computed on data that is not there. Re-export the source if the database has been
              fixed, or end the holdout at {v.to}.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/** The data audit of the rows a task objective is scored on, fetched once per objective. */
export default function ObjectiveDataAudit({ id }: { id: string }) {
  const [audit, setAudit] = useState<DataAudit | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(
    (refresh: boolean) => {
      setBusy(true)
      setErr(null)
      objectives
        .objectiveDataAudit(id, refresh)
        .then(setAudit)
        .catch((e: Error) => setErr(e.message))
        .finally(() => setBusy(false))
    },
    [id],
  )

  useEffect(() => {
    setAudit(null)
    load(false)
  }, [load])

  if (err)
    return (
      <div className="font-mono text-[10.5px] text-ink-faint">
        data audit unavailable: {err}{' '}
        <button type="button" onClick={() => load(false)} className="underline-offset-2 hover:text-ink hover:underline">
          retry
        </button>
      </div>
    )
  if (!audit) return <div className="font-mono text-[10.5px] text-ink-faint">{busy ? 'verifying the data…' : null}</div>
  return <DataAuditView audit={audit} busy={busy} onRecheck={() => load(true)} />
}
