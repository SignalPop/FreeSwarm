'use client'

import { useCallback, useEffect, useState } from 'react'
import { duration } from '@/lib/format'
import { fmtMetric, objectives, type CandidateMetrics, type CurveStats, type SourceRun, type SourceRuns as Runs } from '@/lib/objectives'
import { EquityCurve } from './Charts'
import TaskDayChart from './TaskDayChart'

/**
 * A task candidate replayed on the task server's OTHER data sources: its code as it was scored,
 * run unchanged (no agent, no training) on that source's rows and valued there. Shows the
 * replay's equity curve (click a day to drill in), how it compares with the scored run period by
 * period -- the dates it was built on, the objective's holdout, dates the objective never had --
 * and the server's full valuation.
 */
export default function SourceRuns({
  objectiveId,
  candidateId,
  additive,
  renderResults,
}: {
  objectiveId: string
  candidateId: string
  additive: boolean
  renderResults: (t: NonNullable<CandidateMetrics['task']>) => React.ReactNode
}) {
  const [data, setData] = useState<Runs | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const load = useCallback(
    () =>
      objectives
        .sourceRuns(objectiveId, candidateId)
        .then((d) => {
          setData(d)
          setErr(null)
        })
        .catch((e: Error) => setErr(e.message)),
    [objectiveId, candidateId],
  )

  useEffect(() => {
    setData(null)
    void load()
  }, [load])

  const running = Object.values(data?.runs ?? {}).some((r) => r?.state === 'running')
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => void load(), 4000)
    return () => clearInterval(t)
  }, [running, load])

  if (err) return <div className="font-mono text-[11px] text-bad">data sources: {err}</div>
  const others = (data?.sources ?? []).filter((s) => s.name !== data?.own)
  if (!data || others.length === 0) return null
  const ownTitle = data.sources.find((s) => s.name === data.own)?.title ?? data.own ?? 'its own source'

  return (
    <div className="space-y-3">
      {others.map((s) => (
        <SourceCard
          key={s.name}
          objectiveId={objectiveId}
          candidateId={candidateId}
          source={s}
          ownTitle={ownTitle}
          run={data.runs[s.name] ?? null}
          split={data.split_date}
          additive={additive}
          onStarted={load}
          renderResults={renderResults}
        />
      ))}
    </div>
  )
}

function SourceCard({
  objectiveId,
  candidateId,
  source,
  ownTitle,
  run,
  split,
  additive,
  onStarted,
  renderResults,
}: {
  objectiveId: string
  candidateId: string
  source: { name: string; title?: string; description?: string }
  ownTitle: string
  run: SourceRun | null
  split: string | null
  additive: boolean
  onStarted: () => Promise<void> | void
  renderResults: (t: NonNullable<CandidateMetrics['task']>) => React.ReactNode
}) {
  const [starting, setStarting] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [day, setDay] = useState<string | null>(null)
  const [details, setDetails] = useState(false)
  const title = source.title ?? source.name

  async function start() {
    setStarting(true)
    setErr(null)
    try {
      await objectives.startSourceRun(objectiveId, candidateId, source.name)
      await onStarted()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setStarting(false)
    }
  }

  const busy = starting || run?.state === 'running'
  const curve = run?.state === 'done' ? (run.curve ?? []) : []
  const m = run?.metrics

  return (
    <div className="rounded-xl border border-seam p-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-[11px] uppercase tracking-wide text-ink-faint">On another data source</span>
        <span className="font-mono text-[12px] text-ink">{title}</span>
        <button
          type="button"
          onClick={start}
          disabled={busy}
          className="ml-auto rounded-md border border-accent/40 px-2.5 py-1 font-mono text-[11px] text-accent transition-colors hover:bg-accent/10 disabled:cursor-wait disabled:opacity-60"
        >
          {busy ? 'running…' : run?.state === 'done' ? '↻ run again' : `▶ run on ${title}`}
        </button>
      </div>
      <div className="mt-1 text-[11px] leading-snug text-ink-faint">
        The code as it was scored on {ownTitle}, run unchanged on {title}&apos;s rows -- no agent, no retraining -- and valued
        by the task server there. Nothing about the candidate or its ranking changes.
        {source.description ? ` ${title}: ${source.description}.` : ''}
      </div>

      {err && <div className="mt-2 font-mono text-[11px] text-bad">{err}</div>}
      {run?.state === 'running' && (
        <div className="mt-2 font-mono text-[11px] text-ink-dim">
          ● {run.phase ?? 'running'}
          {run.started_at ? ` · ${duration(Date.now() / 1000 - run.started_at)}` : ''}
        </div>
      )}
      {run?.state === 'failed' && (
        <pre className="mt-2 max-h-[220px] overflow-auto whitespace-pre-wrap rounded-lg border border-bad/40 bg-bad/[0.07] p-2 font-mono text-[11px] text-ink-dim">
          {run.error}
        </pre>
      )}

      {run?.state === 'done' && (
        <div className="mt-3 space-y-3">
          <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-ink-dim">
            <span>
              score on {title}: <span className="text-ink">{fmtNum(run.score)}</span>
            </span>
            {run.comparison?.daily_correlation != null && (
              <span title="correlation of the two runs' daily values over the days both have">
                daily correlation with the scored run {run.comparison.daily_correlation.toFixed(2)} over {run.comparison.shared_days}{' '}
                shared days
              </span>
            )}
            {run.duration_s != null && <span>ran in {run.duration_s.toFixed(0)}s</span>}
            {run.finished_at && <span>{new Date(run.finished_at * 1000).toLocaleString()}</span>}
          </div>
          {run.note && <div className="text-[11px] text-warn">{run.note}</div>}

          {curve.length > 1 && (
            <div>
              <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-faint">
                {additive ? 'Cumulative value' : 'Equity (growth of 1)'} on {title}
                {split ? ` · line = the objective's holdout (${split})` : ''} · click a day for its trades
              </div>
              <EquityCurve returns={curve} split={split} onDay={setDay} additive={additive} />
              {day && (
                <TaskDayChart
                  objectiveId={objectiveId}
                  candidateId={candidateId}
                  source={source.name}
                  day={day}
                  days={curve.map(([d]) => d)}
                  dayValue={curve.find(([d]) => d === day)?.[1]}
                  additive={additive}
                  onDay={setDay}
                  onClose={() => setDay(null)}
                />
              )}
            </div>
          )}

          {run.comparison && <Comparison c={run.comparison} title={title} ownTitle={ownTitle} additive={additive} />}

          {m?.task && (
            <div>
              <button type="button" onClick={() => setDetails(!details)} className="font-mono text-[11.5px] text-accent hover:underline">
                {details ? '▾' : '▸'} the task server&apos;s valuation on {title}
                {run.holdout_from ? ` (its segments split at ${title}'s own holdout, ${run.holdout_from.slice(0, 10)})` : ''}
              </button>
              {details && <div className="mt-2">{renderResults(m.task)}</div>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function Comparison({
  c,
  title,
  ownTitle,
  additive,
}: {
  c: NonNullable<SourceRun['comparison']>
  title: string
  ownTitle: string
  additive: boolean
}) {
  const ret = (v: number | undefined) => (v == null ? '—' : additive ? v.toFixed(4) : fmtMetric('total_return', v))
  const cols: { label: string; get: (s: CurveStats) => string }[] = [
    { label: 'days', get: (s) => `${s.active_days}/${s.days}` },
    { label: additive ? 'sum' : 'return', get: (s) => ret(s.total_return) },
    { label: 'Sharpe', get: (s) => (s.sharpe == null ? '—' : fmtMetric('sharpe', s.sharpe)) },
    { label: 'max DD', get: (s) => (additive ? s.max_drawdown.toFixed(4) : fmtMetric('max_drawdown', s.max_drawdown)) },
    { label: 'win', get: (s) => (s.win_rate == null ? '—' : `${(s.win_rate * 100).toFixed(0)}%`) },
  ]
  return (
    <div className="overflow-x-auto">
      <table className="w-full font-mono text-[11px]">
        <thead>
          <tr className="text-ink-faint">
            <th className="py-1 text-left font-normal">period</th>
            <th className="py-1 text-left font-normal">run</th>
            {cols.map((k) => (
              <th key={k.label} className="py-1 text-right font-normal">
                {k.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {c.periods.map((p) =>
            (
              [
                [ownTitle + ' (scored)', p.scored],
                [title, p.replay],
              ] as const
            ).map(([label, s], i) => (
              <tr key={`${p.name}-${i}`} className={i === 0 ? 'border-t border-seam/60' : ''}>
                <td className="py-0.5 pr-2 align-top text-ink" title={p.about}>
                  {i === 0 ? p.name : ''}
                </td>
                <td className={`py-0.5 pr-2 ${i === 1 ? 'text-accent' : 'text-ink-dim'}`}>{label}</td>
                {cols.map((k) => (
                  <td key={k.label} className="py-0.5 text-right text-ink-dim">
                    {s ? k.get(s) : '—'}
                  </td>
                ))}
              </tr>
            )),
          )}
        </tbody>
      </table>
      <div className="mt-1 text-[10.5px] text-ink-faint">
        Sharpe annualised from daily values; days = active/all. Hover a period for its dates.
      </div>
    </div>
  )
}

function fmtNum(v: number | null | undefined) {
  return v == null ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3)
}
