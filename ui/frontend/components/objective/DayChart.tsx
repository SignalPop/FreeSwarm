'use client'

import { useEffect, useMemo, useState } from 'react'
import { objectives, type CandidateDay, type DayTrade } from '@/lib/objectives'
import { fmtHold } from './PnlCalendar'
import { HOLD, holdShade } from './tradeColors'

const W = 640
const H = 240
const HP = 56 // the position strip under the prices
const PAD = { l: 52, r: 10, t: 8, b: 18 }

// The chart reads in New York time: US equity hours, with daylight saving handled by Intl.
const TZ = 'America/New_York'
const etParts = (t: number) => {
  const p = Object.fromEntries(
    new Intl.DateTimeFormat('en-US', {
      timeZone: TZ, hourCycle: 'h23', year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    }).formatToParts(new Date(t * 1000)).map((x) => [x.type, x.value]),
  )
  return { date: `${p.year}-${p.month}-${p.day}`, hm: `${p.hour}:${p.minute}`, hms: `${p.hour}:${p.minute}:${p.second}` }
}
const hhmm = (t: number) => etParts(t).hm
const hhmmss = (t: number) => etParts(t).hms
/** Epoch seconds of a New York wall-clock time on `day` (YYYY-MM-DD). */
function etTime(day: string, h: number, m: number): number {
  const [y, mo, d] = day.split('-').map(Number)
  const guess = Date.UTC(y, mo - 1, d, h, m) / 1000
  // New York's offset from UTC on that day (-4h in summer, -5h in winter), read back via Intl.
  const p = etParts(guess)
  const [ph, pm] = p.hm.split(':').map(Number)
  const [py, pmo, pd] = p.date.split('-').map(Number)
  const offset = Date.UTC(py, pmo - 1, pd, ph, pm) / 1000 - guess
  return guess - offset
}
/** The regular session every day is drawn over, so days line up: 9:30 to 16:00 New York. */
const SESSION = { open: [9, 30], close: [16, 0] } as const
const signedPct = (v: number, digits = 2) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`
const fmtPos = (p: number) => (p === 0 ? 'flat' : `${p > 0 ? 'long' : 'short'} ${Math.abs(p).toFixed(2).replace(/\.?0+$/, '')}`)

/**
 * One day of the dataset's prices with the candidate's positions laid over them: candles when
 * the dataset has open/high/low (a line of the price column when it does not), each holding shaded
 * by direction and by whether it made money (see HOLD), and the position's size as a step line
 * underneath. The x-axis is always the regular session, 9:30 to 16:00 New York time, so every
 * day lines up; bars and holdings outside it are left off the chart (the trade table lists them).
 */
export default function DayChart({
  objectiveId,
  candidateId,
  day,
  days,
  dayReturn,
  onDay,
  onClose,
  popup = false,
}: {
  objectiveId: string
  candidateId: string
  day: string
  /** Every day on the equity curve, in order, for stepping to the previous and next day. */
  days: string[]
  dayReturn: number | undefined
  onDay: (day: string) => void
  onClose: () => void
  /** Open as a dialog over the page rather than inline under the chart. */
  popup?: boolean
}) {
  const [data, setData] = useState<CandidateDay | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [slow, setSlow] = useState(false)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    let live = true
    setData(null)
    setErr(null)
    setHover(null)
    // A candidate scored before positions were kept is re-run once to recover them: say so
    // rather than look stuck.
    const t = setTimeout(() => live && setSlow(true), 1500)
    objectives
      .candidateDay(objectiveId, candidateId, day)
      .then((d) => live && setData(d))
      .catch((e: Error) => live && setErr(e.message))
      .finally(() => {
        clearTimeout(t)
        if (live) setSlow(false)
      })
    return () => {
      live = false
      clearTimeout(t)
    }
  }, [objectiveId, candidateId, day])

  const i = days.indexOf(day)
  const prev = i > 0 ? days[i - 1] : null
  const next = i >= 0 && i < days.length - 1 ? days[i + 1] : null
  // As a dialog: Escape closes it (and only it -- the candidate view underneath also listens),
  // the arrow keys step a day.
  useEffect(() => {
    if (!popup) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowLeft' && prev) onDay(prev)
      else if (e.key === 'ArrowRight' && next) onDay(next)
      else return
      e.stopImmediatePropagation()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [popup, prev, next, onDay, onClose])

  const arrow = 'rounded border border-seam px-1.5 leading-[18px] text-ink-dim hover:border-ink-faint hover:text-ink disabled:opacity-30'

  const body = (
    <div
      className={popup ? 'max-h-[90vh] w-full max-w-[920px] overflow-y-auto rounded-2xl border border-seam bg-panel p-4 shadow-2xl' : 'mt-2 rounded-xl border border-seam bg-panel-hi/30 p-3'}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px]">
        <button type="button" className={arrow} disabled={!prev} onClick={() => prev && onDay(prev)} title="previous day">
          ‹
        </button>
        <span className="text-ink">{day}</span>
        <button type="button" className={arrow} disabled={!next} onClick={() => next && onDay(next)} title="next day">
          ›
        </button>
        {dayReturn !== undefined && (
          <span className="text-ink-dim">
            day <span className={dayReturn >= 0 ? 'text-good' : 'text-bad'}>{signedPct(dayReturn)}</span> net
          </span>
        )}
        {data?.bars ? (
          <span className="text-ink-faint">
            {data.bucket_s! >= 60 ? `${data.bucket_s! / 60}-min` : `${data.bucket_s}-s`} {data.ohlc ? 'candles' : `${data.price} line`} ·{' '}
            {data.changes} position change{data.changes === 1 ? '' : 's'} · New York time, 9:30–16:00
          </span>
        ) : null}
        <span className="ml-auto flex items-center gap-3 text-ink-faint">
          {Object.values(HOLD).map((h) => (
            <span key={h.label} className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: h.fill, opacity: Math.min(1, h.opacity * 2) }} />
              {h.label}
            </span>
          ))}
          <button type="button" onClick={onClose} className="hover:text-accent">
            close
          </button>
        </span>
      </div>
      {err ? (
        <div className="text-[12px] text-bad">✗ {err}</div>
      ) : !data ? (
        <div className="grid h-[200px] place-items-center text-center text-[12px] text-ink-faint">
          {slow
            ? "Recovering this candidate's positions — it was scored before they were kept, so it is re-run once (as long as its evaluation took). Later days open instantly."
            : 'Loading…'}
        </div>
      ) : !data.bars ? (
        <div className="grid h-[120px] place-items-center text-[12px] text-ink-faint">No bars in the dataset on {day}.</div>
      ) : (
        <Plot data={data} hover={hover} setHover={setHover} />
      )}
      {data?.recovered && (
        <div className="mt-1 font-mono text-[10px] text-ink-faint">positions recovered from the {data.recovered}; kept for next time</div>
      )}
    </div>
  )
  if (!popup) return body
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-6" onClick={onClose}>
      {body}
    </div>
  )
}

function Plot({
  data,
  hover,
  setHover,
}: {
  data: CandidateDay
  hover: number | null
  setHover: (t: number | null) => void
}) {
  const t0 = etTime(data.day, ...SESSION.open)
  const t1 = etTime(data.day, ...SESSION.close)
  const bucket = data.bucket_s ?? 60
  const candles = useMemo(() => (data.candles ?? []).filter((c) => c[0] >= t0 && c[0] < t1), [data.candles, t0, t1])
  // Holdings clipped to the session; one wholly outside it is not drawn.
  const spans = useMemo(
    () =>
      (data.spans ?? [])
        .filter((s) => s.to > t0 && s.from < t1)
        .map((s) => ({ ...s, from: Math.max(s.from, t0), to: Math.min(s.to, t1) })),
    [data.spans, t0, t1],
  )
  const trades = data.trades ?? []
  // A span is shaded by the outcome of the whole trade it belongs to, so a trade that resizes
  // does not come out in patches of winning and losing colour.
  const tradeAt = (t: number): DayTrade | undefined => trades.find((tr) => tr.entry_t <= t && t < tr.exit_t)
  const model = useMemo(() => {
    const x = (t: number) => PAD.l + ((t - t0) / (t1 - t0 || 1)) * (W - PAD.l - PAD.r)
    const lows = candles.map((c) => c[3] ?? c[4]).filter((v): v is number => v != null)
    const highs = candles.map((c) => c[2] ?? c[4]).filter((v): v is number => v != null)
    let lo = Math.min(...lows)
    let hi = Math.max(...highs)
    const pad = (hi - lo) * 0.06 || Math.abs(hi) * 0.001 || 1
    lo -= pad
    hi += pad
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b)
    const maxPos = Math.max(1e-9, ...spans.map((s) => Math.abs(s.pos)))
    const yp = (p: number) => 6 + (1 - (p + maxPos) / (2 * maxPos)) * (HP - 12)
    const line = candles
      .filter((c) => c[4] != null)
      .map((c, k) => `${k ? 'L' : 'M'}${x(c[0] + bucket / 2).toFixed(1)},${y(c[4] as number).toFixed(1)}`)
      .join(' ')
    const step = spans
      .map((s, k) => `${k ? 'L' : 'M'}${x(s.from).toFixed(1)},${yp(s.pos).toFixed(1)}L${x(s.to).toFixed(1)},${yp(s.pos).toFixed(1)}`)
      .join(' ')
    // The open, then every hour to the close.
    const ticks = [t0, ...Array.from({ length: 7 }, (_, k) => etTime(data.day, 10 + k, 0))]
    return { x, y, lo, hi, yp, maxPos, line, step, ticks }
  }, [candles, spans, bucket, t0, t1, data.day])
  if (!candles.length)
    return <div className="grid h-[120px] place-items-center text-[12px] text-ink-faint">No bars between 9:30 and 16:00 New York time on {data.day}.</div>
  const { x, y, lo, hi, yp, maxPos, line, step, ticks } = model
  const cw = Math.max(1, (x(t0 + bucket) - x(t0)) * 0.7)
  // The close of the candle a time falls in: where an entry or exit marker sits.
  const priceAt = (t: number): number | null => {
    let c: (typeof candles)[number] | null = null
    for (const k of candles) {
      if (k[0] > t) break
      c = k
    }
    return c?.[4] ?? null
  }

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    const vx = ((e.clientX - box.left) / box.width) * W
    const t = t0 + ((vx - PAD.l) / (W - PAD.l - PAD.r)) * (t1 - t0)
    setHover(t >= t0 && t <= t1 ? t : null)
  }
  const hc = hover === null ? null : candles.reduce<(typeof candles)[number] | null>((best, c) => (c[0] <= hover ? c : best), null)
  const hs = hover === null ? null : spans.find((s) => hover >= s.from && hover < s.to) ?? null
  const ht = hover === null ? undefined : tradeAt(hover)
  const guide = hover !== null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={9999} className="stroke-ink-dim" strokeWidth={0.8} pointerEvents="none" />

  return (
    <div onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full cursor-crosshair" role="img" aria-label="the day's prices with positions shaded" onMouseMove={onMove}>
        {spans.map((s, k) => {
          if (s.pos === 0) return null
          const tr = tradeAt(s.from)
          const { fill, opacity } = holdShade(s.pos, tr ? tr.net : s.ret)
          return (
            <rect
              key={k}
              x={x(s.from)}
              y={PAD.t}
              width={Math.max(0.5, x(s.to) - x(s.from))}
              height={H - PAD.t - PAD.b}
              fill={fill}
              opacity={opacity}
            />
          )
        })}
        {[lo, (lo + hi) / 2, hi].map((v, k) => (
          <text key={k} x={PAD.l - 6} y={y(v) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
            {v.toFixed(2)}
          </text>
        ))}
        {data.ohlc
          ? candles.map(([t, o, h, l, c]) => {
              if (o == null || h == null || l == null || c == null) return null
              const cx = x(t + bucket / 2)
              const up = c >= o
              return (
                <g key={t}>
                  <line x1={cx} x2={cx} y1={y(h)} y2={y(l)} className="stroke-ink-dim" strokeWidth={0.8} />
                  <rect
                    x={cx - cw / 2}
                    y={y(Math.max(o, c))}
                    width={cw}
                    height={Math.max(0.8, Math.abs(y(o) - y(c)))}
                    className={up ? 'fill-panel stroke-ink' : 'fill-ink-dim stroke-ink-dim'}
                    strokeWidth={0.7}
                  />
                </g>
              )
            })
          : <path d={line} fill="none" className="stroke-ink" strokeWidth={1.2} />}
        {/* Entries as a triangle on the price (up for a long, down for a short) in the trade's
            colour; exits as a ring. A trade carried in from yesterday has no entry here. */}
        {trades.map((tr, k) => {
          const { fill } = holdShade(tr.side, tr.net)
          const ex = tr.exit_t >= t0 && tr.exit_t <= t1 && !tr.open ? priceAt(tr.exit_t) : null
          const en = !tr.carried && tr.entry_t >= t0 && tr.entry_t < t1 ? priceAt(tr.entry_t) : null
          return (
            <g key={k} pointerEvents="none">
              {en != null && (
                <path
                  d={
                    tr.side > 0
                      ? `M${x(tr.entry_t)},${y(en) + 3}l-5,9h10z`
                      : `M${x(tr.entry_t)},${y(en) - 3}l-5,-9h10z`
                  }
                  fill={fill}
                  className="stroke-panel"
                  strokeWidth={1}
                />
              )}
              {ex != null && <circle cx={x(tr.exit_t)} cy={y(ex)} r={3.5} fill="none" stroke={fill} strokeWidth={2} />}
            </g>
          )
        })}
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={H - 4} textAnchor="middle" className="fill-ink-faint font-mono text-[9px]">
            {hhmm(t)}
          </text>
        ))}
        {guide}
      </svg>
      <svg viewBox={`0 0 ${W} ${HP}`} className="w-full cursor-crosshair" role="img" aria-label="position through the day" onMouseMove={onMove}>
        <line x1={PAD.l} x2={W - PAD.r} y1={yp(0)} y2={yp(0)} className="stroke-seam" />
        <text x={PAD.l - 6} y={yp(maxPos) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
          +{+maxPos.toFixed(2)}
        </text>
        <text x={PAD.l - 6} y={yp(-maxPos) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
          −{+maxPos.toFixed(2)}
        </text>
        <text x={PAD.l - 6} y={yp(0) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
          pos
        </text>
        <path d={step} fill="none" className="stroke-accent" strokeWidth={1.4} />
        {guide}
      </svg>
      <div className="flex min-h-[18px] flex-wrap items-center gap-x-3 font-mono text-[10.5px] text-ink-dim">
        {hover !== null && hc ? (
          <>
            <span className="text-ink">{hhmmss(hover)}</span>
            {data.ohlc ? (
              <span>
                O {hc[1]?.toFixed(2)} H {hc[2]?.toFixed(2)} L {hc[3]?.toFixed(2)} C {hc[4]?.toFixed(2)}
              </span>
            ) : (
              <span>
                {data.price} {hc[4]?.toFixed(2)}
              </span>
            )}
            {hs && <span className="text-ink">{fmtPos(hs.pos)}</span>}
            {ht ? (
              <span>
                trade {when(ht.entry_t, data.day)}–{ht.open ? 'open' : when(ht.exit_t, data.day)} · net{' '}
                <span className={ht.net >= 0 ? 'text-good' : 'text-bad'}>{signedPct(ht.net)}</span> (gross {signedPct(ht.gross)})
              </span>
            ) : (
              hs &&
              hs.pos !== 0 && (
                <span>
                  {hhmm(hs.from)}–{hhmm(hs.to)} ·{' '}
                  <span className={hs.ret >= 0 ? 'text-good' : 'text-bad'}>{signedPct(hs.ret)}</span> before costs
                </span>
              )
            )}
          </>
        ) : (
          <span className="text-ink-faint">hover for the bar, the position held and the trade it belongs to</span>
        )}
      </div>
      {trades.length > 0 && <TradeTable trades={trades} day={data.day} costBps={data.cost_bps ?? 0} />}
    </div>
  )
}

/** A New York time on the chart's day as HH:MM; on another day with its date, MM-DD HH:MM. */
function when(t: number, day: string): string {
  const p = etParts(t)
  return p.date === day ? p.hm : `${p.date.slice(5)} ${p.hm}`
}

function TradeTable({ trades, day, costBps }: { trades: DayTrade[]; day: string; costBps: number }) {
  return (
    <table className="mt-3 w-full font-mono text-[11px]">
      <thead className="text-ink-faint">
        <tr>
          <th className="py-1 text-left font-normal">trade</th>
          <th className="py-1 text-right font-normal">size</th>
          <th className="py-1 text-right font-normal">entry</th>
          <th className="py-1 text-right font-normal">exit</th>
          <th className="py-1 text-right font-normal">held</th>
          <th className="py-1 text-right font-normal">gross</th>
          <th className="py-1 text-right font-normal" title={`after ${costBps} bps per unit traded`}>
            net
          </th>
        </tr>
      </thead>
      <tbody>
        {trades.map((tr, k) => {
          const shade = holdShade(tr.side, tr.net)
          return (
            <tr key={k} className="border-t border-seam/60">
              <td className="py-1">
                <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm align-[-1px]" style={{ background: shade.fill }} />
                <span className="text-ink">{tr.side > 0 ? 'long' : 'short'}</span>
                {tr.carried && <span className="text-ink-faint"> · carried in</span>}
              </td>
              <td className="py-1 text-right tabular-nums">{+tr.size.toFixed(2)}</td>
              <td className="py-1 text-right tabular-nums">{when(tr.entry_t, day)}</td>
              <td className="py-1 text-right tabular-nums">{tr.open ? 'still open' : when(tr.exit_t, day)}</td>
              <td className="py-1 text-right tabular-nums">{fmtHold(tr.exit_t - tr.entry_t)}</td>
              <td className={`py-1 text-right tabular-nums ${tr.gross >= 0 ? 'text-good' : 'text-bad'}`}>{signedPct(tr.gross)}</td>
              <td className={`py-1 text-right tabular-nums ${tr.net >= 0 ? 'text-good' : 'text-bad'}`}>{signedPct(tr.net)}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
