'use client'

import { useEffect, useMemo, useState } from 'react'
import { objectives, type TaskDrill } from '@/lib/objectives'
import { holdShade } from './tradeColors'

const W = 640
const H = 220
const HS = 56 // the managed-state strip under the bars
const PAD = { l: 52, r: 10, t: 8, b: 18 }

const toMs = (iso: string) => Date.parse(iso.endsWith('Z') ? iso : `${iso}Z`) // the server speaks UTC
const signedPct = (v: number, d = 2) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`

/**
 * One day of a task candidate's result, drawn from what its data/action MCP returns for the window
 * (harness_actions): the target as candles or a line, the managed state underneath (a position, a
 * battery's charge), and what the actions did -- trades (entry/exit) or blocks (from/to) -- shaded
 * over the bars and listed below. Nothing here knows the task: the server describes its own result.
 */
export default function TaskDayChart({
  objectiveId,
  candidateId,
  day,
  days,
  dayValue,
  additive,
  onDay,
  onClose,
}: {
  objectiveId: string
  candidateId: string
  day: string
  days: string[]
  dayValue: number | undefined
  additive: boolean
  onDay: (day: string) => void
  onClose: () => void
}) {
  const [data, setData] = useState<TaskDrill | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    let live = true
    setData(null)
    setErr(null)
    const next = new Date(Date.parse(`${day}T00:00:00Z`) + 86_400_000).toISOString().slice(0, 10)
    objectives
      .candidateActions(objectiveId, candidateId, day, next)
      .then((d) => live && (d.problem ? setErr(d.problem) : setData(d)))
      .catch((e: Error) => live && setErr(e.message))
    return () => {
      live = false
    }
  }, [objectiveId, candidateId, day])

  const i = days.indexOf(day)
  const prev = i > 0 ? days[i - 1] : null
  const next = i >= 0 && i < days.length - 1 ? days[i + 1] : null
  const arrow = 'rounded border border-seam px-1.5 leading-[18px] text-ink-dim hover:border-ink-faint hover:text-ink disabled:opacity-30'
  const tz = data?.bars?.tz || 'UTC'

  return (
    <div className="mt-2 rounded-xl border border-seam bg-panel-hi/30 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px]">
        <button type="button" className={arrow} disabled={!prev} onClick={() => prev && onDay(prev)} title="previous day">
          ‹
        </button>
        <span className="text-ink">{day}</span>
        <button type="button" className={arrow} disabled={!next} onClick={() => next && onDay(next)} title="next day">
          ›
        </button>
        {dayValue !== undefined && (
          <span className="text-ink-dim">
            day{' '}
            <span className={dayValue >= 0 ? 'text-good' : 'text-bad'}>
              {additive ? `${dayValue >= 0 ? '+' : ''}${dayValue.toFixed(2)}` : signedPct(dayValue)}
            </span>
          </span>
        )}
        {data?.bars && (
          <span className="text-ink-faint">
            {data.bars.kind === 'ohlc' ? `${data.bars.every ?? ''} candles` : `${data.bars.columns[1]} line`} · {tz} time ·
            from the data/action MCP
          </span>
        )}
        <button
          type="button"
          onClick={onClose}
          aria-label="close"
          title="close"
          className="ml-auto grid h-6 w-6 place-items-center rounded border border-seam text-[14px] leading-none text-ink-dim hover:border-ink-faint hover:text-ink"
        >
          ×
        </button>
      </div>
      {err ? (
        <div className="text-[12px] text-bad">✗ {err}</div>
      ) : !data ? (
        <div className="grid h-[160px] place-items-center text-[12px] text-ink-faint">Asking the data/action MCP…</div>
      ) : !data.bars || !data.bars.rows.length ? (
        <div className="grid h-[100px] place-items-center text-[12px] text-ink-faint">No bars on {day}.</div>
      ) : (
        <Plot data={data} tz={tz} hover={hover} setHover={setHover} />
      )}
    </div>
  )
}

type Span = { from: number; to: number; fill: string; opacity: number; label: string }

function Plot({ data, tz, hover, setHover }: { data: TaskDrill; tz: string; hover: number | null; setHover: (t: number | null) => void }) {
  const bars = data.bars!
  const ohlc = bars.kind === 'ohlc'
  const model = useMemo(() => {
    const pts = bars.rows.map((r) => ({
      t: toMs(String(r[0])),
      o: r[1] as number | null,
      h: (ohlc ? r[2] : r[1]) as number | null,
      l: (ohlc ? r[3] : r[1]) as number | null,
      c: (ohlc ? r[4] : r[1]) as number | null,
    }))
    const t0 = pts[0].t
    const step = pts.length > 1 ? Math.max(1, pts[1].t - pts[0].t) : 60_000
    const t1 = pts[pts.length - 1].t + step
    const lo0 = Math.min(...pts.map((p) => p.l ?? Infinity))
    const hi0 = Math.max(...pts.map((p) => p.h ?? -Infinity))
    const pad = (hi0 - lo0) * 0.06 || Math.abs(hi0) * 0.001 || 1
    const lo = lo0 - pad
    const hi = hi0 + pad
    const x = (t: number) => PAD.l + ((t - t0) / (t1 - t0 || 1)) * (W - PAD.l - PAD.r)
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b)
    // The managed state as a step line: held from each point until the next.
    const st = (data.state ?? []).map(([t, v]) => ({ t: toMs(t), v }))
    const smax = Math.max(1e-9, ...st.map((s) => Math.abs(s.v)))
    const smin = Math.min(0, ...st.map((s) => s.v))
    const ys = (v: number) => 6 + (1 - (v - smin) / (smax - smin || 1)) * (HS - 12)
    const step_ = st
      .map((s, k) => {
        const xa = x(Math.max(s.t, t0))
        const xb = x(Math.min(k + 1 < st.length ? st[k + 1].t : t1, t1))
        return `${k ? 'L' : 'M'}${xa.toFixed(1)},${ys(s.v).toFixed(1)}L${xb.toFixed(1)},${ys(s.v).toFixed(1)}`
      })
      .join(' ')
    // What the actions did: trades (entry/exit, side, net) or blocks (from/to, what, profit).
    const spans: Span[] = []
    for (const e of data.events ?? []) {
      const a = (e.entry ?? e.from) as string | undefined
      const b = (e.exit ?? e.to) as string | undefined
      if (!a || !b) continue
      const side = e.side === 'short' || e.what === 'sell' ? -1 : 1
      const pnl = Number(e.net ?? e.profit ?? 0)
      const shade = holdShade(side, pnl)
      const end = e.to ? toMs(b) + step : toMs(b)
      spans.push({ from: Math.max(toMs(a), t0), to: Math.min(end, t1), fill: shade.fill, opacity: shade.opacity, label: String(e.side ?? e.what ?? '') })
    }
    const ticks = Array.from({ length: 6 }, (_, k) => t0 + ((t1 - t0) * k) / 5)
    return { pts, t0, t1, lo, hi, x, y, ys, step_, spans, ticks, smax, smin, step }
  }, [bars, data.state, data.events, ohlc])
  const { pts, t0, t1, lo, hi, x, y, ys, step_, spans, ticks, smax, smin, step } = model
  const fmt = (t: number, withSec = false) =>
    new Intl.DateTimeFormat('en-GB', { timeZone: tz, hour: '2-digit', minute: '2-digit', ...(withSec ? { second: '2-digit' } : {}) }).format(t)
  const cw = Math.max(1, (x(t0 + step) - x(t0)) * 0.7)
  const line = pts
    .filter((p) => p.c != null)
    .map((p, k) => `${k ? 'L' : 'M'}${x(p.t + step / 2).toFixed(1)},${y(p.c as number).toFixed(1)}`)
    .join(' ')

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    const vx = ((e.clientX - box.left) / box.width) * W
    const t = t0 + ((vx - PAD.l) / (W - PAD.l - PAD.r)) * (t1 - t0)
    setHover(t >= t0 && t <= t1 ? t : null)
  }
  const hp = hover === null ? null : pts.reduce<(typeof pts)[number] | null>((b, p) => (p.t <= hover ? p : b), null)
  const hs = hover === null ? null : (data.state ?? []).reduce<[string, number] | null>((b, s) => (toMs(s[0]) <= hover ? s : b), null)
  const guide = hover !== null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={9999} className="stroke-ink-dim" strokeWidth={0.8} pointerEvents="none" />

  return (
    <div onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full cursor-crosshair" role="img" aria-label="the day's bars with the actions shaded" onMouseMove={onMove}>
        {spans.map((s, k) => (
          <rect key={k} x={x(s.from)} y={PAD.t} width={Math.max(0.5, x(s.to) - x(s.from))} height={H - PAD.t - PAD.b} fill={s.fill} opacity={s.opacity} />
        ))}
        {[lo, (lo + hi) / 2, hi].map((v, k) => (
          <text key={k} x={PAD.l - 6} y={y(v) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
            {v.toFixed(2)}
          </text>
        ))}
        {ohlc
          ? pts.map((p) => {
              if (p.o == null || p.h == null || p.l == null || p.c == null) return null
              const cx = x(p.t + step / 2)
              const up = p.c >= p.o
              return (
                <g key={p.t}>
                  <line x1={cx} x2={cx} y1={y(p.h)} y2={y(p.l)} className="stroke-ink-dim" strokeWidth={0.8} />
                  <rect x={cx - cw / 2} y={y(Math.max(p.o, p.c))} width={cw} height={Math.max(0.8, Math.abs(y(p.o) - y(p.c)))}
                    className={up ? 'fill-panel stroke-ink' : 'fill-ink-dim stroke-ink-dim'} strokeWidth={0.7} />
                </g>
              )
            })
          : <path d={line} fill="none" className="stroke-ink" strokeWidth={1.2} />}
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={H - 4} textAnchor="middle" className="fill-ink-faint font-mono text-[9px]">
            {fmt(t)}
          </text>
        ))}
        {guide}
      </svg>
      {(data.state ?? []).length > 0 && (
        <svg viewBox={`0 0 ${W} ${HS}`} className="w-full cursor-crosshair" role="img" aria-label="managed state through the day" onMouseMove={onMove}>
          {smin < 0 && <line x1={PAD.l} x2={W - PAD.r} y1={ys(0)} y2={ys(0)} className="stroke-seam" />}
          <text x={PAD.l - 6} y={ys(smax) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">{+smax.toFixed(2)}</text>
          <text x={PAD.l - 6} y={ys(smin) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">{+smin.toFixed(2)}</text>
          <path d={step_} fill="none" className="stroke-accent" strokeWidth={1.4} />
          {guide}
        </svg>
      )}
      <div className="flex min-h-[18px] flex-wrap items-center gap-x-3 font-mono text-[10.5px] text-ink-dim">
        {hover !== null && hp ? (
          <>
            <span className="text-ink">{fmt(hover, true)}</span>
            {ohlc ? (
              <span>O {hp.o?.toFixed(2)} H {hp.h?.toFixed(2)} L {hp.l?.toFixed(2)} C {hp.c?.toFixed(2)}</span>
            ) : (
              <span>{bars.columns[1]} {hp.c?.toFixed(2)}</span>
            )}
            {hs && <span className="text-ink">{data.state_kind ?? 'state'} {+hs[1].toFixed(3)}</span>}
          </>
        ) : (
          <span className="text-ink-faint">hover for the bar and the managed state · {data.state_kind ?? ''}</span>
        )}
      </div>
      <EventsTable events={data.events ?? []} fmt={(iso) => fmt(toMs(iso))} />
    </div>
  )
}

/** Whatever the server reports the actions did, as a table: its own columns, in its own words. */
function EventsTable({ events, fmt }: { events: Record<string, unknown>[]; fmt: (iso: string) => string }) {
  if (!events.length) return <div className="mt-2 font-mono text-[11px] text-ink-faint">No actions took effect this day.</div>
  const cols = Array.from(new Set(events.flatMap((e) => Object.keys(e))))
  const cell = (k: string, v: unknown) => {
    if (typeof v === 'string' && /^\d{4}-\d{2}-\d{2}T/.test(v)) return fmt(v)
    if (typeof v === 'number') {
      if (['gross', 'net'].includes(k)) return <span className={v >= 0 ? 'text-good' : 'text-bad'}>{signedPct(v)}</span>
      if (k === 'profit') return <span className={v >= 0 ? 'text-good' : 'text-bad'}>{v.toFixed(4)}</span>
      return +v.toFixed(4)
    }
    return String(v ?? '')
  }
  return (
    <table className="mt-3 w-full font-mono text-[11px]">
      <thead className="text-ink-faint">
        <tr>
          {cols.map((c) => (
            <th key={c} className="py-1 text-left font-normal">{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {events.slice(0, 200).map((e, k) => (
          <tr key={k} className="border-t border-seam/60">
            {cols.map((c) => (
              <td key={c} className="py-1 tabular-nums">{cell(c, e[c])}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
