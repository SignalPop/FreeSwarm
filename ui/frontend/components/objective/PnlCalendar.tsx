'use client'

import { useEffect, useMemo, useState } from 'react'
import { objectives, type CalendarDay, type TradeCalendar } from '@/lib/objectives'
import { HOLD } from './tradeColors'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

const signedPct = (v: number, digits = 2) => `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(digits)}%`
/** Short enough for a calendar cell: two decimals under 10%, one above. */
const cellPct = (v: number) => signedPct(v, Math.abs(v) >= 0.1 ? 1 : 2)
const pct0 = (v: number | null | undefined) => (v == null ? '—' : `${Math.round(v * 100)}%`)
export function fmtHold(s: number | null | undefined): string {
  if (s == null) return '—'
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.round(s / 60)}m`
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`
  return `${Math.floor(s / 86400)}d ${Math.round((s % 86400) / 3600)}h`
}
/** A day's trades split by direction and outcome, in the day chart's colours. */
function splitTrades(st: CalendarDay): { n: number; fill: string; label: string }[] {
  const lw = st.long_wins ?? 0
  const sw = st.short_wins ?? 0
  return [
    { n: lw, ...HOLD.longWin },
    { n: st.long - lw, ...HOLD.longLoss },
    { n: sw, ...HOLD.shortWin },
    { n: st.short - sw, ...HOLD.shortLoss },
  ]
}
const compound = (rs: number[]) => rs.reduce((a, r) => a * (1 + r), 1) - 1

/**
 * A day's colour on the diverging scale: green up, red down, deeper for a bigger move, fading
 * to the neutral surface at zero. Saturation stops at half so the cell's own text stays legible
 * in both themes; the signed number printed in every cell carries the value, not the colour.
 */
function tint(r: number, scale: number): string {
  if (Math.abs(r) < 1e-9) return 'var(--color-panel-hi)'
  const k = Math.min(1, Math.abs(r) / scale)
  return `color-mix(in oklab, var(--color-${r > 0 ? 'good' : 'bad'}) ${Math.round(10 + 40 * k)}%, var(--color-panel-hi))`
}

type Cell = { date: string; dom: number; r?: number; holdout: boolean } | null
type Month = {
  key: string
  label: string
  weeks: { cells: Cell[]; ret: number | null }[]
  ret: number
  up: number
  down: number
}

/**
 * The candidate's daily P&L as a calendar: one card per month, a cell per day coloured by its
 * net return, with the trades opened that day and their win rate; a weekly total down the
 * right. Summary tiles on top cover every day and every trade. Click a day for its bars and
 * trades. Trade statistics need the candidate's positions -- without them (a self-reported
 * returns objective, an ensemble) the calendar shows P&L alone.
 */
export default function PnlCalendar({
  objectiveId,
  candidateId,
  returns,
  split,
  trades,
  selected,
  onDay,
}: {
  objectiveId: string
  candidateId: string
  returns: [string, number][]
  split: string | null
  /** Whether trade statistics exist for this candidate (it reports positions). */
  trades: boolean
  selected: string | null
  /** Open a day's bars and trades; absent when there are no positions to show. */
  onDay?: (day: string) => void
}) {
  const [cal, setCal] = useState<TradeCalendar | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [slow, setSlow] = useState(false)
  const [hover, setHover] = useState<string | null>(null)

  useEffect(() => {
    if (!trades) return
    let live = true
    setCal(null)
    setErr(null)
    const t = setTimeout(() => live && setSlow(true), 1500)
    objectives
      .candidateCalendar(objectiveId, candidateId)
      .then((c) => live && setCal(c))
      .catch((e: Error) => live && setErr(e.message))
      .finally(() => {
        clearTimeout(t)
        if (live) setSlow(false)
      })
    return () => {
      live = false
      clearTimeout(t)
    }
  }, [objectiveId, candidateId, trades])

  const model = useMemo(() => {
    const ret = new Map(returns)
    const abs = returns.map(([, r]) => Math.abs(r)).sort((a, b) => a - b)
    // The 95th percentile, so one freak day does not wash every other cell out.
    const scale = abs[Math.floor(0.95 * (abs.length - 1))] || abs[abs.length - 1] || 0.01
    const dow = (d: Date) => (d.getUTCDay() + 6) % 7 // Monday first
    const weekend = returns.some(([d]) => dow(new Date(`${d}T00:00:00Z`)) >= 5)
    const cols = weekend ? 7 : 5
    const months: Month[] = []
    if (returns.length) {
      const [y0, m0] = returns[0][0].split('-').map(Number)
      const [y1, m1] = returns[returns.length - 1][0].split('-').map(Number)
      for (let y = y0, m = m0 - 1; y < y1 || (y === y1 && m <= m1 - 1); m === 11 ? (y++, (m = 0)) : m++) {
        const first = new Date(Date.UTC(y, m, 1))
        const days = new Date(Date.UTC(y, m + 1, 0)).getUTCDate()
        const offset = dow(first)
        const weeks: Month['weeks'] = []
        const mr: number[] = []
        for (let d = 1; d <= days; d++) {
          const row = Math.floor((d - 1 + offset) / 7)
          const col = (d - 1 + offset) % 7
          if (col >= cols) continue
          while (weeks.length <= row) weeks.push({ cells: Array(cols).fill(null), ret: null })
          const date = `${y}-${String(m + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`
          const r = ret.get(date)
          weeks[row].cells[col] = { date, dom: d, r, holdout: !!split && date >= split }
          if (r !== undefined) mr.push(r)
        }
        for (const w of weeks) {
          const rs = w.cells.flatMap((c) => (c?.r !== undefined ? [c.r] : []))
          w.ret = rs.length ? compound(rs) : null
        }
        const kept = weeks.filter((w) => w.cells.some(Boolean))
        months.push({
          key: `${y}-${m}`,
          label: `${MONTHS[m]} ${y}`,
          weeks: kept,
          ret: compound(mr),
          up: mr.filter((r) => r > 0).length,
          down: mr.filter((r) => r < 0).length,
        })
      }
    }
    const rs = returns.map(([, r]) => r)
    const best = returns.reduce<[string, number] | null>((b, x) => (!b || x[1] > b[1] ? x : b), null)
    const worst = returns.reduce<[string, number] | null>((b, x) => (!b || x[1] < b[1] ? x : b), null)
    return {
      scale,
      cols,
      months,
      total: compound(rs),
      up: rs.filter((r) => r > 0).length,
      down: rs.filter((r) => r < 0).length,
      best,
      worst,
    }
  }, [returns, split])

  const s = cal?.summary
  const dayStats = (d: string): CalendarDay | undefined => cal?.days[d]
  const exposure = cal ? Object.values(cal.days).reduce((a, d) => a + d.exposure, 0) / Math.max(1, Object.keys(cal.days).length) : null
  const h = hover ? { d: hover, r: new Map(returns).get(hover), st: dayStats(hover) } : null
  const tone = (v: number | null | undefined) => (v == null ? '' : v >= 0 ? 'text-good' : 'text-bad')
  const pending = trades && !cal && !err

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Tile label="Net return" value={<span className={tone(model.total)}>{signedPct(model.total, 1)}</span>}
          sub={`${returns.length} trading days`} />
        <Tile label="Winning days" value={pct0(model.up / Math.max(1, model.up + model.down))}
          sub={`${model.up} up · ${model.down} down`} />
        <Tile label="Best day" value={<span className="text-good">{model.best ? signedPct(model.best[1]) : '—'}</span>}
          sub={model.best?.[0] ?? ''} onClick={model.best && onDay ? () => onDay(model.best![0]) : undefined} />
        <Tile label="Worst day" value={<span className="text-bad">{model.worst ? signedPct(model.worst[1]) : '—'}</span>}
          sub={model.worst?.[0] ?? ''} onClick={model.worst && onDay ? () => onDay(model.worst![0]) : undefined} />
        {trades && (
          <>
            <Tile label="Trades" value={s ? s.trades.toLocaleString() : '…'} sub={s ? `${s.long} long · ${s.short} short` : ''} />
            <Tile label="Trade win rate" value={s ? pct0(s.win_rate) : '…'}
              sub={s ? `long ${pct0(s.long_win_rate)} · short ${pct0(s.short_win_rate)}` : ''} />
            <Tile label="Profit factor" value={s ? (s.profit_factor?.toFixed(2) ?? '—') : '…'}
              sub={s ? `avg win ${s.avg_win != null ? signedPct(s.avg_win) : '—'} · loss ${s.avg_loss != null ? signedPct(s.avg_loss) : '—'}` : ''} />
            <Tile label="Avg hold" value={s ? fmtHold(s.avg_hold_s) : '…'}
              sub={exposure != null ? `in the market ${pct0(exposure)} of the time` : ''} />
          </>
        )}
      </div>

      <div className="flex min-h-[18px] flex-wrap items-center gap-x-3 font-mono text-[10.5px] text-ink-dim">
        {h ? (
          <>
            <span className="text-ink">{h.d}</span>
            {split && h.d >= split && <span className="text-accent">holdout</span>}
            {h.r !== undefined ? (
              <span>
                net <span className={tone(h.r)}>{signedPct(h.r)}</span>
              </span>
            ) : (
              <span className="text-ink-faint">no trading</span>
            )}
            {h.st && h.st.trades > 0 && (
              <>
                <span>
                  {h.st.trades} trade{h.st.trades === 1 ? '' : 's'} · {h.st.wins} won ({pct0(h.st.wins / h.st.trades)})
                </span>
                {h.st.long_wins != null ? (
                  <span className="flex items-center gap-2">
                    {splitTrades(h.st).map((p) => (
                      <span key={p.label} className="flex items-center gap-1" title={p.label}>
                        <span className="inline-block h-2 w-2 rounded-sm" style={{ background: p.fill }} />
                        {p.n}
                      </span>
                    ))}
                  </span>
                ) : (
                  <span>
                    {h.st.long}L / {h.st.short}S
                  </span>
                )}
                <span>
                  best <span className={tone(h.st.best)}>{h.st.best != null ? signedPct(h.st.best) : '—'}</span> · worst{' '}
                  <span className={tone(h.st.worst)}>{h.st.worst != null ? signedPct(h.st.worst) : '—'}</span>
                </span>
                <span>avg hold {fmtHold(h.st.hold_s)}</span>
              </>
            )}
            {h.st && <span>in market {pct0(h.st.exposure)}</span>}
          </>
        ) : (
          <span className="text-ink-faint">
            {err
              ? `trade statistics unavailable: ${err}`
              : pending && slow
                ? "recovering this candidate's positions for its trade statistics — a one-time re-run…"
                : onDay
                  ? 'hover a day for its trades · click it for the price chart and trades'
                  : 'hover a day for its P&L'}
          </span>
        )}
        {trades && (
          <span className="ml-auto flex items-center gap-2 text-ink-faint">
            {Object.values(HOLD).map((p) => (
              <span key={p.label} className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-sm" style={{ background: p.fill }} />
                {p.label}
              </span>
            ))}
          </span>
        )}
        <Legend scale={model.scale} />
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {model.months.map((m) => (
          <div key={m.key} className="rounded-xl border border-seam bg-panel-hi/30 p-2.5">
            <div className="mb-1.5 flex items-baseline gap-2 font-mono text-[11px]">
              <span className="text-ink">{m.label}</span>
              <span className={`ml-auto ${tone(m.ret)}`}>{m.up + m.down ? signedPct(m.ret, 1) : ''}</span>
              <span className="text-ink-faint">
                {m.up}↑ {m.down}↓
              </span>
            </div>
            <div className="grid gap-[3px]" style={{ gridTemplateColumns: `repeat(${model.cols}, minmax(0, 1fr)) 3.2rem` }}>
              {WEEKDAYS.slice(0, model.cols).map((w) => (
                <div key={w} className="text-center font-mono text-[9px] uppercase text-ink-faint">
                  {w}
                </div>
              ))}
              <div className="text-right font-mono text-[9px] uppercase text-ink-faint">week</div>
              {m.weeks.map((w, wi) => (
                <Week key={wi} w={w} scale={model.scale} selected={selected} stats={dayStats} trades={trades}
                  onDay={onDay} setHover={setHover} />
              ))}
            </div>
          </div>
        ))}
      </div>
      {cal?.recovered && (
        <div className="font-mono text-[10px] text-ink-faint">positions recovered from the {cal.recovered}; kept for next time</div>
      )}
    </div>
  )
}

function Week({
  w,
  scale,
  selected,
  stats,
  trades,
  onDay,
  setHover,
}: {
  w: Month['weeks'][number]
  scale: number
  selected: string | null
  stats: (d: string) => CalendarDay | undefined
  trades: boolean
  onDay?: (d: string) => void
  setHover: (d: string | null) => void
}) {
  return (
    <>
      {w.cells.map((c, ci) => {
        if (!c) return <div key={ci} />
        if (c.r === undefined) {
          return (
            <div key={ci} className="h-[54px] rounded-md bg-panel-hi/20 p-1 font-mono text-[9px] text-ink-faint/60">
              {c.dom}
            </div>
          )
        }
        const st = stats(c.date)
        const note = !trades
          ? ''
          : !st
            ? '·'
            : st.trades
              ? `${st.trades}t · ${Math.round((st.wins / st.trades) * 100)}%`
              : st.exposure > 0
                ? 'holding'
                : 'flat'
        return (
          <button
            key={ci}
            type="button"
            onClick={onDay ? () => onDay(c.date) : undefined}
            onMouseEnter={() => setHover(c.date)}
            onMouseLeave={() => setHover(null)}
            className={`relative flex h-[54px] flex-col justify-between rounded-md p-1 text-left transition-transform hover:z-10 hover:scale-[1.06] hover:shadow-lg ${
              onDay ? '' : 'cursor-default'
            } ${
              selected === c.date ? 'ring-2 ring-accent' : ''
            }`}
            style={{ background: tint(c.r, scale) }}
          >
            <span className="flex items-center justify-between font-mono text-[9px] text-ink-dim">
              {c.dom}
              {c.holdout && <span className="h-1.5 w-1.5 rounded-full bg-accent" title="holdout" />}
            </span>
            <span className="font-mono text-[11px] font-semibold tabular-nums text-ink">{cellPct(c.r)}</span>
            <span className="truncate font-mono text-[9px] text-ink-dim">{note}</span>
            {st && st.trades > 0 && st.long_wins != null && (
              <span className="absolute inset-x-1 bottom-0.5 flex h-[3px] overflow-hidden rounded-full">
                {splitTrades(st).map((p) => p.n > 0 && (
                  <span key={p.label} style={{ flex: p.n, background: p.fill }} />
                ))}
              </span>
            )}
          </button>
        )
      })}
      <div className="flex h-[54px] flex-col items-end justify-center font-mono text-[10px] tabular-nums">
        {w.ret !== null && <span className={w.ret >= 0 ? 'text-good' : 'text-bad'}>{cellPct(w.ret)}</span>}
      </div>
    </>
  )
}

function Tile({ label, value, sub, onClick }: { label: string; value: React.ReactNode; sub: string; onClick?: () => void }) {
  const body = (
    <>
      <div className="text-[10.5px] uppercase tracking-wide text-ink-faint">{label}</div>
      <div className="mt-0.5 font-mono text-[18px] text-ink">{value}</div>
      <div className="truncate font-mono text-[10.5px] text-ink-faint">{sub}</div>
    </>
  )
  const cls = 'rounded-xl border border-seam bg-panel-hi/40 p-3 text-left'
  return onClick ? (
    <button type="button" onClick={onClick} className={`${cls} hover:border-ink-faint`} title="open this day">
      {body}
    </button>
  ) : (
    <div className={cls}>{body}</div>
  )
}

/** The diverging scale, clipped at the 95th-percentile day. */
function Legend({ scale }: { scale: number }) {
  return (
    <span className="ml-auto flex items-center gap-1.5 text-ink-faint">
      {signedPct(-scale, 1)}
      <span
        className="inline-block h-2 w-24 rounded-full"
        style={{
          background: `linear-gradient(to right, ${tint(-scale, scale)}, var(--color-panel-hi), ${tint(scale, scale)})`,
        }}
      />
      {signedPct(scale, 1)}
    </span>
  )
}
