'use client'

import { useEffect, useMemo, useRef } from 'react'
import { fmtMetric, type MetricKind, type Point } from '@/lib/objectives'

const W = 640
const PAD = { l: 44, r: 10, t: 10, b: 20 }

function niceRange(values: number[]): [number, number] {
  let lo = Math.min(...values)
  let hi = Math.max(...values)
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1]
  if (lo === hi) {
    lo -= Math.abs(lo) * 0.1 || 1
    hi += Math.abs(hi) * 0.1 || 1
  }
  const pad = (hi - lo) * 0.08
  return [lo - pad, hi + pad]
}

/** Placed on the score axis: scored and not rejected. A rejected cheat scoring 40 would
 *  otherwise stretch the axis until every honest candidate is a flat line at the bottom. */
function onAxis(p: Point): boolean {
  return (
    p.status === 'ok' &&
    p.score !== null &&
    Number.isFinite(p.score) &&
    p.lookahead !== 'fail' &&
    p.audit !== 'fail'
  )
}

/**
 * Every candidate as a dot at its score, in submission order, with the best-so-far as a
 * step line: the shape of the search -- a line that keeps stepping up is a swarm that is
 * still learning; a flat line under a cloud of dots is one that has stalled.
 *
 * Failures cannot be placed on the score axis, so they sit on a strip under the plot:
 * red for look-ahead rejections (the ones worth noticing), grey for scripts that crashed.
 */
export function ProgressChart({
  points,
  kind,
  higher,
  height = 200,
  onPick,
}: {
  points: Point[]
  kind: MetricKind
  higher: boolean
  height?: number
  onPick?: (id: string) => void
}) {
  const H = height
  const plotB = H - PAD.b - 14 // leave a strip for failures
  // Each candidate gets a fixed-width column, so a long run scrolls instead of squeezing
  // hundreds of candidates into one unreadable smear.
  const maxSeq = Math.max(1, ...points.map((p) => p.seq))
  const W = Math.max(640, PAD.l + PAD.r + maxSeq * 14)
  const scroller = useRef<HTMLDivElement>(null)
  const follow = useRef(true)

  const model = useMemo(() => {
    const scored = points.filter(onAxis)
    const [lo, hi] = niceRange(scored.length ? scored.map((p) => p.score as number) : [0, 1])
    const x = (seq: number) => PAD.l + ((seq - 0.5) / maxSeq) * (W - PAD.l - PAD.r)
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (plotB - PAD.t)
    // Best-so-far over the champions only (audited title holders), as a step line.
    let best: number | null = null
    const steps: string[] = []
    for (const p of [...points].sort((a, b) => a.seq - b.seq)) {
      if (p.champion_at && p.score !== null) {
        if (best === null || (higher ? p.score > best : p.score < best)) {
          if (best !== null) steps.push(`L${x(p.seq)},${y(best)}`)
          best = p.score
          steps.push(`${steps.length ? 'L' : 'M'}${x(p.seq)},${y(best)}`)
        }
      }
    }
    if (best !== null) steps.push(`L${x(maxSeq + 0.5)},${y(best)}`)
    const ticks = [lo, (lo + hi) / 2, hi]
    return { x, y, path: steps.join(' '), ticks }
  }, [points, higher, plotB, W, maxSeq])

  // Follow the newest candidates -- unless the operator scrolled back to look at history.
  useEffect(() => {
    const el = scroller.current
    if (el && follow.current) el.scrollLeft = el.scrollWidth
  }, [W])

  if (!points.length) {
    return (
      <div className="grid h-[120px] place-items-center rounded-xl border border-dashed border-seam text-[12px] text-ink-faint">
        No candidates yet — agents submit their first ones within a few minutes of starting.
      </div>
    )
  }

  const scrolls = W > 640
  return (
    <div className="relative">
      <div
        ref={scroller}
        className="overflow-x-auto"
        onScroll={(e) => {
          const el = e.currentTarget
          follow.current = el.scrollLeft + el.clientWidth >= el.scrollWidth - 24
        }}
      >
        <svg
          viewBox={`0 0 ${W} ${H}`}
          width={scrolls ? W : undefined}
          height={scrolls ? H : undefined}
          className={scrolls ? 'block' : 'block w-full'}
          role="img"
          aria-label="candidate scores over time"
        >
          {model.ticks.map((t, i) => (
            <line key={i} x1={PAD.l} x2={W - PAD.r} y1={model.y(t)} y2={model.y(t)} className="stroke-seam" strokeWidth={1} />
          ))}
          <line x1={PAD.l} x2={W - PAD.r} y1={plotB + 7} y2={plotB + 7} className="stroke-seam" strokeDasharray="2 3" />
          {model.path && <path d={model.path} fill="none" className="stroke-good" strokeWidth={2} />}
          {points.map((p) => {
            const cx = model.x(p.seq)
            if (onAxis(p)) {
              const cy = model.y(p.score as number)
              const tone = p.champion_at ? 'fill-good' : p.audit === 'pending' ? 'fill-warn' : 'fill-accent/70'
              return (
                <circle
                  key={p.id}
                  cx={cx}
                  cy={cy}
                  r={p.champion_at ? 4.5 : 3}
                  className={`${tone} ${onPick ? 'cursor-pointer' : ''}`}
                  onClick={onPick ? () => onPick(p.id) : undefined}
                >
                  <title>
                    #{p.seq} · {fmtMetric(kind, p.score)} · {p.model}
                    {p.champion_at ? ' · champion' : ''}
                  </title>
                </circle>
              )
            }
            const bad = p.lookahead === 'fail' || p.audit === 'fail'
            return (
              <rect
                key={p.id}
                x={cx - 2}
                y={plotB + 5}
                width={4}
                height={4}
                className={`${bad ? 'fill-bad' : p.status === 'evaluating' ? 'fill-warn' : 'fill-ink-faint/60'} ${onPick ? 'cursor-pointer' : ''}`}
                onClick={onPick ? () => onPick(p.id) : undefined}
              >
                <title>
                  #{p.seq} · {p.status === 'evaluating' ? 'evaluating' : p.lookahead === 'fail' ? 'rejected: look-ahead' : p.audit === 'fail' ? 'rejected by audit' : 'failed / unranked'} · {p.model}
                </title>
              </rect>
            )
          })}
          {/* Candidate numbers along the bottom, every 10th so they stay legible. */}
          {Array.from({ length: Math.floor(maxSeq / 10) + 1 }, (_, i) => Math.max(1, i * 10)).map((n) => (
            <text key={n} x={model.x(n)} y={H - 4} textAnchor="middle" className="fill-ink-faint font-mono text-[9px]">
              #{n}
            </text>
          ))}
        </svg>
      </div>
      {/* The score axis stays put while the candidates scroll under it. */}
      <svg
        className="pointer-events-none absolute left-0 top-0"
        width={PAD.l}
        height={H}
        viewBox={`0 0 ${PAD.l} ${H}`}
        style={scrolls ? undefined : { width: `${(PAD.l / W) * 100}%`, height: 'auto' }}
      >
        <rect x={0} y={0} width={PAD.l - 2} height={H} className="fill-panel" />
        {model.ticks.map((t, i) => (
          <text key={i} x={PAD.l - 6} y={model.y(t) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
            {fmtMetric(kind, t)}
          </text>
        ))}
      </svg>
      {scrolls && (
        <div className="mt-0.5 text-right font-mono text-[9.5px] text-ink-faint">
          scroll ← for earlier candidates · {maxSeq} total
        </div>
      )}
    </div>
  )
}

/**
 * Growth of 1 over the candidate's daily returns, in-sample and holdout shaded apart, so the
 * question "did it keep working after the split?" is answered by eye.
 */
export function EquityCurve({
  returns,
  split,
  height = 200,
}: {
  returns: [string, number][]
  split: string | null
  height?: number
}) {
  const H = height
  const model = useMemo(() => {
    let eq = 1
    const pts = returns.map(([d, r]) => {
      eq *= 1 + r
      return { d, eq }
    })
    const [lo, hi] = niceRange(pts.map((p) => p.eq).concat([1]))
    const n = Math.max(1, pts.length - 1)
    const x = (i: number) => PAD.l + (i / n) * (W - PAD.l - PAD.r)
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b)
    const splitIdx = split ? pts.findIndex((p) => p.d >= split) : -1
    const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.eq).toFixed(1)}`).join(' ')
    return { pts, x, y, lo, hi, splitIdx, path }
  }, [returns, split, H])

  if (returns.length < 2) {
    return <div className="text-[12px] text-ink-faint">No return stream recorded.</div>
  }
  const { pts, x, y, lo, hi, splitIdx, path } = model
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="equity curve">
      {splitIdx > 0 && (
        <>
          <rect
            x={x(splitIdx)}
            y={PAD.t}
            width={W - PAD.r - x(splitIdx)}
            height={H - PAD.t - PAD.b}
            className="fill-accent/[0.07]"
          />
          <line x1={x(splitIdx)} x2={x(splitIdx)} y1={PAD.t} y2={H - PAD.b} className="stroke-accent" strokeDasharray="3 3" />
          <text x={x(splitIdx) + 4} y={PAD.t + 10} className="fill-accent font-mono text-[9px]">
            holdout (hidden from agents)
          </text>
        </>
      )}
      <line x1={PAD.l} x2={W - PAD.r} y1={y(1)} y2={y(1)} className="stroke-seam" />
      {[lo, hi].map((t, i) => (
        <text key={i} x={PAD.l - 6} y={y(t) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
          {t.toFixed(2)}
        </text>
      ))}
      <path d={path} fill="none" className="stroke-good" strokeWidth={1.6} />
      <text x={PAD.l} y={H - 4} className="fill-ink-faint font-mono text-[9px]">
        {pts[0].d}
      </text>
      <text x={W - PAD.r} y={H - 4} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
        {pts[pts.length - 1].d}
      </text>
    </svg>
  )
}
