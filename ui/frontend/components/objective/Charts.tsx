'use client'

import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { fmtMetric, type MetricKind, type Point, type RegimeInfo } from '@/lib/objectives'

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

/** A score range that ignores far outliers: Tukey fences at 3x the interquartile range,
 *  widened to keep `keep` (the best so far) in view. Falls back to the full range when the
 *  middle of the distribution is a single value. */
function robustRange(values: number[], keep: number | null): [number, number] {
  const s = [...values].sort((a, b) => a - b)
  if (s.length < 4) return niceRange(s.length ? s : [0, 1])
  const q = (f: number) => s[Math.round(f * (s.length - 1))]
  const iqr = q(0.75) - q(0.25)
  if (!(iqr > 0)) return niceRange(s)
  let lo = Math.max(s[0], q(0.25) - 3 * iqr)
  let hi = Math.min(s[s.length - 1], q(0.75) + 3 * iqr)
  if (keep !== null) {
    lo = Math.min(lo, keep)
    hi = Math.max(hi, keep)
  }
  return niceRange([lo, hi])
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
  storageKey,
}: {
  points: Point[]
  kind: MetricKind
  higher: boolean
  height?: number
  onPick?: (id: string) => void
  /** Remember the zoom in localStorage under `ft-zoom-<key>`, so it survives leaving the page. */
  storageKey?: string
}) {
  const H = height
  const plotB = H - PAD.b - 14 // leave a strip for failures
  // Each candidate gets a fixed-width column, so a long run scrolls instead of squeezing
  // hundreds of candidates into one unreadable smear.
  const maxSeq = Math.max(1, ...points.map((p) => p.seq))
  const W = Math.max(640, PAD.l + PAD.r + maxSeq * 14)
  const scroller = useRef<HTMLDivElement>(null)
  const axis = useRef<SVGSVGElement>(null)
  const follow = useRef(true)
  // Whether the newest candidates are in view; only flips at the ends, so scrolling stays cheap.
  const [atEnd, setAtEnd] = useState(true)
  const spanLabel = useRef<HTMLSpanElement>(null)
  // Dragging the plot pans through the history; a drag must not also count as a click on a dot.
  const pan = useRef<{ x: number; left: number; moved: boolean } | null>(null)
  const dragged = useRef(false)
  const clipId = useId()
  // The operator's vertical zoom; null is the automatic range over every scored candidate.
  // It is stored with the metric it was set on, so a range for one metric never lands on
  // another. Read after mount, so the server render and the first client render agree.
  const zoomKey = storageKey ? `ft-zoom-${storageKey}` : null
  const [view, setViewState] = useState<[number, number] | null>(null)
  useEffect(() => {
    let saved: [number, number] | null = null
    try {
      const v = zoomKey ? JSON.parse(localStorage.getItem(zoomKey) ?? 'null') : null
      if (v && v.kind === kind && Number.isFinite(v.lo) && Number.isFinite(v.hi) && v.hi > v.lo) saved = [v.lo, v.hi]
    } catch {}
    setViewState(saved)
  }, [zoomKey, kind])
  function setView(v: [number, number] | null) {
    setViewState(v)
    if (!zoomKey) return
    try {
      if (v) localStorage.setItem(zoomKey, JSON.stringify({ kind, lo: v[0], hi: v[1] }))
      else localStorage.removeItem(zoomKey)
    } catch {}
  }

  const model = useMemo(() => {
    const scored = points.filter(onAxis)
    const values = scored.length ? scored.map((p) => p.score as number) : [0, 1]
    const full = niceRange(values)
    const [lo, hi] = view ?? full
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
    const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4)
    const hidden = view ? values.filter((v) => v < lo || v > hi).length : 0
    return { x, y, lo, hi, full, values, best, path: steps.join(' '), ticks, hidden }
  }, [points, higher, plotB, W, maxSeq, view])

  // The wheel and drag handlers are attached natively and read the latest range from here.
  const range = useRef<[number, number]>([model.lo, model.hi])
  range.current = [model.lo, model.hi]

  /** Scale the visible range by `f` (<1 zooms in) around `anchor` -- by default the best so
   *  far, the value worth looking at closely. Zooming out past the automatic range snaps to it. */
  function zoom(f: number, anchor?: number) {
    const [lo, hi] = range.current
    const a = anchor ?? (model.best !== null && model.best >= lo && model.best <= hi ? model.best : (lo + hi) / 2)
    const next: [number, number] = [a - (a - lo) * f, a + (hi - a) * f]
    const [flo, fhi] = model.full
    if (next[1] - next[0] >= fhi - flo) setView(null)
    else if (next[1] - next[0] > 1e-9) setView(next)
  }

  /** Screen y (client px) on the axis to a score. */
  function scoreAt(clientY: number): number | undefined {
    const el = axis.current
    if (!el) return undefined
    const box = el.getBoundingClientRect()
    const vy = ((clientY - box.top) / box.height) * H
    const [lo, hi] = range.current
    return lo + (1 - (vy - PAD.t) / (plotB - PAD.t)) * (hi - lo)
  }

  // Wheel over the score axis zooms at the cursor. React's wheel listener is passive, so it
  // cannot stop the page scrolling; attach one that can.
  const onWheel = useRef<(e: WheelEvent) => void>(() => {})
  onWheel.current = (e) => {
    e.preventDefault()
    zoom(Math.exp(e.deltaY * 0.0015), scoreAt(e.clientY))
  }
  const hasPoints = points.length > 0
  useEffect(() => {
    const el = axis.current
    if (!el) return
    const handler = (e: WheelEvent) => onWheel.current(e)
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [hasPoints])

  // Dragging the score axis pans the range.
  const drag = useRef<{ y: number; lo: number; hi: number; px: number } | null>(null)
  function onAxisDown(e: React.PointerEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    e.currentTarget.setPointerCapture(e.pointerId)
    const [lo, hi] = range.current
    drag.current = { y: e.clientY, lo, hi, px: ((plotB - PAD.t) / H) * box.height }
  }
  function onAxisMove(e: React.PointerEvent<SVGSVGElement>) {
    const d = drag.current
    if (!d) return
    const dv = ((e.clientY - d.y) / d.px) * (d.hi - d.lo)
    if (dv !== 0) setView([d.lo + dv, d.hi + dv])
  }

  /** Which candidates are on screen, written straight into the label (no re-render per scroll). */
  function showSpan(el: HTMLDivElement) {
    const seqAt = (px: number) => Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * maxSeq + 0.5)
    const a = Math.max(1, seqAt(el.scrollLeft + PAD.l))
    const b = Math.min(maxSeq, seqAt(el.scrollLeft + el.clientWidth))
    if (spanLabel.current) spanLabel.current.textContent = `#${a}–#${b} of ${maxSeq}`
  }
  function onScrolled(el: HTMLDivElement) {
    follow.current = el.scrollLeft + el.clientWidth >= el.scrollWidth - 24
    setAtEnd(follow.current)
    showSpan(el)
  }
  function page(dir: -1 | 1) {
    const el = scroller.current
    if (el) el.scrollBy({ left: dir * el.clientWidth * 0.8, behavior: 'smooth' })
  }
  function latest() {
    const el = scroller.current
    if (!el) return
    follow.current = true
    el.scrollTo({ left: el.scrollWidth, behavior: 'smooth' })
  }

  // Follow the newest candidates -- unless the operator scrolled back to look at history.
  useEffect(() => {
    const el = scroller.current
    if (el && follow.current) el.scrollLeft = el.scrollWidth
    if (el) showSpan(el)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [W])

  if (!points.length) {
    return (
      <div className="grid h-[120px] place-items-center rounded-xl border border-dashed border-seam text-[12px] text-ink-faint">
        No candidates yet — agents submit their first ones within a few minutes of starting.
      </div>
    )
  }

  const scrolls = W > 640
  const btn =
    'rounded border border-seam px-1.5 leading-[16px] text-ink-dim hover:border-ink-faint hover:text-ink disabled:opacity-40 disabled:hover:border-seam disabled:hover:text-ink-dim'
  return (
    <div className="relative">
      <div
        ref={scroller}
        className={`overflow-x-auto ${W > 640 ? 'cursor-grab active:cursor-grabbing' : ''}`}
        onScroll={(e) => onScrolled(e.currentTarget)}
        onPointerDown={(e) => {
          if (W <= 640 || e.button !== 0) return
          pan.current = { x: e.clientX, left: e.currentTarget.scrollLeft, moved: false }
          dragged.current = false
        }}
        onPointerMove={(e) => {
          const p = pan.current
          if (!p) return
          const dx = e.clientX - p.x
          if (!p.moved && Math.abs(dx) > 4) {
            p.moved = dragged.current = true
            e.currentTarget.setPointerCapture(e.pointerId)
          }
          if (p.moved) e.currentTarget.scrollLeft = p.left - dx
        }}
        onPointerUp={() => (pan.current = null)}
        onPointerCancel={() => (pan.current = null)}
        onClickCapture={(e) => {
          if (dragged.current) {
            e.stopPropagation()
            dragged.current = false
          }
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
          <defs>
            <clipPath id={clipId}>
              <rect x={PAD.l} y={PAD.t - 5} width={W - PAD.l - PAD.r} height={plotB - PAD.t + 10} />
            </clipPath>
          </defs>
          {model.ticks.map((t, i) => (
            <line key={i} x1={PAD.l} x2={W - PAD.r} y1={model.y(t)} y2={model.y(t)} className="stroke-seam" strokeWidth={1} />
          ))}
          <line x1={PAD.l} x2={W - PAD.r} y1={plotB + 7} y2={plotB + 7} className="stroke-seam" strokeDasharray="2 3" />
          {model.path && <path d={model.path} fill="none" className="stroke-good" strokeWidth={2} clipPath={`url(#${clipId})`} />}
          {points.map((p) => {
            const cx = model.x(p.seq)
            if (onAxis(p)) {
              const v = p.score as number
              // Orange: its look-ahead test could not run (interrupted, crashed) -- scored, but unverified.
              const tone = p.champion_at
                ? 'fill-good'
                : p.lookahead === 'error'
                  ? 'fill-[#f97316]'
                  : p.audit === 'pending'
                    ? 'fill-warn'
                    : 'fill-accent/70'
              const label = (
                <title>
                  #{p.seq} · {fmtMetric(kind, p.score)} · {p.model}
                  {p.champion_at ? ' · champion' : ''}
                  {p.lookahead === 'error' ? ' · look-ahead test could not run -- re-run it' : ''}
                </title>
              )
              // Zoomed past it: a faint caret on the edge it lies beyond, so outliers stay visible.
              if (v > model.hi || v < model.lo) {
                const up = v > model.hi
                const ey = up ? PAD.t : plotB
                return (
                  <path
                    key={p.id}
                    d={up ? `M${cx - 3},${ey + 3}L${cx},${ey - 2}L${cx + 3},${ey + 3}Z` : `M${cx - 3},${ey - 3}L${cx},${ey + 2}L${cx + 3},${ey - 3}Z`}
                    className={`${tone} opacity-50 ${onPick ? 'cursor-pointer' : ''}`}
                    onClick={onPick ? () => onPick(p.id) : undefined}
                  >
                    {label}
                  </path>
                )
              }
              return (
                <circle
                  key={p.id}
                  cx={cx}
                  cy={model.y(v)}
                  r={p.champion_at ? 4.5 : 3}
                  className={`${tone} ${onPick ? 'cursor-pointer' : ''}`}
                  onClick={onPick ? () => onPick(p.id) : undefined}
                >
                  {label}
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
      {/* The score axis stays put while the candidates scroll under it. Wheel over it to zoom
          the scores, drag it to pan, double-click to reset. */}
      <svg
        ref={axis}
        className="absolute left-0 top-0 cursor-ns-resize touch-none select-none"
        width={PAD.l}
        height={H}
        viewBox={`0 0 ${PAD.l} ${H}`}
        style={scrolls ? undefined : { width: `${(PAD.l / W) * 100}%`, height: 'auto' }}
        onPointerDown={onAxisDown}
        onPointerMove={onAxisMove}
        onPointerUp={() => (drag.current = null)}
        onPointerCancel={() => (drag.current = null)}
        onDoubleClick={() => setView(null)}
      >
        <title>wheel to zoom the score axis · drag to pan · double-click to reset</title>
        <rect x={0} y={0} width={PAD.l - 2} height={H} className="fill-panel" />
        {model.ticks.map((t, i) => (
          <text key={i} x={PAD.l - 6} y={model.y(t) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
            {fmtMetric(kind, t)}
          </text>
        ))}
      </svg>
      <div className="mt-0.5 flex items-center gap-1 font-mono text-[9.5px] text-ink-faint">
        <button type="button" className={btn} onClick={() => zoom(0.5)} title="zoom in on the best so far">
          +
        </button>
        <button type="button" className={btn} onClick={() => zoom(2)} disabled={!view} title="zoom out">
          −
        </button>
        <button
          type="button"
          className={btn}
          onClick={() => {
            const r = robustRange(model.values, model.best)
            const [flo, fhi] = model.full
            setView(r[1] - r[0] >= fhi - flo ? null : r)
          }}
          title="fit the score axis to the bulk of candidates, ignoring far outliers"
        >
          fit
        </button>
        {view && (
          <button type="button" className={btn} onClick={() => setView(null)} title="show every scored candidate">
            reset
          </button>
        )}
        <span className="ml-1">{view ? `zoomed · ${model.hidden} off-scale` : 'wheel / drag the score axis to zoom'}</span>
        {scrolls && (
          <span className="ml-auto flex items-center gap-1">
            <span ref={spanLabel} className="mr-1" />
            <button type="button" className={btn} onClick={() => page(-1)} title="earlier candidates (or drag the plot)">
              ‹ older
            </button>
            <button type="button" className={btn} onClick={() => page(1)} disabled={atEnd} title="later candidates">
              newer ›
            </button>
            <button type="button" className={btn} onClick={latest} disabled={atEnd} title="back to the newest candidates">
              latest ⇥
            </button>
          </span>
        )}
      </div>
    </div>
  )
}

/**
 * Growth of 1 over the candidate's daily returns, in-sample and holdout shaded apart, so the
 * question "did it keep working after the split?" is answered by eye. The hovered day's values
 * read out under the plot, where they cover nothing; clicking a day hands it to `onDay`.
 */
export function EquityCurve({
  returns,
  split,
  height = 200,
  onDay,
  additive = false,
}: {
  returns: [string, number][]
  split: string | null
  height?: number
  onDay?: (day: string) => void
  /** The values are amounts to SUM (a task's daily profit or score), not returns to compound. */
  additive?: boolean
}) {
  const H = height
  const [hover, setHover] = useState<number | null>(null)
  const base = additive ? 0 : 1
  const model = useMemo(() => {
    let eq = base
    const pts = returns.map(([d, r]) => {
      eq = additive ? eq + r : eq * (1 + r)
      return { d, eq, r }
    })
    const [lo, hi] = niceRange(pts.map((p) => p.eq).concat([base]))
    const n = Math.max(1, pts.length - 1)
    const x = (i: number) => PAD.l + (i / n) * (W - PAD.l - PAD.r)
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b)
    const splitIdx = split ? pts.findIndex((p) => p.d >= split) : -1
    const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.eq).toFixed(1)}`).join(' ')
    return { pts, x, y, lo, hi, splitIdx, path }
  }, [returns, split, H, additive, base])

  if (returns.length < 2) {
    return <div className="text-[12px] text-ink-faint">No return stream recorded.</div>
  }
  const { pts, x, y, lo, hi, splitIdx, path } = model
  const num = (v: number) => `${v >= 0 ? '+' : ''}${Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)}`
  const n = Math.max(1, pts.length - 1)
  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    const vx = ((e.clientX - box.left) / box.width) * W
    const i = Math.round(((vx - PAD.l) / (W - PAD.l - PAD.r)) * n)
    setHover(Math.max(0, Math.min(pts.length - 1, i)))
  }
  const h = hover !== null ? pts[hover] : null
  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className={`w-full ${onDay ? 'cursor-pointer' : 'cursor-crosshair'}`}
        role="img"
        aria-label="equity curve"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={onDay && h ? () => onDay(h.d) : undefined}
      >
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
        <line x1={PAD.l} x2={W - PAD.r} y1={y(base)} y2={y(base)} className="stroke-seam" />
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
        {h && hover !== null && <Crosshair x={x(hover)} y={y(h.eq)} H={H} />}
      </svg>
      <div className="flex min-h-[18px] flex-wrap items-center gap-x-3 font-mono text-[10.5px] text-ink-dim">
        {h ? (
          <>
            <span className="text-ink">{h.d}</span>
            {splitIdx > 0 && hover !== null && hover >= splitIdx && <span className="text-accent">holdout</span>}
            <span>
              total{' '}
              <span className={h.eq >= base ? 'text-good' : 'text-bad'}>{additive ? num(h.eq) : signedPct(h.eq - 1, 1)}</span>
            </span>
            <span>
              day <span className={h.r >= 0 ? 'text-good' : 'text-bad'}>{additive ? num(h.r) : signedPct(h.r)}</span>
            </span>
            {!additive && <span>equity {h.eq.toFixed(3)}</span>}
            {onDay && <span className="ml-auto text-ink-faint">click for the day's bars and positions</span>}
          </>
        ) : (
          <span className="text-ink-faint">
            hover for each day's values{onDay ? ' · click a day for its bars and positions' : ''}
          </span>
        )}
      </div>
    </div>
  )
}

const signedPct = (v: number, digits = 2) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`

/** The hovered day on an equity curve: guide lines through the point. Its values read out
 *  under the plot, so nothing is drawn over the curve. */
function Crosshair({ x, y, H }: { x: number; y: number; H: number }) {
  return (
    <g pointerEvents="none">
      <line x1={x} x2={x} y1={PAD.t} y2={H - PAD.b} className="stroke-ink-dim" strokeWidth={0.8} />
      <line x1={PAD.l} x2={W - PAD.r} y1={y} y2={y} className="stroke-ink-dim" strokeWidth={0.8} strokeDasharray="3 3" />
      <circle cx={x} cy={y} r={3.5} className="fill-good stroke-panel" strokeWidth={1.5} />
    </g>
  )
}

const SERIES = 8
const seriesColor = (i: number) => (i >= 0 && i < SERIES ? `var(--color-series-${i + 1})` : 'var(--color-ink-faint)')

/**
 * The equity curve coloured by the regime each day was in (so by the signal traded then),
 * with the series the regime was derived from plotted beneath on the same time axis -- the
 * question "which signal made the money, when, and was the regime switch right?" by eye.
 * One crosshair spans both plots. Colours follow the label, in fixed order, never the rank.
 */
export function RegimeCurves({
  returns,
  split,
  regime,
  onDay,
}: {
  returns: [string, number][]
  split: string | null
  regime: RegimeInfo
  onDay?: (day: string) => void
}) {
  const [hover, setHover] = useState<number | null>(null)
  const H1 = 200
  const H2 = 110
  const model = useMemo(() => {
    const dayLabel = new Map(regime.days)
    const sig = new Map(regime.signal ?? [])
    const labels = [
      ...Object.keys(regime.routes),
      ...[...new Set(regime.days.map(([, l]) => l))].filter((l) => !(l in regime.routes) && l !== 'warmup').sort(),
    ]
    const color = (l: string | undefined) =>
      l === undefined || l === 'warmup' ? 'var(--color-ink-faint)' : seriesColor(labels.indexOf(l))
    let eq = 1
    const pts = returns.map(([d, r]) => {
      eq *= 1 + r
      return { d, eq, r, label: dayLabel.get(d), v: sig.get(d) }
    })
    const n = Math.max(1, pts.length - 1)
    const x = (i: number) => PAD.l + (i / n) * (W - PAD.l - PAD.r)
    const [lo, hi] = niceRange(pts.map((p) => p.eq).concat([1]))
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H1 - PAD.t - PAD.b)
    const vals = pts.map((p) => p.v).filter((v): v is number => v !== undefined && Number.isFinite(v))
    const [slo, shi] = niceRange(vals.length ? vals : [0, 1])
    const ys = (v: number) => PAD.t + (1 - (v - slo) / (shi - slo)) * (H2 - PAD.t - PAD.b)
    // Runs of consecutive days in one regime: one coloured path (and one band) per run.
    const runs: { label: string | undefined; from: number; to: number }[] = []
    pts.forEach((p, i) => {
      const last = runs[runs.length - 1]
      if (last && last.label === p.label) last.to = i
      else runs.push({ label: p.label, from: i, to: i })
    })
    // Each run starts from the previous day's point so the coloured pieces join up.
    const eqPath = (from: number, to: number) => {
      const start = Math.max(0, from - 1)
      return pts
        .slice(start, to + 1)
        .map((p, k) => `${k ? 'L' : 'M'}${x(start + k).toFixed(1)},${y(p.eq).toFixed(1)}`)
        .join(' ')
    }
    let sigPath = ''
    let pen = false
    pts.forEach((p, i) => {
      if (p.v === undefined || !Number.isFinite(p.v)) {
        pen = false
        return
      }
      sigPath += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${ys(p.v).toFixed(1)}`
      pen = true
    })
    const splitIdx = split ? pts.findIndex((p) => p.d >= split) : -1
    return { pts, x, y, ys, lo, hi, slo, shi, runs, eqPath, sigPath, labels, color, splitIdx, n }
  }, [returns, split, regime])

  if (returns.length < 2) return <div className="text-[12px] text-ink-faint">No return stream recorded.</div>
  const { pts, x, y, ys, lo, hi, slo, shi, runs, eqPath, sigPath, labels, color, splitIdx, n } = model

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    const vx = ((e.clientX - box.left) / box.width) * W
    const i = Math.round(((vx - PAD.l) / (W - PAD.l - PAD.r)) * n)
    setHover(i >= 0 && i < pts.length ? i : null)
  }
  const h = hover !== null ? pts[hover] : null

  const splitShade = (height: number, caption: boolean) =>
    splitIdx > 0 && (
      <>
        <rect
          x={x(splitIdx)}
          y={PAD.t}
          width={W - PAD.r - x(splitIdx)}
          height={height - PAD.t - PAD.b}
          className="fill-accent/[0.05]"
        />
        <line x1={x(splitIdx)} x2={x(splitIdx)} y1={PAD.t} y2={height - PAD.b} className="stroke-accent" strokeDasharray="3 3" />
        {caption && (
          <text x={x(splitIdx) + 4} y={PAD.t + 10} className="fill-accent font-mono text-[9px]">
            holdout (hidden from agents)
          </text>
        )}
      </>
    )
  const crosshair = (height: number) =>
    hover !== null && (
      <line x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={height - PAD.b} className="stroke-ink-dim" strokeWidth={1} />
    )
  const rows = [...labels, ...(regime.by_label.warmup ? ['warmup'] : [])]

  return (
    <div className="space-y-1">
      <div className="flex min-h-[18px] flex-wrap items-center gap-x-3 font-mono text-[10.5px] text-ink-dim">
        {h ? (
          <>
            <span>{h.d}</span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: color(h.label) }} />
              <span className="text-ink">{h.label ?? '—'}</span>
              {h.label && regime.routes[h.label] && <span>→ {regime.routes[h.label]}</span>}
            </span>
            <span>
              total {signedPct(h.eq - 1, 1)} · day {signedPct(h.r)}
            </span>
            {h.v !== undefined && (
              <span>
                {regime.name} {h.v.toPrecision(4)}
              </span>
            )}
          </>
        ) : (
          <span className="text-ink-faint">
            hover either chart for the day, its regime and the regime signal
            {onDay ? ' · click a day for its bars and positions' : ''}
          </span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H1}`}
        className={`w-full ${onDay ? 'cursor-pointer' : ''}`}
        role="img"
        aria-label="equity curve coloured by regime"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={onDay && h ? () => onDay(h.d) : undefined}
      >
        {splitShade(H1, true)}
        <line x1={PAD.l} x2={W - PAD.r} y1={y(1)} y2={y(1)} className="stroke-seam" />
        {[lo, hi].map((t, i) => (
          <text key={i} x={PAD.l - 6} y={y(t) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
            {t.toFixed(2)}
          </text>
        ))}
        {runs.map((r, i) => (
          <path key={i} d={eqPath(r.from, r.to)} fill="none" stroke={color(r.label)} strokeWidth={2} strokeLinejoin="round" />
        ))}
        {crosshair(H1)}
        {h && hover !== null && (
          <circle cx={x(hover)} cy={y(h.eq)} r={4} fill={color(h.label)} className="stroke-panel" strokeWidth={2} />
        )}
      </svg>
      <div className="font-mono text-[10px] uppercase tracking-wide text-ink-faint">
        regime signal · {regime.name}
        {!regime.signal?.length && <span className="normal-case"> — report it with ft.report_regime(labels, signal=…) to plot it</span>}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H2}`}
        className={`w-full ${onDay ? 'cursor-pointer' : ''}`}
        role="img"
        aria-label="regime signal with regime bands"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={onDay && h ? () => onDay(h.d) : undefined}
      >
        {runs.map((r, i) => {
          const x0 = x(Math.max(0, r.from - 0.5))
          return (
            <rect
              key={i}
              x={x0}
              y={PAD.t}
              width={Math.max(1, x(Math.min(n, r.to + 0.5)) - x0)}
              height={H2 - PAD.t - PAD.b}
              fill={color(r.label)}
              opacity={0.18}
            />
          )
        })}
        {splitShade(H2, false)}
        {regime.signal?.length ? (
          <>
            {[slo, shi].map((t, i) => (
              <text key={i} x={PAD.l - 6} y={ys(t) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
                {t.toPrecision(3)}
              </text>
            ))}
            <path d={sigPath} fill="none" className="stroke-ink" strokeWidth={1.5} />
            {h?.v !== undefined && hover !== null && (
              <circle cx={x(hover)} cy={ys(h.v)} r={4} className="fill-ink stroke-panel" strokeWidth={2} />
            )}
          </>
        ) : null}
        {crosshair(H2)}
        <text x={PAD.l} y={H2 - 4} className="fill-ink-faint font-mono text-[9px]">
          {pts[0].d}
        </text>
        <text x={W - PAD.r} y={H2 - 4} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
          {pts[pts.length - 1].d}
        </text>
      </svg>
      <table className="mt-2 w-full font-mono text-[11px]">
        <thead className="text-ink-faint">
          <tr>
            <th className="py-1 text-left font-normal">regime → signal</th>
            <th className="py-1 text-right font-normal">days in · hold</th>
            <th className="py-1 text-right font-normal">sharpe in-sample</th>
            <th className="py-1 text-right font-normal">sharpe holdout</th>
            <th className="py-1 text-right font-normal">return in · hold</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((l) => {
            const s = regime.by_label[l] ?? {}
            return (
              <tr key={l} className="border-t border-seam/60 text-ink-dim">
                <td className="py-1">
                  <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: color(l) }} />
                  <span className="text-ink">{l}</span>
                  {regime.routes[l] && <span> → {regime.routes[l]}</span>}
                </td>
                <td className="py-1 text-right">
                  {s.in_sample?.days ?? 0} · {s.holdout?.days ?? 0}
                </td>
                <td className="py-1 text-right">{fmtMetric('sharpe', s.in_sample?.sharpe)}</td>
                <td className="py-1 text-right">{fmtMetric('sharpe', s.holdout?.sharpe)}</td>
                <td className="py-1 text-right">
                  {fmtMetric('total_return', s.in_sample?.total_return)} · {fmtMetric('total_return', s.holdout?.total_return)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div className="text-[10.5px] text-ink-faint">A day takes the regime it spent the most bars in.</div>
    </div>
  )
}
