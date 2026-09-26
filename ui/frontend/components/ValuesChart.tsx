'use client'

import { useMemo, useState } from 'react'

/**
 * A time-series line chart in the house style (objective/Charts.tsx): inline SVG, theme tokens,
 * 2px lines, y-axis min/max labels, a hover crosshair with a readout of the timestamp and every
 * line's value there, an as-of marker, shaded bands and whisker markers, and a text legend
 * (identity is never colour alone) whose items toggle their series on and off -- the y-axis
 * rescales to what is shown. Shared by the agent inspector's forecast values and the
 * objective's Forecasts tab. Times are epoch seconds, shown in UTC.
 */

export type ChartLine = {
  key: string
  label: string
  t: number[]
  v: (number | null)[]
  stroke: string
  dash?: string
  width?: number
  /** Values to READ OUT on hover when `v` is transformed (e.g. z-scored): the raw ones. */
  raw?: (number | null)[]
}
export type ChartBand = { key: string; label: string; t: number[]; lo: (number | null)[]; hi: (number | null)[]; fill: string }
/** A point with an optional lo-hi whisker (e.g. the stored forecast at the horizon). */
export type ChartMarker = { key: string; label: string; t: number; v: number | null; lo?: number | null; hi?: number | null; stroke: string }

const W = 640
const PAD = { l: 56, r: 12, t: 12, b: 22 }

export const isoSec = (s: number) => new Date(s * 1000).toISOString().slice(0, 19).replace('T', ' ')

export function fmtVal(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  const a = Math.abs(v)
  if (a !== 0 && (a >= 1e7 || a < 1e-4)) return v.toExponential(3)
  return String(parseFloat(v.toPrecision(6)))
}

/** Index of the point nearest `t` in a sorted array. */
function nearest(ts: number[], t: number): number {
  let lo = 0
  let hi = ts.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (ts[mid] <= t) lo = mid
    else hi = mid
  }
  return Math.abs(ts[hi] - t) < Math.abs(ts[lo] - t) ? hi : lo
}

const finite = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v)

export default function ValuesChart({
  lines: allLines,
  bands: allBands = [],
  markers: allMarkers = [],
  asOf,
  shadeAfterAsOf = false,
  height = 210,
  ariaLabel,
  yFormat = fmtVal,
  hint = 'hover for the timestamp and values · UTC',
}: {
  lines: ChartLine[]
  bands?: ChartBand[]
  markers?: ChartMarker[]
  asOf?: number | null
  shadeAfterAsOf?: boolean
  height?: number
  ariaLabel: string
  yFormat?: (v: number) => string
  hint?: string
}) {
  const H = height
  const [hover, setHover] = useState<number | null>(null)
  // Keys of the series switched off from the legend (markers share the one key 'markers').
  const [hidden, setHidden] = useState<Set<string>>(() => new Set())
  const toggle = (key: string) =>
    setHidden((h) => {
      const n = new Set(h)
      if (!n.delete(key)) n.add(key)
      return n
    })

  const shown = useMemo(() => {
    const ls = allLines.filter((l) => !hidden.has(l.key))
    const bs = allBands.filter((b) => !hidden.has(b.key))
    const ms = hidden.has('markers') ? [] : allMarkers
    return { lines: ls, bands: bs, markers: ms }
  }, [allLines, allBands, allMarkers, hidden])

  const geo = useMemo(() => {
    // Scale to what is shown; with everything switched off, keep the full frame.
    const any = shown.lines.length + shown.bands.length + shown.markers.length > 0
    const { lines, bands, markers } = any ? shown : { lines: allLines, bands: allBands, markers: allMarkers }
    const ts = [...lines.flatMap((l) => l.t), ...bands.flatMap((b) => b.t), ...markers.map((m) => m.t)]
    if (!ts.length) return null
    const vs = [
      ...lines.flatMap((l) => l.v),
      ...bands.flatMap((b) => [...b.lo, ...b.hi]),
      ...markers.flatMap((m) => [m.v, m.lo, m.hi]),
    ].filter(finite)
    let [lo, hi] = vs.length ? [Math.min(...vs), Math.max(...vs)] : [0, 1]
    if (lo === hi) {
      lo -= Math.abs(lo) * 0.1 || 1
      hi += Math.abs(hi) * 0.1 || 1
    }
    const pad = (hi - lo) * 0.06
    const [ylo, yhi] = [lo - pad, hi + pad]
    let [t0, t1] = [Math.min(...ts), Math.max(...ts)]
    if (t1 - t0 < 1) [t0, t1] = [t0 - 30, t1 + 30]
    const x = (t: number) => PAD.l + ((t - t0) / (t1 - t0)) * (W - PAD.l - PAD.r)
    const y = (v: number) => PAD.t + (1 - (v - ylo) / (yhi - ylo)) * (H - PAD.t - PAD.b)
    const path = (t: number[], v: (number | null)[]) => {
      let d = ''
      let pen = false
      t.forEach((tt, i) => {
        const vv = v[i]
        if (!finite(vv)) {
          pen = false
          return
        }
        d += `${pen ? 'L' : 'M'}${x(tt).toFixed(1)},${y(vv).toFixed(1)}`
        pen = true
      })
      return d
    }
    const area = (b: ChartBand) => {
      const idx = b.t.map((_, i) => i).filter((i) => finite(b.lo[i]) && finite(b.hi[i]))
      if (idx.length < 2) return ''
      return (
        idx.map((i, k) => `${k ? 'L' : 'M'}${x(b.t[i]).toFixed(1)},${y(b.hi[i] as number).toFixed(1)}`).join('') +
        [...idx].reverse().map((i) => `L${x(b.t[i]).toFixed(1)},${y(b.lo[i] as number).toFixed(1)}`).join('') +
        'Z'
      )
    }
    const span = t1 - t0
    const tick = (s: number) => {
      const d = isoSec(s)
      return span > 60 * 86400 ? d.slice(0, 10) : span > 36 * 3600 ? d.slice(5, 16) : d.slice(11, 19)
    }
    const all = [...new Set(ts)].sort((a, b) => a - b)
    return { x, y, ylo, yhi, t0, t1, path, area, tick, all }
  }, [shown, allLines, allBands, allMarkers, H])

  if (!geo) return <div className="text-[10.5px] text-ink-faint">nothing to draw</div>
  const { x, y, ylo, yhi, t0, t1, path, area, tick, all } = geo
  const { lines: visLines, bands: visBands, markers: visMarkers } = shown
  const tHover = hover !== null ? all[hover] : null
  const within = (ts: number[], t: number) => ts.length > 0 && t >= ts[0] && t <= ts[ts.length - 1]
  const reading =
    tHover === null
      ? []
      : visLines.flatMap((l) => {
          if (!within(l.t, tHover)) return []
          const i = nearest(l.t, tHover)
          const v = l.v[i]
          return finite(v) ? [{ l, t: l.t[i], v, shown: l.raw ? l.raw[i] : v }] : []
        })
  const bandReads =
    tHover === null
      ? []
      : visBands.flatMap((b) => {
          if (!within(b.t, tHover)) return []
          const i = nearest(b.t, tHover)
          return [{ b, lo: b.lo[i], hi: b.hi[i] }]
        })
  const markerReads = tHover === null ? [] : visMarkers.filter((m) => m.t === tHover)

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    const vx = ((e.clientX - box.left) / box.width) * W
    if (vx < PAD.l || vx > W - PAD.r) return setHover(null)
    setHover(nearest(all, t0 + ((vx - PAD.l) / (W - PAD.l - PAD.r)) * (t1 - t0)))
  }

  const swatch = (stroke: string, dash?: string) => (
    <svg width="16" height="6" aria-hidden>
      <line x1="0" x2="16" y1="3" y2="3" stroke={stroke} strokeWidth={2} strokeDasharray={dash} />
    </svg>
  )
  const asX = finite(asOf) ? x(asOf) : null
  const legendItem = (key: string, label: string, mark: React.ReactNode) => {
    const off = hidden.has(key)
    return (
      <button
        key={key}
        type="button"
        aria-pressed={!off}
        title={off ? 'show' : 'hide'}
        onClick={() => toggle(key)}
        className={`flex items-center gap-1 rounded hover:text-ink ${off ? 'text-ink-faint line-through opacity-50' : ''}`}
      >
        {mark}
        {label}
      </button>
    )
  }

  return (
    <div className="space-y-1">
      <div className="flex min-h-[16px] flex-wrap items-center gap-x-3 font-mono text-[10px] text-ink-dim">
        {tHover !== null ? (
          <>
            <span className="text-ink">{isoSec(tHover)}</span>
            {reading.map((r) => (
              <span key={r.l.key} className="flex items-center gap-1">
                {swatch(r.l.stroke, r.l.dash)}
                {r.l.label} {fmtVal(r.shown)}
              </span>
            ))}
            {bandReads.map((r) => (
              <span key={r.b.key}>
                {r.b.label} {fmtVal(r.lo)} – {fmtVal(r.hi)}
              </span>
            ))}
            {markerReads.map((m) => (
              <span key={m.key}>
                {m.label} {fmtVal(m.v)}
                {finite(m.lo) && finite(m.hi) ? ` (${fmtVal(m.lo)} – ${fmtVal(m.hi)})` : ''}
              </span>
            ))}
          </>
        ) : (
          <span className="text-ink-faint">{hint}</span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={ariaLabel}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {[ylo, yhi].map((v, i) => (
          <g key={i}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="stroke-seam" strokeWidth={1} />
            <text x={PAD.l - 6} y={y(v) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
              {yFormat(v)}
            </text>
          </g>
        ))}
        {shadeAfterAsOf && asX !== null && (
          <rect x={asX} y={PAD.t} width={Math.max(0, W - PAD.r - asX)} height={H - PAD.t - PAD.b} className="fill-accent" opacity={0.05} />
        )}
        {visBands.map((b) => (
          <g key={b.key}>
            <path d={area(b)} style={{ fill: b.fill }} opacity={0.3} />
            <path d={path(b.t, b.lo)} fill="none" stroke={b.fill} strokeWidth={1} opacity={0.6} />
            <path d={path(b.t, b.hi)} fill="none" stroke={b.fill} strokeWidth={1} opacity={0.6} />
          </g>
        ))}
        {visLines.map((l) => (
          <path key={l.key} d={path(l.t, l.v)} fill="none" stroke={l.stroke} strokeWidth={l.width ?? 2} strokeDasharray={l.dash} strokeLinejoin="round" />
        ))}
        {visMarkers.map((m) =>
          finite(m.v) ? (
            <g key={m.key}>
              {finite(m.lo) && finite(m.hi) && (
                <line x1={x(m.t)} x2={x(m.t)} y1={y(m.lo)} y2={y(m.hi)} stroke={m.stroke} strokeWidth={2} />
              )}
              <circle cx={x(m.t)} cy={y(m.v)} r={4} fill={m.stroke} className="stroke-panel" strokeWidth={2} />
            </g>
          ) : null,
        )}
        {asX !== null && (
          <>
            <line x1={asX} x2={asX} y1={PAD.t} y2={H - PAD.b} className="stroke-ink" strokeWidth={1.5} />
            <text x={asX + 4} y={PAD.t + 9} className="fill-ink-dim font-mono text-[9px]">
              as of
            </text>
          </>
        )}
        {tHover !== null && (
          <>
            <line x1={x(tHover)} x2={x(tHover)} y1={PAD.t} y2={H - PAD.b} className="stroke-ink-dim" strokeWidth={1} />
            {reading.map((r) => (
              <circle key={r.l.key} cx={x(r.t)} cy={y(r.v)} r={4} fill={r.l.stroke} className="stroke-panel" strokeWidth={2} />
            ))}
          </>
        )}
        <text x={PAD.l} y={H - 6} className="fill-ink-faint font-mono text-[9px]">
          {tick(t0)}
        </text>
        <text x={W - PAD.r} y={H - 6} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
          {tick(t1)}
        </text>
      </svg>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-ink-dim">
        {allLines.map((l) => legendItem(l.key, l.label, swatch(l.stroke, l.dash)))}
        {allBands.map((b) =>
          legendItem(b.key, b.label, <span className="inline-block h-2 w-3 rounded-sm" style={{ background: b.fill, opacity: 0.35 }} />),
        )}
        {allMarkers.length > 0 &&
          legendItem(
            'markers',
            allMarkers[0].label,
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: allMarkers[0].stroke }} />,
          )}
        {asX !== null && <span className="text-ink-faint">| as-of · UTC</span>}
      </div>
    </div>
  )
}
