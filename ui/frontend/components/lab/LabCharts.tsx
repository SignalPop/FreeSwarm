'use client'

import type { Band } from '@/lib/tslab'

const W = 640

/**
 * Horizontal bars with +/- 2 SE whiskers, sorted, centred on zero. A bar whose whisker does
 * not cross zero is significant (solid); one that does is shown faded -- visually "might be
 * noise" -- so the eye does not read a lucky bar as a finding.
 */
export function ImpactBars({
  items,
  familyOf,
}: {
  items: { label: string; gain: number | null; se: number | null; significant: boolean }[]
  familyOf?: (label: string) => string
}) {
  const rows = items.filter((i) => i.gain !== null)
  if (!rows.length) return <div className="text-[12px] text-ink-faint">No results.</div>
  const rowH = 20
  const labelW = 190
  const H = rows.length * rowH + 24
  const extent = Math.max(1e-6, ...rows.map((r) => Math.abs(r.gain as number) + 2 * (r.se ?? 0)))
  const x0 = labelW + (W - labelW - 16) / 2
  const scale = (W - labelW - 16) / 2 / extent
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="input impact">
      <line x1={x0} x2={x0} y1={4} y2={H - 18} className="stroke-ink-faint" strokeWidth={1} />
      {rows.map((r, i) => {
        const y = 6 + i * rowH
        const g = r.gain as number
        const se = r.se ?? 0
        const pos = g >= 0
        const tone = pos ? 'fill-good' : 'fill-bad'
        return (
          <g key={r.label} opacity={r.significant ? 1 : 0.45}>
            <text x={labelW - 6} y={y + 12} textAnchor="end" className="fill-ink-dim font-mono text-[10px]">
              {r.label}
              {familyOf ? '' : ''}
            </text>
            <rect x={pos ? x0 : x0 + g * scale} y={y + 3} width={Math.max(1, Math.abs(g) * scale)} height={rowH - 8} className={tone} rx={2}>
              <title>
                {r.label}: {g >= 0 ? '+' : ''}
                {g.toFixed(4)} ± {(2 * se).toFixed(4)} {r.significant ? '(significant)' : '(within noise)'}
              </title>
            </rect>
            <line x1={x0 + (g - 2 * se) * scale} x2={x0 + (g + 2 * se) * scale} y1={y + rowH / 2 - 1} y2={y + rowH / 2 - 1}
              className="stroke-ink" strokeWidth={1} />
            <line x1={x0 + (g - 2 * se) * scale} x2={x0 + (g - 2 * se) * scale} y1={y + 5} y2={y + rowH - 7} className="stroke-ink" />
            <line x1={x0 + (g + 2 * se) * scale} x2={x0 + (g + 2 * se) * scale} y1={y + 5} y2={y + rowH - 7} className="stroke-ink" />
            {r.significant && (
              <text x={W - 4} y={y + 12} textAnchor="end" className="fill-good font-mono text-[9px]">
                ●
              </text>
            )}
          </g>
        )
      })}
      <text x={x0} y={H - 4} textAnchor="middle" className="fill-ink-faint font-mono text-[9px]">
        0
      </text>
      <text x={labelW} y={H - 4} className="fill-bad font-mono text-[9px]">
        hurts
      </text>
      <text x={W - 16} y={H - 4} textAnchor="end" className="fill-good font-mono text-[9px]">
        helps
      </text>
    </svg>
  )
}

/** Skill as inputs are added one by one (greedy forward selection). */
export function GreedyPath({ path }: { path: { added: string | null; skill: number | null; significant?: boolean }[] }) {
  const pts = path.filter((p) => p.skill !== null)
  if (pts.length < 1) return null
  const H = 170
  const pad = { l: 50, r: 16, t: 14, b: 44 }
  const vals = pts.map((p) => p.skill as number)
  let lo = Math.min(0, ...vals)
  let hi = Math.max(0, ...vals)
  if (hi - lo < 1e-6) {
    lo -= 0.01
    hi += 0.01
  }
  const x = (i: number) => pad.l + (pts.length === 1 ? 0.5 : i / (pts.length - 1)) * (W - pad.l - pad.r)
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b)
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i)},${y(p.skill as number)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="best combination path">
      <line x1={pad.l} x2={W - pad.r} y1={y(0)} y2={y(0)} className="stroke-seam" strokeDasharray="3 3" />
      <text x={pad.l - 6} y={y(0) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
        0
      </text>
      <text x={pad.l - 6} y={y(hi) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
        {hi.toFixed(3)}
      </text>
      <text x={pad.l - 6} y={y(lo) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
        {lo.toFixed(3)}
      </text>
      <path d={d} fill="none" className="stroke-accent" strokeWidth={2} />
      {pts.map((p, i) => (
        <g key={i}>
          <circle cx={x(i)} cy={y(p.skill as number)} r={4} className={i === 0 ? 'fill-ink-faint' : p.significant ? 'fill-good' : 'fill-accent'} />
          <text x={x(i)} y={H - pad.b + 16} textAnchor="middle" className="fill-ink-dim font-mono text-[9px]">
            {i === 0 ? 'target only' : `+ ${p.added}`}
          </text>
          <text x={x(i)} y={y(p.skill as number) - 8} textAnchor="middle" className="fill-ink font-mono text-[9px]">
            {(p.skill as number).toFixed(3)}
          </text>
        </g>
      ))}
    </svg>
  )
}

/** History, what actually happened, and both forecast bands (without / with inputs). */
export function FanChart({
  history,
  actual,
  baseline,
  withInputs,
  title,
}: {
  history: number[]
  actual: number[]
  baseline: Band
  withInputs: Band
  title: string
}) {
  const H = 170
  const pad = { l: 46, r: 8, t: 16, b: 16 }
  const n = history.length + actual.length
  const all = [...history, ...actual, ...baseline.q10, ...baseline.q90, ...withInputs.q10, ...withInputs.q90]
  let lo = Math.min(...all)
  let hi = Math.max(...all)
  if (hi - lo < 1e-9) {
    lo -= 1
    hi += 1
  }
  const x = (i: number) => pad.l + (i / Math.max(1, n - 1)) * (W - pad.l - pad.r)
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b)
  const off = history.length
  const line = (vals: number[], start: number) => vals.map((v, i) => `${i ? 'L' : 'M'}${x(start + i)},${y(v)}`).join(' ')
  const band = (b: Band) =>
    `M${b.q90.map((v, i) => `${x(off + i)},${y(v)}`).join(' L')} L${[...b.q10].reverse().map((v, i) => `${x(off + b.q10.length - 1 - i)},${y(v)}`).join(' L')} Z`
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={title}>
      <text x={pad.l} y={11} className="fill-ink-faint font-mono text-[9px]">
        {title}
      </text>
      <text x={pad.l - 4} y={y(hi) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
        {hi.toFixed(2)}
      </text>
      <text x={pad.l - 4} y={y(lo) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
        {lo.toFixed(2)}
      </text>
      <line x1={x(off - 1)} x2={x(off - 1)} y1={pad.t} y2={H - pad.b} className="stroke-seam" strokeDasharray="2 3" />
      <path d={band(baseline)} className="fill-ink-faint/15" />
      <path d={band(withInputs)} className="fill-accent/20" />
      <path d={line(history, 0)} fill="none" className="stroke-ink-dim" strokeWidth={1.2} />
      <path d={line([history[history.length - 1], ...actual], off - 1)} fill="none" className="stroke-ink" strokeWidth={1.6} />
      <path d={line(baseline.median, off)} fill="none" className="stroke-ink-faint" strokeWidth={1.4} strokeDasharray="4 3" />
      <path d={line(withInputs.median, off)} fill="none" className="stroke-accent" strokeWidth={1.8} />
    </svg>
  )
}
