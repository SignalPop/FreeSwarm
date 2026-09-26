'use client'

import { useEffect, useMemo, useState } from 'react'
import { fmtMetric, objectives, type CandidateFull, type EnsembleInfo } from '@/lib/objectives'
import CopyButton from '@/components/CopyButton'

// The same frame as Charts.tsx, so the ensemble's plots line up with the equity curve above.
const W = 640
const PAD = { l: 44, r: 10, t: 10, b: 20 }
const H_ALL = 200
const H_MEMBER = 72
const H_WEIGHT = 22

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

/** Members keep their slot in the fixed series order (by seq, as stored), never their rank. */
const seriesColor = (i: number) => (i >= 0 && i < 8 ? `var(--color-series-${i + 1})` : 'var(--color-ink-faint)')

function linePath(vals: number[], x: (i: number) => number, y: (v: number) => number): string {
  return vals.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
}

/**
 * How an ensemble earns its return: the combined curve over its members' own curves, then each
 * member on its own row with the weight it carried each day -- so "which strategy made the
 * money, and how much of the book did it have then?" is answered by eye. One crosshair spans
 * every plot. Click a member to read its code.
 */
export default function EnsembleView({
  objectiveId,
  ens,
  returns,
  split,
  onOpenCandidate,
}: {
  objectiveId: string
  ens: EnsembleInfo
  /** The ensemble's own (combined) daily returns. */
  returns: [string, number][]
  split: string | null
  onOpenCandidate?: (id: string) => void
}) {
  const [hover, setHover] = useState<number | null>(null)
  const [open, setOpen] = useState<string | null>(null)

  const model = useMemo(() => {
    const dates = ens.weights.length ? ens.weights.map(([d]) => d) : returns.map(([d]) => d)
    const n = Math.max(1, dates.length - 1)
    const x = (i: number) => PAD.l + (i / n) * (W - PAD.l - PAD.r)
    const cumulate = (rs: number[]) => {
      let eq = 1
      return rs.map((r) => (eq *= 1 + r))
    }
    const combinedByDate = new Map(returns)
    const combined = cumulate(dates.map((d) => combinedByDate.get(d) ?? 0))
    // A member with no return on a date counts as 0 there -- the same rule as the backend.
    const members = ens.members.map((m, k) => {
      const byDate = new Map(ens.member_returns[String(m.seq)] ?? [])
      const eq = cumulate(dates.map((d) => byDate.get(d) ?? 0))
      const w = ens.weights.map(([, ws]) => ws[k] ?? 0)
      const [lo, hi] = niceRange(eq.concat([1]))
      const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H_MEMBER - PAD.t - 4)
      return { ...m, k, eq, w, lo, hi, y, stats: ens.member_stats?.find((s) => s.seq === m.seq) }
    })
    const [lo, hi] = niceRange(combined.concat(members.flatMap((m) => m.eq), [1]))
    const y = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H_ALL - PAD.t - PAD.b)
    const wMax = Math.max(0.05, ...members.flatMap((m) => m.w))
    const splitIdx = split ? dates.findIndex((d) => d >= split) : -1
    return { dates, n, x, y, lo, hi, combined, members, wMax, splitIdx }
  }, [ens, returns, split])

  const { dates, n, x, y, lo, hi, combined, members, wMax, splitIdx } = model
  if (dates.length < 2) return null

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    const vx = ((e.clientX - box.left) / box.width) * W
    const i = Math.round(((vx - PAD.l) / (W - PAD.l - PAD.r)) * n)
    setHover(i >= 0 && i < dates.length ? i : null)
  }
  const hoverProps = { onMouseMove: onMove, onMouseLeave: () => setHover(null) }

  const splitShade = (height: number, bottom: number, caption: boolean) =>
    splitIdx > 0 && (
      <>
        <rect
          x={x(splitIdx)}
          y={PAD.t}
          width={W - PAD.r - x(splitIdx)}
          height={height - PAD.t - bottom}
          className="fill-accent/[0.05]"
        />
        <line x1={x(splitIdx)} x2={x(splitIdx)} y1={PAD.t} y2={height - bottom} className="stroke-accent" strokeDasharray="3 3" />
        {caption && (
          <text x={x(splitIdx) + 4} y={PAD.t + 10} className="fill-accent font-mono text-[9px]">
            holdout (hidden from agents)
          </text>
        )}
      </>
    )
  const crosshair = (height: number, bottom: number) =>
    hover !== null && <line x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={height - bottom} className="stroke-ink-dim" strokeWidth={1} />

  const avg = ens.avg_weights
  const weighting =
    ens.weighting === 'inverse_vol'
      ? `inverse volatility, ${ens.lookback_days}-day lookback (weights from returns before each day; equal until ${Math.max(5, Math.floor(ens.lookback_days / 2))} days of history)`
      : 'equal weights (1/n each)'

  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-faint">
          Ensemble · {ens.members.length} strategies · {weighting}
        </div>
        <div className="text-[11.5px] leading-relaxed text-ink-faint">{ens.members_note}</div>
      </div>

      {/* Legend + hover readout: identity is never colour alone. */}
      <div className="flex min-h-[18px] flex-wrap items-center gap-x-3 gap-y-0.5 font-mono text-[10.5px] text-ink-dim">
        {hover !== null ? (
          <>
            <span>{dates[hover]}</span>
            <span className="text-ink">combined {combined[hover].toFixed(3)}</span>
            {members.map((m) => (
              <span key={m.id} className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: seriesColor(m.k) }} />#{m.seq}{' '}
                {m.eq[hover].toFixed(3)} · w {m.w[hover]?.toFixed(2)}
              </span>
            ))}
          </>
        ) : (
          <>
            <span className="flex items-center gap-1 text-ink">
              <span className="inline-block h-[2px] w-3 bg-ink" /> combined
            </span>
            {members.map((m) => (
              <span key={m.id} className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: seriesColor(m.k) }} />#{m.seq}
              </span>
            ))}
            <span className="text-ink-faint">· hover for each day's equity and weights</span>
          </>
        )}
      </div>
      <svg viewBox={`0 0 ${W} ${H_ALL}`} className="w-full" role="img" aria-label="combined equity over the members' equity" {...hoverProps}>
        {splitShade(H_ALL, PAD.b, true)}
        <line x1={PAD.l} x2={W - PAD.r} y1={y(1)} y2={y(1)} className="stroke-seam" />
        {[lo, hi].map((t, i) => (
          <text key={i} x={PAD.l - 6} y={y(t) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
            {t.toFixed(2)}
          </text>
        ))}
        {members.map((m) => (
          <path key={m.id} d={linePath(m.eq, x, y)} fill="none" stroke={seriesColor(m.k)} strokeWidth={1} opacity={0.6} />
        ))}
        <path d={linePath(combined, x, y)} fill="none" className="stroke-ink" strokeWidth={2} strokeLinejoin="round" />
        {crosshair(H_ALL, PAD.b)}
        {hover !== null && (
          <circle cx={x(hover)} cy={y(combined[hover])} r={4} className="fill-ink stroke-panel" strokeWidth={2} />
        )}
        <text x={PAD.l} y={H_ALL - 4} className="fill-ink-faint font-mono text-[9px]">
          {dates[0]}
        </text>
        <text x={W - PAD.r} y={H_ALL - 4} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
          {dates[dates.length - 1]}
        </text>
      </svg>

      <div className="text-[11px] uppercase tracking-wide text-ink-faint">
        Members — equity and the weight each carried (click one for its code)
      </div>
      <div className="space-y-1.5">
        {members.map((m) => {
          const expanded = open === m.id
          const toggle = () => setOpen(expanded ? null : m.id)
          const wy = (v: number) => H_WEIGHT - 2 - (v / wMax) * (H_WEIGHT - 4)
          const area =
            m.w.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${wy(v).toFixed(1)}`).join(' ') +
            ` L${x(m.w.length - 1).toFixed(1)},${wy(0)} L${x(0).toFixed(1)},${wy(0)} Z`
          return (
            <div key={m.id} className={`rounded-lg border ${expanded ? 'border-accent/50' : 'border-seam'}`}>
              <div
                role="button"
                tabIndex={0}
                aria-expanded={expanded}
                aria-label={`member #${m.seq}: show its code`}
                onClick={toggle}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    toggle()
                  }
                }}
                className="cursor-pointer rounded-lg px-2 pb-1 pt-1.5 outline-none hover:bg-panel-hi/60 focus-visible:ring-2 focus-visible:ring-accent"
              >
                <div className="flex flex-wrap items-center gap-x-3 font-mono text-[11px]">
                  <span className="flex items-center gap-1.5 text-ink">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ background: seriesColor(m.k) }} />#{m.seq} ·{' '}
                    {m.model?.split('/').pop() ?? '?'} · avg weight {(avg[m.k] ?? 0).toFixed(2)}
                  </span>
                  <span className="ml-auto text-ink-dim">
                    Sharpe in-sample {fmtMetric('sharpe', m.stats?.in_sample_sharpe)} · holdout{' '}
                    {fmtMetric('sharpe', m.stats?.holdout_sharpe)}
                  </span>
                  <span className="text-ink-faint">{expanded ? '▾ code' : '▸ code'}</span>
                </div>
                {m.rationale && <div className="truncate text-[11px] text-ink-faint" title={m.rationale}>{m.rationale}</div>}
                <svg viewBox={`0 0 ${W} ${H_MEMBER}`} className="w-full" role="img" aria-label={`equity of member #${m.seq}`} {...hoverProps}>
                  {splitShade(H_MEMBER, 4, false)}
                  <line x1={PAD.l} x2={W - PAD.r} y1={m.y(1)} y2={m.y(1)} className="stroke-seam" />
                  {[m.lo, m.hi].map((t, i) => (
                    <text key={i} x={PAD.l - 6} y={m.y(t) + 3} textAnchor="end" className="fill-ink-faint font-mono text-[9px]">
                      {t.toFixed(2)}
                    </text>
                  ))}
                  <path d={linePath(m.eq, x, m.y)} fill="none" stroke={seriesColor(m.k)} strokeWidth={1.5} />
                  {crosshair(H_MEMBER, 4)}
                  {hover !== null && (
                    <circle cx={x(hover)} cy={m.y(m.eq[hover])} r={3.5} fill={seriesColor(m.k)} className="stroke-panel" strokeWidth={2} />
                  )}
                </svg>
                <svg viewBox={`0 0 ${W} ${H_WEIGHT}`} className="w-full" role="img" aria-label={`weight of member #${m.seq} over time`} {...hoverProps}>
                  <text x={PAD.l - 6} y={12} textAnchor="end" className="fill-ink-faint font-mono text-[8.5px]">
                    w
                  </text>
                  <path d={area} fill={seriesColor(m.k)} opacity={0.3} />
                  <path
                    d={m.w.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${wy(v).toFixed(1)}`).join(' ')}
                    fill="none"
                    stroke={seriesColor(m.k)}
                    strokeWidth={1}
                  />
                  {hover !== null && (
                    <line x1={x(hover)} x2={x(hover)} y1={0} y2={H_WEIGHT} className="stroke-ink-dim" strokeWidth={1} />
                  )}
                </svg>
              </div>
              {expanded && <MemberCode objectiveId={objectiveId} id={m.id} seq={m.seq} onOpenCandidate={onOpenCandidate} />}
            </div>
          )
        })}
        <div className="font-mono text-[10px] text-ink-faint">
          weight strips share one scale, 0 to {wMax.toFixed(2)}; the weights of a day sum to 1
        </div>
      </div>

      <CorrelationMatrix seqs={ens.members.map((m) => m.seq)} matrix={ens.correlation_in_sample} />
    </div>
  )
}

/** A member's code, fetched when its row is opened. */
function MemberCode({
  objectiveId,
  id,
  seq,
  onOpenCandidate,
}: {
  objectiveId: string
  id: string
  seq: number
  onOpenCandidate?: (id: string) => void
}) {
  const [c, setC] = useState<CandidateFull | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    objectives
      .candidate(objectiveId, id)
      .then((d) => alive && setC(d))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [objectiveId, id])

  return (
    <div className="border-t border-seam px-2 py-2">
      <div className="mb-1 flex items-center gap-3 font-mono text-[11px]">
        <span className="text-ink-dim">#{seq} code</span>
        {c?.code && <CopyButton text={c.code} label="copy code" />}
        {onOpenCandidate && (
          <button onClick={() => onOpenCandidate(id)} className="text-accent hover:underline">
            open candidate →
          </button>
        )}
      </div>
      {err ? (
        <div className="text-[12px] text-bad">Could not load #{seq}: {err}</div>
      ) : !c ? (
        <div className="text-[12px] text-ink-faint">Loading…</div>
      ) : (
        <pre className="max-h-[360px] overflow-auto whitespace-pre rounded-md bg-panel-hi p-2.5 font-mono text-[11px] leading-relaxed text-ink-dim">
          {c.code || c.answer || '(no code)'}
        </pre>
      )}
    </div>
  )
}

/** In-sample daily-return correlations, each cell tinted by |rho| (darker = more alike). */
export function CorrelationMatrix({ seqs, matrix }: { seqs: number[]; matrix: (number | null)[][] }) {
  if (!matrix.length) return null
  return (
    <div>
      <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-faint">
        Correlation of daily returns · in-sample only · |ρ| &lt; 0.3 ≈ uncorrelated
      </div>
      <table className="font-mono text-[11px]">
        <thead>
          <tr className="text-ink-faint">
            <th className="px-2 py-1" />
            {seqs.map((s, j) => (
              <th key={s} className="px-2 py-1 text-right font-normal">
                <span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: seriesColor(j) }} />#{s}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {seqs.map((s, i) => (
            <tr key={s}>
              <td className="px-2 py-1 text-ink-faint">
                <span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: seriesColor(i) }} />#{s}
              </td>
              {seqs.map((t, j) => {
                const v = matrix[i]?.[j]
                const a = v === null || v === undefined ? 0 : Math.min(1, Math.abs(v))
                return (
                  <td
                    key={t}
                    className={`border border-panel px-2 py-1 text-right ${i === j ? 'text-ink-faint' : 'text-ink'}`}
                    style={i === j ? undefined : { background: `color-mix(in oklab, var(--color-warn) ${Math.round(a * 55)}%, transparent)` }}
                    title={v === null || v === undefined ? 'undefined (no variance)' : `ρ = ${v.toFixed(3)}`}
                  >
                    {v === null || v === undefined ? '—' : v.toFixed(2)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
