'use client'

import { useState } from 'react'
import ForecastValuesPanel from '@/components/ForecastValuesPanel'
import type { ForecastValuesSummary } from '@/lib/agents'

/** What a forecast request sent, with dates (backend: objectives.input_streams, attached to the
 *  request's row by agent_activity.note_inputs). The forecaster itself only sees arrays. */
export type ForecastInputs = {
  kind: string
  model: string
  dataset?: string | null
  requested_by?: string | null
  bar?: string
  step_seconds?: number
  as_of?: { first: string | null; last: string | null; anchors: number; every: number | null }
  horizon?: { bars: number; end: string | null; approximate?: boolean }
  context_bars?: number
  streams: { name: string; role: string; dataset?: string | null; from: string | null; to: string | null; points: number; bar?: string }[]
  samples?: { as_of: string | null; from: string | null; to: string | null; context: number; horizon_end: string | null }[]
  also_without_inputs?: boolean
  /** The values at the sample anchors (older rows have none): fetched by id when opened. */
  values?: ForecastValuesSummary | null
}

// Fixed colour per ROLE (never cycled, never by position), from the validated categorical tokens.
const ROLES: { role: string; token: string; label: string }[] = [
  { role: 'target', token: '--color-series-1', label: 'target (what is forecast)' },
  { role: 'past covariate', token: '--color-series-2', label: 'past input' },
  { role: 'candidate input', token: '--color-series-3', label: 'candidate input (lab)' },
  { role: 'calendar (past)', token: '--color-series-4', label: 'calendar, past' },
  { role: 'known ahead', token: '--color-series-7', label: 'known ahead (future input)' },
]
const colourOf = (role: string) => `var(${(ROLES.find((r) => r.role === role) ?? ROLES[0]).token})`

const W = 640
const LABEL_W = 170
const ROW_H = 18

const ms = (s: string | null | undefined) => (s ? Date.parse(s.endsWith('Z') ? s : `${s}Z`) : NaN)
const short = (s: string | null | undefined) => (s ? s.replace('T', ' ') : '—')

/**
 * One request as a timeline on a single time axis: a block per input stream spanning the dates
 * actually sent, a marker at the as-of time the forecasts were made from (a shaded band when a
 * feature build made them at many anchors), and a hatched block for the forecast horizon.
 * Click a block (or Enter on it) for its exact timestamps and counts and a line chart of the values
 * sent at a sample anchor -- for a target, with the forecast returned; click a sample-anchor line to
 * switch anchors (ForecastValuesPanel).
 */
export default function ForecastInputsTimeline({ details }: { details: ForecastInputs[] | undefined | null }) {
  if (!details?.length) return null
  return (
    <div className="mt-2 space-y-3">
      {details.map((d, i) => (
        <Timeline key={i} d={d} />
      ))}
    </div>
  )
}

function Timeline({ d }: { d: ForecastInputs }) {
  const [pick, setPick] = useState<number | null>(null)
  // The sample anchor whose values are shown (null = the latest one captured).
  const [sample, setSample] = useState<number | null>(null)
  const [open, setOpen] = useState(false)
  const streams = d.streams ?? []
  const vals = d.values?.anchors?.length ? d.values : null
  const toggleStream = (i: number) => {
    const off = pick === i
    setPick(off ? null : i)
    setOpen(!off)
  }
  const pickSample = (k: number) => {
    if (open && sample === k) return setOpen(false)
    setSample(k)
    setOpen(true)
  }
  const asOf0 = ms(d.as_of?.first)
  const asOf1 = ms(d.as_of?.last)
  const hEnd = ms(d.horizon?.end)
  const times = [...streams.flatMap((s) => [ms(s.from), ms(s.to)]), asOf0, asOf1, hEnd].filter(Number.isFinite)
  if (!times.length) return null
  let lo = Math.min(...times)
  let hi = Math.max(...times)
  if (hi - lo < 1000) {
    lo -= 30_000
    hi += 30_000
  }
  const x = (t: number) => LABEL_W + ((t - lo) / (hi - lo)) * (W - LABEL_W - 10)
  const rows = streams.length + 1 // + the horizon
  const H = rows * ROW_H + 34
  const axisY = rows * ROW_H + 8
  const ticks = [0, 1 / 3, 2 / 3, 1].map((f) => lo + f * (hi - lo))
  const multiDay = hi - lo > 36 * 3600_000
  const tickLabel = (t: number) => {
    const iso = new Date(t).toISOString()
    return multiDay ? iso.slice(0, 10) : iso.slice(5, 16).replace('T', ' ')
  }
  const roles = ROLES.filter((r) => streams.some((s) => s.role === r.role))
  const sel = pick === null ? null : pick < streams.length ? streams[pick] : null
  const patternId = `fc-hatch-${Math.round(asOf1 || 0)}-${streams.length}`

  return (
    <div className="rounded-md border border-seam/70 bg-panel/60 p-2">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-[10.5px]">
        <span className="text-ink">{d.kind}</span>
        <span className="text-ink-faint">
          as of{' '}
          {d.as_of && d.as_of.anchors > 1
            ? `${short(d.as_of.first)} → ${short(d.as_of.last)} (${d.as_of.anchors.toLocaleString()} forecasts${d.as_of.every ? `, every ${d.as_of.every} bars` : ''})`
            : short(d.as_of?.last)}
        </span>
        <span className="text-ink-faint">
          · {d.context_bars ?? '?'} bars of context each · horizon {d.horizon?.bars ?? '?'} × {d.bar ?? '?'}
          {d.also_without_inputs ? ' · also run without the inputs (for the lift)' : ''}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 w-full" role="img" aria-label="input streams sent to the forecaster">
        <defs>
          <pattern id={patternId} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" className="stroke-ink-faint" strokeWidth="2" />
          </pattern>
        </defs>
        {Number.isFinite(asOf0) && Number.isFinite(asOf1) && asOf1 > asOf0 && (
          <rect x={x(asOf0)} y={0} width={Math.max(1, x(asOf1) - x(asOf0))} height={rows * ROW_H} className="fill-accent" opacity={0.07}>
            <title>forecasts made from {short(d.as_of?.first)} to {short(d.as_of?.last)}</title>
          </rect>
        )}
        {streams.map((s, i) => {
          const a = ms(s.from)
          const b = ms(s.to)
          const y = i * ROW_H
          return (
            <g
              key={`${s.role}|${s.name}|${i}`}
              className="cursor-pointer"
              role="button"
              tabIndex={0}
              aria-pressed={pick === i}
              aria-label={`${s.name}, ${s.role}: ${short(s.from)} to ${short(s.to)}${vals ? ', show its values' : ''}`}
              onClick={() => toggleStream(i)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  toggleStream(i)
                }
              }}
            >
              <text x={LABEL_W - 6} y={y + 12} textAnchor="end" className="fill-ink-dim font-mono text-[9.5px]">
                {s.name.length > 26 ? `${s.name.slice(0, 25)}…` : s.name}
              </text>
              {Number.isFinite(a) && Number.isFinite(b) && (
                <rect
                  x={x(a)}
                  y={y + 3}
                  width={Math.max(3, x(b) - x(a))}
                  height={ROW_H - 6}
                  rx={3}
                  style={{ fill: colourOf(s.role) }}
                  opacity={pick === null || pick === i ? 0.9 : 0.4}
                >
                  <title>{`${s.name} (${s.role}): ${short(s.from)} → ${short(s.to)}, ${s.points.toLocaleString()} points`}</title>
                </rect>
              )}
            </g>
          )
        })}
        {Number.isFinite(asOf1) && Number.isFinite(hEnd) && (
          <g>
            <text x={LABEL_W - 6} y={streams.length * ROW_H + 12} textAnchor="end" className="fill-ink-faint font-mono text-[9.5px]">
              forecast horizon
            </text>
            <rect
              x={x(asOf1)}
              y={streams.length * ROW_H + 3}
              width={Math.max(3, x(hEnd) - x(asOf1))}
              height={ROW_H - 6}
              rx={3}
              fill={`url(#${patternId})`}
              className="stroke-ink-faint"
              strokeWidth={1}
            >
              <title>{`forecast: ${d.horizon?.bars} bars after ${short(d.as_of?.last)} → ≈ ${short(d.horizon?.end)}`}</title>
            </rect>
          </g>
        )}
        {Number.isFinite(asOf1) && (
          <g>
            <line x1={x(asOf1)} x2={x(asOf1)} y1={0} y2={rows * ROW_H} className="stroke-ink" strokeWidth={1.5} />
            <title>as of {short(d.as_of?.last)}</title>
          </g>
        )}
        <line x1={LABEL_W} x2={W - 10} y1={axisY} y2={axisY} className="stroke-seam" />
        {ticks.map((t, i) => (
          <text
            key={i}
            x={x(t)}
            y={axisY + 14}
            textAnchor={i === 0 ? 'start' : i === ticks.length - 1 ? 'end' : 'middle'}
            className="fill-ink-faint font-mono text-[9px]"
          >
            {tickLabel(t)}
          </text>
        ))}
      </svg>
      {/* A text legend beside the colours: some tokens are low-contrast on the light panel. */}
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-ink-dim">
        {roles.map((r) => (
          <span key={r.role} className="flex items-center gap-1">
            <span className="inline-block h-2 w-3 rounded-sm" style={{ background: `var(${r.token})` }} />
            {r.label}
          </span>
        ))}
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-0.5 bg-ink" /> as-of
        </span>
        <span className="text-ink-faint">▨ horizon · UTC</span>
      </div>
      {sel && (
        <div className="mt-1 rounded bg-panel-hi px-2 py-1 text-[10.5px] text-ink-dim">
          <span className="text-ink">{sel.name}</span> · {sel.role} · {sel.dataset ?? d.dataset ?? '—'} · {short(sel.from)} →{' '}
          {short(sel.to)} · {sel.points.toLocaleString()} points at {sel.bar ?? d.bar}
          {!vals && <span className="text-ink-faint"> · no values recorded for this request</span>}
        </div>
      )}
      {open && vals && (
        <ForecastValuesPanel
          id={vals.id}
          streamIndex={sel ? pick : null}
          stream={sel ? { name: sel.name, role: sel.role } : null}
          sample={sample ?? vals.anchors[vals.anchors.length - 1].sample}
          onSample={setSample}
          colour={colourOf}
        />
      )}
      {d.samples && d.samples.length > 0 && (
        <div className="mt-1 space-y-0.5 text-[10px] text-ink-faint">
          {d.samples.map((s, i) => {
            const text = `forecast at ${short(s.as_of)} read ${short(s.from)} → ${short(s.to)} (${s.context} bars), forecasting to ≈ ${short(s.horizon_end)}`
            const cap = vals?.anchors.find((a) => a.sample === i)
            if (!vals) return <div key={i}>{text}</div>
            const current = open && (sample ?? vals.anchors[vals.anchors.length - 1].sample) === i
            return (
              <button
                key={i}
                type="button"
                aria-pressed={current}
                onClick={() => pickSample(i)}
                className={`block text-left hover:text-ink-dim ${current ? 'text-ink-dim underline decoration-dotted' : ''}`}
                title={cap?.note ?? 'show the values sent and the forecast returned'}
              >
                {text}
                {cap ? (cap.note ? ` · values shown at ${short(cap.as_of)} (in-sample)` : ' · values ▸') : ' · values: none (holdout)'}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
