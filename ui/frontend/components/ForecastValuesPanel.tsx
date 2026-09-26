'use client'

import { useEffect, useMemo, useState } from 'react'
import ValuesChart, { fmtVal, isoSec, type ChartBand, type ChartLine } from '@/components/ValuesChart'
import { agentActivity, type ForecastValues } from '@/lib/agents'

/**
 * The values behind one input stream of a forecast request, at one of its sample anchors: the
 * context the model read, and -- for a target -- the forecast it returned (median continuing
 * past the as-of, 10-90% band shaded), the same forecast without the inputs (dashed) when the
 * build ran one, and what actually followed (in-sample only). Values are fetched on demand.
 */

// Fetched once per id; a failed fetch is forgotten so a retry can succeed.
const cache = new Map<string, Promise<ForecastValues>>()
function load(id: string): Promise<ForecastValues> {
  let p = cache.get(id)
  if (!p) {
    p = agentActivity.forecastValues(id)
    p.catch(() => cache.delete(id))
    cache.set(id, p)
  }
  return p
}

// Fixed tokens by the job of the line (roles take 1-4 and 7 on the timeline).
export const FORECAST_COLOUR = 'var(--color-series-5)'
export const WITHOUT_COLOUR = 'var(--color-series-6)'
export const REALIZED_COLOUR = 'var(--color-ink-dim)'
const DASHES = ['', '6 3', '2 3', '8 3 2 3']

/** Prefix a forecast path with the last value sent, so context and forecast read as one line. */
export function joinPath(lastT: number | undefined, lastV: number | null | undefined, t: number[], v: (number | null)[]) {
  return lastT !== undefined && lastV !== null && lastV !== undefined && (t[0] ?? Infinity) > lastT
    ? { t: [lastT, ...t], v: [lastV, ...v] }
    : { t, v }
}

export default function ForecastValuesPanel({
  id,
  streamIndex,
  stream,
  sample,
  onSample,
  colour,
}: {
  id: string
  /** The timeline block picked (index into the request's streams), or null for the first target. */
  streamIndex: number | null
  stream: { name: string; role: string } | null
  /** The sample anchor picked (index into the request's samples). */
  sample: number
  onSample: (sample: number) => void
  colour: (role: string) => string
}) {
  const [doc, setDoc] = useState<ForecastValues | null>(null)
  const [err, setErr] = useState<string | null>(null)
  // 'all': every input z-scored over its context window on one axis, with the target's
  // forecast cone and what followed on the target's scale; 'one': the picked stream, raw.
  const [mode, setMode] = useState<'all' | 'one'>('all')

  useEffect(() => {
    let live = true
    setErr(null)
    load(id)
      .then((d) => live && setDoc(d))
      .catch((e: unknown) => live && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      live = false
    }
  }, [id])

  const anchor = useMemo(() => {
    if (!doc?.anchors.length) return null
    return doc.anchors.find((a) => a.sample === sample) ?? doc.anchors[doc.anchors.length - 1]
  }, [doc, sample])

  const model = useMemo(() => {
    if (!anchor) return null
    const st =
      (streamIndex !== null ? anchor.streams.find((s) => s.stream === streamIndex) : undefined) ??
      (stream ? anchor.streams.find((s) => s.name === stream.name && s.role === stream.role) : undefined) ??
      (streamIndex === null ? anchor.streams.find((s) => s.role === 'target') : undefined)
    if (!st) return null
    const lines: ChartLine[] = st.lines.map((l, i) => ({
      key: `ctx-${i}`,
      label: st.lines.length > 1 ? `${l.label} (sent)` : `${st.name} (sent)`,
      t: st.t,
      v: l.v,
      stroke: colour(st.role),
      dash: DASHES[i % DASHES.length] || undefined,
    }))
    const bands: ChartBand[] = []
    const isTarget = st.role === 'target'
    const fc = isTarget ? anchor.forecasts.find((f) => f.target === st.name) ?? null : null
    const real = isTarget ? anchor.realized.find((r) => r.target === st.name) ?? null : null
    const lastT = st.t[st.t.length - 1]
    const lastV = st.lines[0]?.v[st.lines[0].v.length - 1]
    if (real) lines.push({ key: 'real', label: 'what followed (in-sample)', t: real.t, v: real.v, stroke: REALIZED_COLOUR, width: 1.5 })
    if (fc?.without_inputs) {
      const j = joinPath(lastT, lastV, fc.without_inputs.t, fc.without_inputs.median)
      lines.push({ key: 'without', label: 'forecast without the inputs (median)', ...j, stroke: WITHOUT_COLOUR, dash: '5 3' })
    }
    if (fc) {
      const j = joinPath(lastT, lastV, fc.with_inputs.t, fc.with_inputs.median)
      lines.push({ key: 'fc', label: 'forecast (median)', ...j, stroke: FORECAST_COLOUR })
      bands.push({ key: 'band', label: '10–90% band', t: fc.with_inputs.t, lo: fc.with_inputs.q10, hi: fc.with_inputs.q90, fill: FORECAST_COLOUR })
    }
    return { st, lines, bands, fc, real }
  }, [anchor, streamIndex, stream, colour])

  const combined = useMemo(() => {
    if (!anchor) return null
    const lines: ChartLine[] = []
    const bands: ChartBand[] = []
    let targetZ: ((x: number | null) => number | null) | null = null
    let targetName = ''
    anchor.streams.forEach((st, k) => {
      const v = st.lines[0]?.v ?? []
      const nums = v.filter((x): x is number => x !== null && Number.isFinite(x))
      if (nums.length < 2) return
      const mu = nums.reduce((a, b) => a + b, 0) / nums.length
      const sd = Math.sqrt(nums.reduce((a, b) => a + (b - mu) ** 2, 0) / (nums.length - 1)) || 1
      const z = (x: number | null) => (x === null || !Number.isFinite(x) ? null : (x - mu) / sd)
      const isTarget = st.role === 'target'
      if (isTarget && !targetZ) {
        targetZ = z
        targetName = st.name
      }
      lines.push({
        key: `z-${k}`,
        label: `${st.name} (${st.role})`,
        t: st.t,
        v: v.map(z),
        raw: v,
        // Each input its own colour, fixed by position; the target drawn heavier.
        stroke: `var(--color-series-${(k % 8) + 1})`,
        width: isTarget ? 2 : 1.25,
      })
    })
    const zf = targetZ as ((x: number | null) => number | null) | null
    const tgt = anchor.streams.find((s) => s.name === targetName && s.role === 'target')
    const fc = zf ? anchor.forecasts.find((f) => f.target === targetName) ?? null : null
    const real = zf ? anchor.realized.find((r) => r.target === targetName) ?? null : null
    if (zf && tgt) {
      const lastT = tgt.t[tgt.t.length - 1]
      const lastV = tgt.lines[0]?.v[tgt.lines[0].v.length - 1] ?? null
      if (real) lines.push({ key: 'real', label: 'what followed (in-sample)', t: real.t, v: real.v.map(zf), raw: real.v,
                             stroke: REALIZED_COLOUR, width: 1.5 })
      if (fc) {
        const j = joinPath(lastT, lastV, fc.with_inputs.t, fc.with_inputs.median)
        lines.push({ key: 'fc', label: 'forecast (median)', t: j.t, v: j.v.map(zf), raw: j.v, stroke: FORECAST_COLOUR })
        bands.push({ key: 'band', label: '10–90% band', t: fc.with_inputs.t, lo: fc.with_inputs.q10.map(zf),
                     hi: fc.with_inputs.q90.map(zf), fill: FORECAST_COLOUR })
      }
    }
    return { lines, bands, fc, real, targetName }
  }, [anchor])

  if (err) return <div className="mt-1 rounded bg-panel-hi px-2 py-1 text-[10.5px] text-ink-faint">values: {err}</div>
  if (!doc || !anchor) return <div className="mt-1 px-2 py-1 text-[10.5px] text-ink-faint">loading values…</div>

  const tabs = (
    <div role="tablist" aria-label="sample anchor" className="flex flex-wrap gap-1">
      {doc.anchors.map((a) => (
        <button
          key={a.sample}
          role="tab"
          aria-selected={a === anchor}
          onClick={() => onSample(a.sample)}
          className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${a === anchor ? 'bg-accent/20 text-ink' : 'text-ink-faint hover:text-ink-dim'}`}
        >
          as of {a.as_of.replace('T', ' ')}
        </button>
      ))}
    </div>
  )

  const modeTabs = (
    <div role="tablist" aria-label="chart" className="flex gap-1">
      {(
        [
          ['all', 'all inputs (scaled) + forecast'],
          ['one', stream ? `${stream.name} only (raw)` : 'this stream (raw)'],
        ] as const
      ).map(([k, label]) => (
        <button
          key={k}
          role="tab"
          aria-selected={mode === k}
          onClick={() => setMode(k)}
          className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${mode === k ? 'bg-accent/20 text-ink' : 'text-ink-faint hover:text-ink-dim'}`}
        >
          {label}
        </button>
      ))}
    </div>
  )

  if (mode === 'all' && combined && combined.lines.length) {
    return (
      <div className="mt-1 space-y-1 rounded bg-panel-hi px-2 py-1.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          {tabs}
          {modeTabs}
        </div>
        {anchor.note && <div className="text-[10px] text-warn">{anchor.note}</div>}
        <div className="font-mono text-[10px] text-ink-faint">
          every series z-scored over its own context window so they share one axis (hover reads the raw values);
          the forecast cone and what followed are on {combined.targetName || 'the target'}&apos;s scale
        </div>
        <ValuesChart
          lines={combined.lines}
          bands={combined.bands}
          asOf={anchor.as_of_t}
          shadeAfterAsOf={!!combined.fc}
          ariaLabel={`all inputs scaled, with the forecast of ${combined.targetName}, as of ${anchor.as_of}`}
        />
      </div>
    )
  }

  if (!model) {
    return (
      <div className="mt-1 space-y-1 rounded bg-panel-hi px-2 py-1.5 text-[10.5px]">
        {tabs}
        {modeTabs}
        <div className="text-ink-faint">
          No values were captured for {stream ? `${stream.name} (${stream.role})` : 'this stream'} at this anchor.
        </div>
      </div>
    )
  }
  const { st, lines, bands, fc, real } = model
  return (
    <div className="mt-1 space-y-1 rounded bg-panel-hi px-2 py-1.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        {tabs}
        {modeTabs}
        <span className="font-mono text-[10px] text-ink-faint">
          {st.name} · {st.role} · {st.sent.toLocaleString()} points sent
          {st.t.length < st.sent ? ` (${st.t.length} drawn, extremes kept)` : ''}
        </span>
      </div>
      {anchor.note && <div className="text-[10px] text-warn">{anchor.note}</div>}
      <ValuesChart
        lines={lines}
        bands={bands}
        asOf={anchor.as_of_t}
        shadeAfterAsOf={!!fc}
        ariaLabel={`${st.name}: values sent${fc ? ' and the forecast returned' : ''}, as of ${anchor.as_of}`}
      />
      {fc && (
        <details className="text-[10px] text-ink-dim">
          <summary className="cursor-pointer text-ink-faint">forecast table ({fc.with_inputs.t.length} steps)</summary>
          <table className="mt-1 w-full font-mono">
            <thead className="text-ink-faint">
              <tr>
                <th className="text-left font-normal">step time</th>
                <th className="text-right font-normal">median</th>
                <th className="text-right font-normal">q10</th>
                <th className="text-right font-normal">q90</th>
                {fc.without_inputs && <th className="text-right font-normal">without inputs</th>}
                {real && <th className="text-right font-normal">what followed</th>}
              </tr>
            </thead>
            <tbody>
              {fc.with_inputs.t.map((t, i) => {
                const wi = fc.without_inputs ? fc.without_inputs.t.indexOf(t) : -1
                const ri = real ? real.t.indexOf(t) : -1
                return (
                  <tr key={t} className="border-t border-seam/60">
                    <td>{isoSec(t)}</td>
                    <td className="text-right">{fmtVal(fc.with_inputs.median[i])}</td>
                    <td className="text-right">{fmtVal(fc.with_inputs.q10[i])}</td>
                    <td className="text-right">{fmtVal(fc.with_inputs.q90[i])}</td>
                    {fc.without_inputs && <td className="text-right">{wi >= 0 ? fmtVal(fc.without_inputs.median[wi]) : '—'}</td>}
                    {real && <td className="text-right">{ri >= 0 ? fmtVal(real.v[ri]) : '—'}</td>}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </details>
      )}
    </div>
  )
}
