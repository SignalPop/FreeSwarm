'use client'

import { useEffect, useMemo, useState } from 'react'
import { FORECAST_COLOUR, joinBand, joinPath, REALIZED_COLOUR } from '@/components/ForecastValuesPanel'
import ValuesChart, { type ChartBand, type ChartLine, type ChartMarker } from '@/components/ValuesChart'
import { insight, type FeatureView } from '@/lib/insight'

/**
 * One stored forecast feature, drawn at a few in-sample anchors (backend: app/forecast_view.py):
 *  1. every input it read (target + covariates) over the anchor's context, each z-scored over
 *     that window so they share one axis -- hover reads out the RAW values;
 *  2. the forecast cone in the target's own units: the median path and 10-90% band past the
 *     as-of (re-run from the same inputs when the model is loaded; otherwise the stored value at
 *     the horizon), against the actual values that followed -- in-sample only, cut at the split.
 * Plus how often the actual fell inside the band across the drawn anchors.
 */

// Fixed order by the input's position in the recipe (targets first): colour follows the input.
const seriesColour = (k: number) => `var(--color-series-${Math.min(8, k + 1)})`

export default function FeatureForecastView({ objectiveId, view }: { objectiveId: string; view: string }) {
  const [doc, setDoc] = useState<FeatureView | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [k, setK] = useState<number | null>(null)

  useEffect(() => {
    let live = true
    setDoc(null)
    setErr(null)
    insight
      .forecastView(objectiveId, view, 5)
      .then((d) => live && setDoc(d))
      .catch((x: unknown) => live && setErr(x instanceof Error ? x.message : String(x)))
    return () => {
      live = false
    }
  }, [objectiveId, view])

  const anchor = doc?.anchors.length ? doc.anchors[k ?? doc.anchors.length - 1] ?? null : null

  const inputsChart = useMemo(() => {
    if (!anchor) return null
    const lines: ChartLine[] = anchor.inputs.map((s, i) => {
      const mean = s.mean ?? 0
      const std = s.std || 1
      return {
        key: s.name,
        label: s.role === 'target' ? `${s.name} (target)` : s.name,
        t: anchor.t,
        v: s.v.map((x) => (x === null ? null : (x - mean) / std)),
        raw: s.v,
        stroke: seriesColour(i),
      }
    })
    return lines
  }, [anchor])

  const cones = useMemo(() => {
    if (!anchor || !doc) return []
    const tailFrom = anchor.as_of_t - Math.max(4 * doc.horizon, 24) * doc.step_seconds
    return anchor.forecasts.map((f) => {
      const i = anchor.inputs.findIndex((s) => s.name === f.target)
      const src = anchor.inputs[i]
      const keep = anchor.t.map((_, j) => j).filter((j) => anchor.t[j] >= tailFrom)
      const ctxT = keep.map((j) => anchor.t[j])
      const ctxV = src ? keep.map((j) => src.v[j]) : []
      const lastT = ctxT[ctxT.length - 1]
      const lastV = ctxV[ctxV.length - 1]
      const lines: ChartLine[] = []
      const bands: ChartBand[] = []
      if (src) lines.push({ key: 'ctx', label: `${f.target} (sent)`, t: ctxT, v: ctxV, stroke: seriesColour(i) })
      if (f.actual.t.length)
        lines.push({ key: 'actual', label: 'actual (in-sample)', t: f.actual.t, v: f.actual.v, stroke: REALIZED_COLOUR, width: 1.5 })
      if (f.path) {
        lines.push({ key: 'median', label: 'forecast (median)', ...joinPath(lastT, lastV, f.path.t, f.path.median), stroke: FORECAST_COLOUR })
        bands.push({ key: 'band', label: '10–90% band', ...joinBand(lastT, lastV, f.path.t, f.path.q10, f.path.q90), fill: FORECAST_COLOUR })
      }
      const markers: ChartMarker[] = [
        { key: 'end', label: 'stored forecast at the horizon (10–90%)', t: f.end.t, v: f.end.median, lo: f.end.q10, hi: f.end.q90, stroke: FORECAST_COLOUR },
      ]
      return { f, lines, bands, markers }
    })
  }, [anchor, doc])

  if (err) return <div className="text-[10.5px] text-bad">✗ could not draw the forecast: {err}</div>
  if (!doc) return <div className="text-[10.5px] text-ink-faint">drawing the inputs and the forecast…</div>
  if (!anchor) return <div className="text-[10.5px] text-ink-faint">{doc.note ?? 'nothing to draw'}</div>

  const cov = doc.coverage
  return (
    <div className="space-y-2 rounded-md border border-seam/70 bg-panel/60 p-2 font-sans">
      <div className="flex flex-wrap items-center gap-2">
        <div role="tablist" aria-label="anchor" className="flex flex-wrap gap-1">
          {doc.anchors.map((a, i) => {
            const cur = a === anchor
            return (
              <button
                key={a.as_of_t}
                role="tab"
                aria-selected={cur}
                onClick={() => setK(i)}
                className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${cur ? 'bg-accent/20 text-ink' : 'text-ink-faint hover:text-ink-dim'}`}
                title={a.near_split ? 'its horizon reaches the split: the actual values stop there' : undefined}
              >
                as of {a.as_of.replace('T', ' ')}
                {a.near_split ? ' · cut at split' : ''}
              </button>
            )
          })}
        </div>
        <span className="ml-auto font-mono text-[10px] text-ink-faint">
          {doc.model} · h{doc.horizon} · ctx {doc.context} · {anchor.sent.toLocaleString()} rows read
        </span>
      </div>
      <div className="text-[10.5px] text-ink-dim">
        actual inside the 10–90% band:{' '}
        {cov.endpoint.n ? (
          <span className="text-ink">
            at the horizon {cov.endpoint.inside}/{cov.endpoint.n} anchors
          </span>
        ) : (
          'at the horizon — (no anchor with its full horizon in-sample)'
        )}
        {cov.path.n > 0 && (
          <>
            {' · '}along the paths {cov.path.inside}/{cov.path.n} points ({Math.round((cov.path.share ?? 0) * 100)}%)
          </>
        )}
        <span className="text-ink-faint"> · 80% is calibrated · {doc.anchors.length} anchors drawn</span>
      </div>
      {doc.note && <div className="text-[10px] text-warn">{doc.note}</div>}

      <div>
        <div className="font-mono text-[10px] uppercase tracking-wide text-ink-faint">
          inputs over the context · each z-scored over this window (hover: raw values)
        </div>
        {inputsChart && (
          <ValuesChart
            lines={inputsChart}
            asOf={anchor.as_of_t}
            height={190}
            yFormat={(v) => `${v.toFixed(1)}σ`}
            ariaLabel={`inputs to ${view}, z-scored over the context window ending ${anchor.as_of}`}
            hint="hover for each input's raw value · UTC"
          />
        )}
        {(doc.dropped_series > 0 || doc.calendar) && (
          <div className="text-[10px] text-ink-faint">
            {doc.dropped_series > 0 ? `${doc.dropped_series} more input(s) not drawn (8 colours at most). ` : ''}
            {doc.calendar ? 'Calendar inputs (time of day, weekday) are not drawn.' : ''}
          </div>
        )}
      </div>

      {cones.map(({ f, lines, bands, markers }) => (
        <div key={f.target}>
          <div className="font-mono text-[10px] uppercase tracking-wide text-ink-faint">
            forecast cone · {f.target} in its own units · {doc.path_source}
          </div>
          <ValuesChart
            lines={lines}
            bands={bands}
            markers={markers}
            asOf={anchor.as_of_t}
            shadeAfterAsOf
            ariaLabel={`forecast of ${f.target} from ${anchor.as_of} against the actual values`}
          />
          {f.end.inside !== undefined && (
            <div className={`text-[10px] ${f.end.inside ? 'text-ink-dim' : 'text-warn'}`}>
              at the horizon the actual was {f.end.inside ? 'inside' : 'outside'} the stored 10–90% band
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
