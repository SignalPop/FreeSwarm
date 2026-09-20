'use client'

import { useState } from 'react'
import { api, type ConsoleDoc, type ModelEntry, type TsForecast, type TsInstance } from '@/lib/api'
import { bytesLabel } from '@/lib/format'
import { usePoll } from '@/lib/usePoll'
import { Button, Panel, Pill } from '@/components/ui'

/**
 * The time-series category: forecasting models, placed on whichever GPU has room.
 *
 * These are not LLMs and never went through the LLM launch panel -- they were listed as red
 * "not supported" cards, which was accurate about the engine and misleading about the app.
 * They are small (hundreds of MiB), so the placement rule is the opposite of the LLMs': share
 * a card with a running model when there is headroom, never crowd it. The server enforces
 * that; the GPU picker here only shows what it will be judged against.
 */
export default function TimeSeriesModels({
  models,
  console_,
}: {
  models: ModelEntry[]
  console_: ConsoleDoc | null | undefined
}) {
  const { data, refresh } = usePoll<{ instances: TsInstance[] }>(api.tsInstances, 3000)
  const instances = data?.instances ?? []
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<Record<string, string>>({})
  const [gpuFor, setGpuFor] = useState<Record<string, string>>({})
  const [open, setOpen] = useState<string | null>(null)

  const gpus = console_?.gpus ?? []
  if (models.length === 0) return null

  async function load(m: ModelEntry) {
    setBusy(m.id)
    setErr((e) => ({ ...e, [m.id]: '' }))
    try {
      await api.tsStart(m.id, gpuFor[m.id] || undefined)
      refresh()
    } catch (e) {
      setErr((x) => ({ ...x, [m.id]: e instanceof Error ? e.message : String(e) }))
    } finally {
      setBusy(null)
    }
  }

  async function unload(inst: TsInstance) {
    setBusy(inst.model_id)
    try {
      await api.tsStop(inst.id)
      refresh()
    } finally {
      setBusy(null)
    }
  }

  return (
    <section className="mt-8">
      <div className="mb-3 flex items-baseline gap-3">
        <h2 className="text-[16px] font-medium text-ink">Time series</h2>
        <span className="font-mono text-[11px] text-ink-faint">
          forecasting models · share a GPU with a running LLM when there is room
        </span>
      </div>

      <div className="space-y-3">
        {models.map((m) => {
          const inst = instances.find((i) => i.model_id === m.id && i.state !== 'stopped')
          const running = inst?.state === 'running'
          return (
            <Panel key={m.id} className="p-4">
              <div className="flex flex-wrap items-center gap-3">
                <div className="min-w-0 flex-1">
                  <div className="text-[15px] text-ink">{m.id}</div>
                  <div className="mt-0.5 font-mono text-[11px] text-ink-faint">
                    {m.architecture} · {bytesLabel(m.size_bytes)}
                  </div>
                </div>
                {running && (
                  <Pill tone="good" pulse>
                    GPU {inst!.gpu} · {bytesLabel(inst!.health?.vram_bytes ?? 0)}
                  </Pill>
                )}
                {inst?.state === 'starting' && <Pill tone="warn" pulse>loading</Pill>}
                {!m.ts_servable && <Pill tone="neutral">no adapter yet</Pill>}

                {m.ts_servable && !inst && (
                  <>
                    <select
                      value={gpuFor[m.id] ?? ''}
                      onChange={(e) => setGpuFor((x) => ({ ...x, [m.id]: e.target.value }))}
                      className="rounded-lg border border-seam bg-panel-hi px-2 py-1.5 font-mono text-[12px] text-ink-dim outline-none"
                    >
                      <option value="">auto — most room</option>
                      {gpus.map((g) => (
                        <option key={g.index} value={String(g.index)}>
                          GPU {g.index} — {bytesLabel(g.memory_total_bytes - g.memory_used_bytes)} free
                        </option>
                      ))}
                    </select>
                    <Button tone="primary" onClick={() => load(m)} disabled={busy === m.id}>
                      {busy === m.id ? 'loading…' : 'Load'}
                    </Button>
                  </>
                )}
                {running && (
                  <>
                    <Button
                      tone="default"
                      onClick={() => setOpen((o) => (o === m.id ? null : m.id))}
                    >
                      {open === m.id ? 'Hide' : 'Try a forecast'}
                    </Button>
                    <Button tone="danger" onClick={() => unload(inst!)} disabled={busy === m.id}>
                      Unload
                    </Button>
                  </>
                )}
              </div>

              {m.ts_note && (
                <div className="mt-2 text-[12px] leading-relaxed text-ink-dim">{m.ts_note}</div>
              )}
              {(err[m.id] || inst?.error) && (
                <div className="mt-2 rounded-lg border border-bad/40 bg-bad/[0.07] px-3 py-2 text-[12px] text-ink-dim">
                  {err[m.id] || inst?.error}
                </div>
              )}
              {running && open === m.id && <TryForecast model={m.id} />}
            </Panel>
          )
        })}
      </div>
    </section>
  )
}

/** A seasonal-plus-trend series: enough structure that a good forecast is visibly good. */
function sampleSeries(): string {
  const out: string[] = []
  for (let i = 0; i < 120; i++) {
    const v = 20 + i * 0.05 + 6 * Math.sin((2 * Math.PI * i) / 24) + 2 * Math.sin((2 * Math.PI * i) / 7)
    out.push(v.toFixed(2))
  }
  return out.join(', ')
}

function TryForecast({ model }: { model: string }) {
  const [text, setText] = useState(sampleSeries)
  const [horizon, setHorizon] = useState(36)
  const [res, setRes] = useState<TsForecast | null>(null)
  const [hist, setHist] = useState<number[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function run() {
    const series = text
      .split(/[\s,;]+/)
      .map(Number)
      .filter((v) => Number.isFinite(v))
    if (series.length < 4) {
      setErr('Enter at least a few numbers.')
      return
    }
    setBusy(true)
    setErr(null)
    try {
      setHist(series)
      setRes(await api.tsForecast({ model, series, horizon, quantiles: [0.1, 0.5, 0.9] }))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mt-3 space-y-3 border-t border-seam pt-3">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        className="h-20 w-full resize-y rounded-lg border border-seam bg-canvas p-2 font-mono text-[11.5px] text-ink-dim outline-none"
        placeholder="history, comma or space separated"
      />
      <div className="flex items-center gap-3">
        <label className="font-mono text-[11px] text-ink-faint">
          horizon{' '}
          <input
            type="number"
            min={1}
            max={512}
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            className="ml-1 w-20 rounded border border-seam bg-canvas px-2 py-1 text-ink"
          />
        </label>
        <Button tone="primary" onClick={run} disabled={busy}>
          {busy ? 'forecasting…' : 'Forecast'}
        </Button>
        {res && (
          <span className="font-mono text-[11px] text-ink-faint">
            {res.horizon} steps in {res.seconds}s
          </span>
        )}
      </div>
      {err && <div className="font-mono text-[12px] text-bad">{err}</div>}
      {res && <ForecastChart history={hist} forecast={res} />}
      {res?.notes.map((n) => (
        <div key={n} className="text-[12px] text-warn">
          {n}
        </div>
      ))}
    </div>
  )
}

/** History, median forecast and (when the model gives one) the 10-90% band, in one SVG. */
function ForecastChart({ history, forecast }: { history: number[]; forecast: TsForecast }) {
  const f = forecast.forecasts[0]
  const lo = f.quantiles?.['0.1']
  const hi = f.quantiles?.['0.9']
  const all = [...history, ...f.median, ...(lo ?? []), ...(hi ?? [])]
  const min = Math.min(...all)
  const max = Math.max(...all)
  const n = history.length + f.median.length
  const W = 800
  const H = 220
  const x = (i: number) => (i / Math.max(1, n - 1)) * W
  const y = (v: number) => H - ((v - min) / (max - min || 1)) * (H - 16) - 8
  const path = (vals: number[], offset: number) =>
    vals.map((v, i) => `${i ? 'L' : 'M'}${x(i + offset).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const o = history.length - 1
  // Connect the forecast to the last observed point so the two read as one line.
  const median = [history[o], ...f.median]
  const band =
    lo && hi
      ? `${[history[o], ...hi].map((v, i) => `${i ? 'L' : 'M'}${x(i + o).toFixed(1)},${y(v).toFixed(1)}`).join(' ')} ` +
        `${[history[o], ...lo]
          .map((v, i) => [x(i + o), y(v)] as const)
          .reverse()
          .map(([px, py]) => `L${px.toFixed(1)},${py.toFixed(1)}`)
          .join(' ')} Z`
      : null

  return (
    <div className="rounded-lg border border-seam bg-canvas p-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-[220px] w-full" preserveAspectRatio="none">
        <line x1={x(o)} x2={x(o)} y1={0} y2={H} className="stroke-seam" strokeDasharray="3 4" />
        {band && <path d={band} className="fill-accent/15" />}
        <path d={path(history, 0)} fill="none" className="stroke-ink-dim" strokeWidth={1.4} />
        <path d={path(median, o)} fill="none" className="stroke-accent" strokeWidth={2} />
      </svg>
      <div className="mt-1 flex gap-4 font-mono text-[10px] text-ink-faint">
        <span>— history</span>
        <span className="text-accent">— median forecast</span>
        {band && <span className="text-accent/70">▮ 10–90% range</span>}
      </div>
    </div>
  )
}
