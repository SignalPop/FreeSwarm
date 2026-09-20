'use client'

import { useEffect, useRef, useState } from 'react'
import { api, type ConsoleDoc, type UnloadReport } from '@/lib/api'
import { Button, Panel } from '@/components/ui'

/** A card counts as still draining if its usage dropped this much within the settle window. */
const DROP_BYTES = 128 * 2 ** 20
const SETTLE_MS = 6000
const WATCH_MS = 120_000

const gib = (b: number) => (b / 2 ** 30).toFixed(1)

/**
 * Unload every model at once: engines, forecasters, and orphaned engine workers.
 *
 * A big offloaded model does not free its memory the moment Unload returns. Tens of GiB of
 * pinned memory are released after the process exits. A Console that just went blank during
 * that window looked the same as memory stuck in an orphan. So after the unload this keeps
 * watching each card and reports whether memory is still coming back or has settled.
 *
 * Returns the button (always shown, because orphans can hold memory while the engine list is
 * empty) and the report panel separately, so the page can place each one.
 */
export function useUnloadAll(data: ConsoleDoc | null | undefined, onDone: () => void) {
  const [busy, setBusy] = useState(false)
  const [report, setReport] = useState<UnloadReport | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [watchFrom, setWatchFrom] = useState(0)
  const baseline = useRef<Record<number, number>>({})
  const last = useRef<Record<number, { bytes: number; droppedAt: number }>>({})

  const gpus = data?.gpus ?? []
  const now = Date.now()
  const watching = watchFrom > 0 && now - watchFrom < WATCH_MS

  // Track the last time each card's usage fell, from the page's own 1 s poll.
  useEffect(() => {
    if (!watchFrom) return
    const t = Date.now()
    for (const g of data?.gpus ?? []) {
      const prev = last.current[g.index]
      if (!prev) last.current[g.index] = { bytes: g.memory_used_bytes, droppedAt: t }
      else if (prev.bytes - g.memory_used_bytes > DROP_BYTES) last.current[g.index] = { bytes: g.memory_used_bytes, droppedAt: t }
      else if (g.memory_used_bytes > prev.bytes) prev.bytes = g.memory_used_bytes
    }
  }, [data, watchFrom])

  const settled =
    watchFrom > 0 &&
    now - watchFrom > SETTLE_MS &&
    Object.values(last.current).every((c) => now - c.droppedAt > SETTLE_MS)
  const freed = gpus.reduce((sum, g) => {
    const b = baseline.current[g.index]
    return b === undefined ? sum : sum + Math.max(0, b - g.memory_used_bytes)
  }, 0)

  const loaded = (data?.engines ?? []).filter((e) => e.model_id || e.state !== 'stopped').length

  async function run() {
    const what = loaded
      ? `Unload all ${loaded} model${loaded === 1 ? '' : 's'} and any forecasters? Requests in flight are cut off.`
      : 'No model is listed as loaded. Sweep for orphaned engine processes that are still holding memory?'
    if (!window.confirm(what)) return
    setBusy(true)
    setErr(null)
    setReport(null)
    baseline.current = Object.fromEntries(gpus.map((g) => [g.index, g.memory_used_bytes]))
    last.current = {}
    try {
      setReport(await api.unloadAll())
      setWatchFrom(Date.now())
      onDone()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const button = (
    <Button tone="danger" onClick={run} disabled={busy}>
      {busy ? 'Unloading…' : 'Unload all'}
    </Button>
  )

  const okCount = report?.engines.filter((e) => e.ok).length ?? 0
  const panel =
    report || err ? (
      <Panel className="mb-6 p-4 text-[12.5px] text-ink-dim">
        <div className="flex items-start gap-3">
          <div className="min-w-0 flex-1 space-y-1">
            {err && <div className="text-bad">Unload all failed: {err}</div>}
            {report && (
              <>
                <div className="text-ink">
                  Stopped {okCount} engine{okCount === 1 ? '' : 's'}
                  {report.forecasters.length > 0 &&
                    ` and ${report.forecasters.length} forecaster${report.forecasters.length === 1 ? '' : 's'}`}{' '}
                  in {report.seconds}s.
                </div>
                {report.orphans.length > 0 && (
                  <div className="text-warn">
                    Killed {report.orphans.length} orphaned engine process
                    {report.orphans.length === 1 ? '' : 'es'} the Console was not tracking (
                    {report.orphans.map((o) => `pid ${o.pid} on port ${o.ports.join('/')}${o.killed ? '' : ' (kill failed)'}`).join(', ')}
                    ).
                  </div>
                )}
                {report.engines
                  .filter((e) => !e.ok)
                  .map((e) => (
                    <div key={e.instance} className="text-bad">
                      {e.model ?? e.instance}: {e.error}
                    </div>
                  ))}
                {watchFrom > 0 && (
                  <div className={settled ? 'text-good' : 'text-warn'}>
                    {settled ? 'Memory released: ' : watching ? 'Releasing memory… ' : 'Stopped watching: '}
                    {gpus
                      .filter((g) => baseline.current[g.index] !== undefined)
                      .map((g) => `GPU ${g.index} ${gib(baseline.current[g.index])} → ${gib(g.memory_used_bytes)} GiB`)
                      .join(' · ')}
                    {freed > 0 && ` (${gib(freed)} GiB freed)`}
                    {!settled && watching && ' Large offloaded models can take up to a minute to release pinned memory.'}
                  </div>
                )}
              </>
            )}
          </div>
          <button
            onClick={() => {
              setReport(null)
              setErr(null)
              setWatchFrom(0)
            }}
            className="shrink-0 font-mono text-[11px] text-ink-faint hover:text-ink"
          >
            dismiss
          </button>
        </div>
      </Panel>
    ) : null

  return { button, panel }
}
