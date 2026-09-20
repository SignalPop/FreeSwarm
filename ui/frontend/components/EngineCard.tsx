'use client'

import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { api, type EngineDetail, type GpuInfo } from '@/lib/api'
import { duration, gib } from '@/lib/format'
import LoadProgress from '@/components/LoadProgress'
import { Button, Metric, Panel, Pill } from '@/components/ui'

function stateTone(state: string) {
  if (state === 'running') return 'good' as const
  if (state === 'starting' || state === 'stopping') return 'warn' as const
  if (state === 'error') return 'bad' as const
  return 'neutral' as const
}

/**
 * One engine instance.
 *
 * With several engines resident the first question is always "which model is this, and
 * where is it?", so the model name, its GPU and its port lead — a person debugging a slow
 * agent needs to know which card is busy, not just that something is.
 */
export default function EngineCard({
  engine,
  onChanged,
  otherGpus = [],
}: {
  engine: EngineDetail
  onChanged: () => void
  /** GPU indices held by the OTHER running engines, so a move cannot target an occupied card. */
  otherGpus?: string[]
}) {
  const router = useRouter()
  const [busy, setBusy] = useState(false)
  const [moving, setMoving] = useState(false)
  const [gpus, setGpus] = useState<GpuInfo[]>([])
  const [moveError, setMoveError] = useState<string | null>(null)

  const stats = engine.stats
  // The REAL context: min(model max, KV pages x page size). Showing only the model's max
  // (262,144 for Qwen3.6) hid an 8,192-token engine behind a big number, until an agent's
  // prompt overflowed it.
  const kvTokens = stats?.kv ? stats.kv.total_pages * (stats.kv.page_size || 1) : 0
  const ctxWindow =
    stats?.model?.ctx && kvTokens ? Math.min(stats.model.ctx, kvTokens) : stats?.model?.ctx || kvTokens
  const loading = engine.load && engine.load.phase !== 'ready' && engine.state !== 'error'

  // The pool, for the move menu. Only fetched once per card; the list is static.
  useEffect(() => {
    let alive = true
    api
      .system()
      .then((doc) => alive && setGpus(doc.gpus ?? []))
      .catch(() => {
        /* the move menu just stays empty -- Stop still works */
      })
    return () => {
      alive = false
    }
  }, [])

  // Where this model could go: any pooled card that is not its current one and not held
  // by another engine. CUDA fixes a process's device at startup, so moving is necessarily
  // a stop-and-relaunch -- the backend sequences it so VRAM is freed before the new start.
  const targets = gpus
    .map((g) => String(g.index))
    .filter((idx) => idx !== engine.gpus && !otherGpus.includes(idx))

  async function stop() {
    setBusy(true)
    try {
      await fetch(`/api/engines/${engine.instance_id}/stop`, { method: 'POST' })
      onChanged()
    } finally {
      setBusy(false)
    }
  }

  async function move(target: string) {
    setMoving(true)
    setMoveError(null)
    try {
      await api.moveEngine(engine.instance_id, target)
      onChanged()
    } catch (e) {
      setMoveError(e instanceof Error ? e.message : String(e))
    } finally {
      setMoving(false)
    }
  }

  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-center gap-4">
        <div className="grid h-11 w-11 shrink-0 place-items-center rounded-xl border border-seam bg-panel-hi font-mono text-[13px] text-accent">
          {engine.gpus ? `G${engine.gpus}` : 'FT'}
        </div>

        <div className="min-w-0 flex-1">
          <div className="truncate text-[16px] font-medium text-ink">
            {engine.model_id ?? 'no model'}
          </div>
          <div className="mt-1 flex flex-wrap gap-x-3 font-mono text-[11px] text-ink-faint">
            <span>GPU {engine.gpus ?? 'all'}</span>
            <span>:{engine.port}</span>
            {stats?.model?.moe && <span>MoE</span>}
            {ctxWindow ? (
              <span
                className={ctxWindow < 16384 ? 'text-warn' : ''}
                title={
                  ctxWindow < (stats?.model?.ctx ?? 0)
                    ? `The model supports ${stats!.model!.ctx.toLocaleString()} tokens, but this engine was launched with ${ctxWindow.toLocaleString()} of KV cache -- the real limit.`
                    : undefined
                }
              >
                ctx {ctxWindow.toLocaleString()}
                {ctxWindow < 16384 ? ' (small)' : ''}
              </span>
            ) : null}
            {engine.pid ? <span>pid {engine.pid}</span> : null}
          </div>
        </div>

        <Pill tone={stateTone(engine.state)} pulse={engine.state === 'running' || engine.state === 'starting'}>
          {loading
            ? `${engine.load!.label} · ${engine.load!.overall_pct.toFixed(0)}%`
            : engine.state === 'running'
              ? `Running · ${duration(engine.uptime_s)}`
              : engine.state}
        </Pill>

        {engine.state === 'running' && (
          <div className="flex items-center gap-6">
            <Metric label="Tokens/s" value={(stats?.throughput.decode_tps ?? 0).toFixed(1)} />
            <Metric label="Active" value={stats?.requests.active ?? 0} />
            <Metric label="Done" value={stats?.requests.completed ?? 0} />
            <Metric label="VRAM" value={`${gib(stats?.vram_bytes)}G`} />
          </div>
        )}

        <div className="flex gap-2">
          {engine.state === 'running' && (
            <Button tone="default" onClick={() => router.push('/chat')}>
              Chat
            </Button>
          )}
          {engine.state === 'running' && targets.length > 0 && (
            <select
              value=""
              disabled={moving || busy}
              onChange={(e) => e.target.value && move(e.target.value)}
              title="Unload from this GPU and reload on another, keeping the same options"
              className="rounded-lg border border-seam bg-panel-hi px-2 py-1.5 font-mono text-[12px] text-ink-dim outline-none transition-colors hover:border-accent/50 disabled:opacity-60"
            >
              <option value="">{moving ? 'moving…' : 'Move to…'}</option>
              {targets.map((idx) => (
                <option key={idx} value={idx}>
                  GPU {idx}
                </option>
              ))}
            </select>
          )}
          <Button tone="danger" onClick={stop} disabled={busy || engine.state === 'stopping'}>
            {busy ? '…' : 'Unload'}
          </Button>
        </div>
      </div>

      {moveError && (
        <div className="mt-3 rounded-lg border border-bad/40 bg-bad/[0.07] px-3 py-2 text-[12.5px] text-ink-dim">
          {moveError}
        </div>
      )}

      {loading && <LoadProgress load={engine.load!} elapsedS={engine.uptime_s} />}

      {engine.error && (
        <div className="mt-4 rounded-xl border border-bad/35 bg-bad/5 p-4">
          <div className="flex items-start gap-2">
            <span className="mt-[3px] text-bad">✕</span>
            <div className="min-w-0 flex-1">
              <div className="text-[13px] font-medium text-bad">
                Engine failed
                {engine.diagnosis?.exit_code != null && ` · exit code ${engine.diagnosis.exit_code}`}
              </div>
              <pre className="mt-1.5 whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed text-ink-dim">
                {engine.diagnosis?.summary ?? engine.error}
              </pre>
            </div>
          </div>

          {engine.diagnosis?.hint && (
            <div className="mt-3 rounded-lg border border-accent/30 bg-accent/[0.06] p-3">
              <div className="text-[11px] uppercase tracking-[0.14em] text-accent">How to fix</div>
              <div className="mt-1.5 text-[12.5px] leading-relaxed text-ink-dim">
                {engine.diagnosis.hint}
              </div>
            </div>
          )}

          {(engine.diagnosis?.tail?.length ?? 0) > 0 && (
            <details className="mt-3">
              <summary className="cursor-pointer text-[12px] text-ink-faint hover:text-ink-dim">
                Last {engine.diagnosis!.tail.length} output lines
              </summary>
              <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-all rounded-lg border border-seam bg-canvas p-3 font-mono text-[10.5px] leading-relaxed text-ink-faint">
                {engine.diagnosis!.tail.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </pre>
            </details>
          )}
        </div>
      )}
    </Panel>
  )
}
