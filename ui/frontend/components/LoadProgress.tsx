'use client'

import type { LoadStatus } from '@/lib/api'
import { duration, gib } from '@/lib/format'

/**
 * What the engine is doing while a model loads.
 *
 * A single bar is not enough here: a 61 GiB offloaded model takes ~12 minutes, and the
 * phases behave completely differently — weight loading reports bytes, CUDA-graph capture
 * reports a batch count, and KV allocation reports nothing at all. So this shows an
 * overall bar plus a stepper, and each step carries its own number when it has one.
 *
 * Steps without a percentage still show as active rather than stalled, because "no number
 * available" and "nothing happening" look identical otherwise — which is exactly the
 * confusion this replaces.
 */
export default function LoadProgress({
  load,
  elapsedS,
}: {
  load: LoadStatus
  elapsedS: number
}) {
  const overall = Math.max(0, Math.min(100, load.overall_pct))

  return (
    <div className="mt-4 rounded-xl border border-accent/25 bg-accent/[0.04] p-4">
      {/* ---- headline ---- */}
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-[13px] font-medium text-ink">{load.label}</span>
        <span className="font-mono text-[12px] text-ink-dim">
          {load.estimated ? '~' : ''}
          {overall.toFixed(0)}% · {duration(elapsedS)} elapsed
        </span>
      </div>

      {load.detail && (
        <div className="mt-1 truncate font-mono text-[11px] text-ink-faint">{load.detail}</div>
      )}

      {/* ---- overall bar ---- */}
      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-seam">
        <div
          className="h-full rounded-full bg-accent transition-all duration-700 ease-out"
          style={{ width: `${overall}%` }}
        />
      </div>

      {/* ---- stepper ---- */}
      <div className="mt-4 space-y-1.5">
        {load.steps.map((step) => {
          const isActive = step.state === 'active'
          const isDone = step.state === 'done'
          return (
            <div key={step.slug} className="flex items-center gap-3">
              {/* marker */}
              <span
                className={`grid h-4 w-4 shrink-0 place-items-center rounded-full border text-[9px] ${
                  isDone
                    ? 'border-good bg-good text-canvas'
                    : isActive
                      ? 'border-accent text-accent'
                      : 'border-seam text-transparent'
                }`}
              >
                {isDone ? (
                  <svg viewBox="0 0 24 24" className="h-2.5 w-2.5" fill="none" stroke="currentColor" strokeWidth={4} strokeLinecap="round" strokeLinejoin="round">
                    <path d="M5 13l4 4L19 7" />
                  </svg>
                ) : isActive ? (
                  <span className="h-1.5 w-1.5 rounded-full bg-accent animate-dot" />
                ) : null}
              </span>

              <span
                className={`flex-1 text-[12px] ${
                  isDone ? 'text-ink-faint' : isActive ? 'text-ink' : 'text-ink-faint/60'
                }`}
              >
                {step.label}
              </span>

              {/* per-step bar, only while running and only when there is a number */}
              {isActive && step.pct !== null && (
                <span className="flex w-28 items-center gap-2">
                  <span className="h-1 flex-1 overflow-hidden rounded-full bg-seam">
                    <span
                      className="block h-full rounded-full bg-accent transition-all duration-700"
                      style={{ width: `${Math.max(0, Math.min(100, step.pct))}%` }}
                    />
                  </span>
                  <span className="w-8 text-right font-mono text-[10px] text-ink-dim">
                    {load.estimated ? '~' : ''}
                    {step.pct.toFixed(0)}%
                  </span>
                </span>
              )}

              {isActive && step.pct === null && (
                <span className="font-mono text-[10px] text-ink-faint">working…</span>
              )}
            </div>
          )
        })}
      </div>

      {load.estimated && (
        <div className="mt-2 font-mono text-[10px] text-ink-faint">
          approximate — estimated from memory growth, as the engine reports no byte count
          for this phase
        </div>
      )}

      {load.total_bytes > 0 && (
        <div className="mt-3 border-t border-seam/60 pt-2 font-mono text-[10.5px] text-ink-faint">
          {gib(load.done_bytes)} / {gib(load.total_bytes)} GiB read
        </div>
      )}

      {(load.phase === 'expert_banks' || load.phase === 'weights') && (
        <div className="mt-2 text-[11px] leading-relaxed text-ink-faint">
          Large offloaded models spend most of the load pinning experts into host RAM —
          roughly 12 minutes for a 61 GiB checkpoint. VRAM jumps near the end, when the
          resident expert cache and CUDA graphs are allocated.
        </div>
      )}
    </div>
  )
}
