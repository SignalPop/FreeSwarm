'use client'

import { useEffect, useMemo, useState } from 'react'
import { api, type CacheDoc } from '@/lib/api'
import { compactTokens, gib } from '@/lib/format'
import { Button, Panel, Pill } from '@/components/ui'

/**
 * Live cache-geometry editor, mirroring the desktop app's cache panel.
 *
 * The engine exposes `/v1/cache/status` (current pools + per-unit VRAM costs + the slider
 * bounds it will actually accept) and `/v1/cache/rebuild` (drain, reallocate, resume). The
 * two sliders here are denominated the way the engine thinks about them:
 *
 *   KV size  -> pages, shown to the user as tokens (pages x page_size)
 *   MoE cache-> expert slots, shown as slots and the VRAM they cost
 *
 * Both compute their VRAM cost from `unit_bytes` so the number next to the slider is the
 * engine's own arithmetic rather than an estimate that can drift from it.
 */
export default function CachePanel({
  doc,
  running,
  onApplied,
}: {
  doc: CacheDoc | null
  running: boolean
  onApplied: () => void
}) {
  const geo = doc?.geometry
  const [pages, setPages] = useState<number | null>(null)
  const [moe, setMoe] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  // Re-seed the controls from the engine whenever it reports a new allocation, but never
  // while the user is mid-edit (dirty) -- otherwise the 1 Hz poll would yank the slider
  // back under their thumb.
  const dirty =
    geo != null && ((pages !== null && pages !== geo.num_pages) || (moe !== null && moe !== geo.moe_cache_size))

  useEffect(() => {
    if (!geo || dirty) return
    setPages(geo.num_pages)
    setMoe(geo.moe_cache_size)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geo?.num_pages, geo?.moe_cache_size])

  const limits = geo?.limits ?? {}
  const kvLimit = limits['num_pages'] ?? {}
  const moeLimit = limits['moe_cache_size'] ?? {}

  const kvMin = Math.max(1, kvLimit.min ?? 512)
  const kvMax = Math.max(kvMin, kvLimit.max ?? Math.max(geo?.num_pages ?? 8192, 8192))
  const moeMin = moeLimit.min ?? 0
  const moeMax = Math.max(moeMin, moeLimit.max ?? Math.max(geo?.moe_cache_size ?? 0, 1))

  const pageSize = geo?.page_size ?? 1
  const kvTokens = (pages ?? 0) * pageSize
  const kvBytes = kvTokens * (geo?.unit_bytes.kv_per_token ?? 0)
  const moeBytes = (moe ?? 0) * (geo?.unit_bytes.moe_per_expert ?? 0)

  const totalExperts = useMemo(
    () => (geo ? geo.num_experts * geo.num_moe_layers : 0),
    [geo],
  )

  async function apply() {
    if (pages === null) return
    setBusy(true)
    setErr(null)
    try {
      const payload: Record<string, number> = { num_pages: pages }
      if (moe !== null && totalExperts > 0) payload.moe_cache_size = moe
      await api.rebuildCache(payload)
      onApplied()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  function reset() {
    if (!geo) return
    setPages(geo.num_pages)
    setMoe(geo.moe_cache_size)
    setErr(null)
  }

  if (!geo) {
    return (
      <Panel className="p-5">
        <div className="text-[15px] font-medium text-ink">Cache config</div>
        <p className="mt-2 text-[13px] text-ink-faint">
          Available once an engine is running — the pools and their VRAM costs are reported by
          the engine itself.
        </p>
      </Panel>
    )
  }

  const rebuilding = doc?.state === 'rebuilding'

  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[15px] font-medium text-ink">Cache config</span>
        {geo.num_experts > 0 && <Pill tone="accent">MoE</Pill>}
        {geo.moe_cache_policy && <Pill>{geo.moe_cache_policy.toUpperCase()}</Pill>}
        <span className="ml-auto text-[12px] text-ink-faint">
          {rebuilding ? 'Rebuilding…' : dirty ? 'Modified — click Apply' : 'Synced'}
        </span>
      </div>

      {geo.cache_budget_bytes > 0 && (
        <p className="mt-1.5 text-[12px] text-ink-faint">
          Engine cache budget {gib(geo.cache_budget_bytes)} GiB across all pools.
        </p>
      )}

      {/* ---- KV cache ---- */}
      <div className="mt-6">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <div className="text-[13px] text-ink">KV cache size</div>
            <div className="mt-0.5 text-[12px] text-ink-faint">
              Max token window for the attention KV cache
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="font-mono text-[17px] text-ink">{compactTokens(kvTokens)} tokens</span>
            {kvBytes > 0 && <Pill tone="accent">{gib(kvBytes, 2)} GiB</Pill>}
          </div>
        </div>
        <input
          type="range"
          className="mt-3"
          min={kvMin}
          max={kvMax}
          step={Math.max(1, Math.round((kvMax - kvMin) / 512))}
          value={pages ?? kvMin}
          disabled={!running || rebuilding}
          onChange={(e) => setPages(Number(e.target.value))}
        />
        <div className="mt-1.5 flex justify-between font-mono text-[11px] text-ink-faint">
          <span>{compactTokens(kvMin * pageSize)}</span>
          <span>{compactTokens(kvMax * pageSize)}</span>
        </div>
      </div>

      {/* ---- MoE expert cache ---- */}
      {totalExperts > 0 && (
        <div className="mt-7">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <div>
              <div className="text-[13px] text-ink">MoE expert cache</div>
              <div className="mt-0.5 text-[12px] text-ink-faint">
                Expert slots resident in VRAM · {totalExperts.toLocaleString()} across all MoE
                layers
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="font-mono text-[17px] text-ink">
                {(moe ?? 0).toLocaleString()}{' '}
                <span className="text-ink-faint">/ {totalExperts.toLocaleString()}</span>
              </span>
              {moeBytes > 0 && <Pill tone="accent">{gib(moeBytes, 1)} GiB</Pill>}
            </div>
          </div>
          <input
            type="range"
            className="mt-3"
            min={moeMin}
            max={moeMax}
            step={Math.max(1, Math.round((moeMax - moeMin) / 256))}
            value={moe ?? moeMin}
            disabled={!running || rebuilding}
            onChange={(e) => setMoe(Number(e.target.value))}
          />
          <div className="mt-1.5 flex justify-between font-mono text-[11px] text-ink-faint">
            <span>{moeMin.toLocaleString()}</span>
            <span>{moeMax.toLocaleString()}</span>
          </div>
        </div>
      )}

      {err && (
        <div className="mt-4 rounded-xl border border-bad/35 bg-bad/5 p-3 font-mono text-[12px] text-bad">
          {err}
        </div>
      )}

      <div className="mt-6 flex items-center gap-2">
        <Button tone="primary" onClick={apply} disabled={!running || !dirty || busy || rebuilding}>
          {busy ? 'Applying…' : 'Apply'}
        </Button>
        <Button tone="ghost" onClick={reset} disabled={!dirty || busy}>
          Reset
        </Button>
        <span className="ml-1 text-[12px] text-ink-faint">
          Applying drains in-flight requests and reallocates VRAM — no model reload.
        </span>
      </div>
    </Panel>
  )
}
