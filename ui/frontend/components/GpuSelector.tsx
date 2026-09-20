'use client'

import { useEffect, useState } from 'react'
import type { GpuInfo } from '@/lib/api'
import { gib } from '@/lib/format'
import { Button, Panel, Pill } from '@/components/ui'

type GpuDoc = {
  gpus: (GpuInfo & { enabled: boolean })[]
  visible_devices: string
  all_gpus: boolean
  applies_to_next_launch: boolean
}

/**
 * Enable/disable GPUs for the engine pool.
 *
 * This writes `CUDA_VISIBLE_DEVICES` for the *next* engine launch. CUDA reads that
 * variable once, when a process initialises its context, so a running engine keeps the
 * cards it started with -- the banner says so rather than letting the toggle look broken.
 *
 * Indices are nvidia-smi (PCI bus) order, because the control plane pins
 * CUDA_DEVICE_ORDER=PCI_BUS_ID.
 */
export default function GpuSelector() {
  const [doc, setDoc] = useState<GpuDoc | null>(null)
  const [draft, setDraft] = useState<Set<number> | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  async function load() {
    try {
      const res = await fetch('/api/gpus')
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const d: GpuDoc = await res.json()
      setDoc(d)
      setDraft(new Set(d.gpus.filter((g) => g.enabled).map((g) => g.index)))
      setErr(null)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  useEffect(() => {
    load()
  }, [])

  function toggle(index: number) {
    if (!draft) return
    const next = new Set(draft)
    if (next.has(index)) next.delete(index)
    else next.add(index)
    setDraft(next)
    setSaved(false)
  }

  async function apply(allGpus = false) {
    if (!draft) return
    setBusy(true)
    setErr(null)
    try {
      const res = await fetch('/api/gpus', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ indices: [...draft], all_gpus: allGpus }),
      })
      const body = await res.json()
      if (!res.ok) throw new Error(body?.detail ?? `${res.status} ${res.statusText}`)
      setSaved(true)
      await load()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  if (!doc || !draft) {
    return (
      <Panel className="p-5">
        <div className="text-[15px] font-medium text-ink">GPU pool</div>
        <div className="mt-2 text-[12px] text-ink-faint">{err ?? 'Loading…'}</div>
      </Panel>
    )
  }

  const current = new Set(doc.gpus.filter((g) => g.enabled).map((g) => g.index))
  const dirty =
    draft.size !== current.size || [...draft].some((i) => !current.has(i))

  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[15px] font-medium text-ink">GPU pool</span>
        {doc.all_gpus ? (
          <Pill tone="accent">all GPUs</Pill>
        ) : (
          <Pill tone="accent">devices {doc.visible_devices || 'none'}</Pill>
        )}
        <span className="ml-auto text-[12px] text-ink-faint">
          {saved && !dirty ? 'Saved' : dirty ? 'Modified' : 'Synced'}
        </span>
      </div>

      <p className="mt-1.5 text-[12px] leading-relaxed text-ink-faint">
        Which cards the engine may use. Indices match <code className="font-mono">nvidia-smi</code>.
        Deselect a GPU to keep it free for other work.
      </p>

      <div className="mt-4 space-y-2">
        {doc.gpus.map((g) => {
          const on = draft.has(g.index)
          const usedPct =
            g.memory_total_bytes > 0 ? (g.memory_used_bytes / g.memory_total_bytes) * 100 : 0
          return (
            <button
              key={g.index}
              onClick={() => toggle(g.index)}
              className={`flex w-full items-center gap-3 rounded-xl border p-3 text-left transition-colors ${
                on
                  ? 'border-accent/45 bg-accent/[0.07]'
                  : 'border-seam bg-panel-hi/30 opacity-60 hover:opacity-90'
              }`}
            >
              <span
                className={`grid h-5 w-5 shrink-0 place-items-center rounded-md border ${
                  on ? 'border-accent bg-accent text-white' : 'border-seam text-transparent'
                }`}
              >
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 13l4 4L19 7" />
                </svg>
              </span>

              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13px] text-ink">
                  <span className="font-mono text-ink-faint">[{g.index}]</span> {g.name}
                </span>
                <span className="mt-0.5 block font-mono text-[10.5px] text-ink-faint">
                  sm_{g.compute_cap.replace('.', '')} · {gib(g.memory_used_bytes)} /{' '}
                  {gib(g.memory_total_bytes)} GiB used
                  {g.temperature_c !== null && ` · ${g.temperature_c.toFixed(0)}°C`}
                </span>
              </span>

              <span className="w-16 shrink-0">
                <span className="block h-1 w-full overflow-hidden rounded-full bg-seam">
                  <span
                    className={`block h-full rounded-full ${usedPct > 90 ? 'bg-bad' : usedPct > 66 ? 'bg-warn' : 'bg-good'}`}
                    style={{ width: `${usedPct}%` }}
                  />
                </span>
              </span>
            </button>
          )
        })}
      </div>

      {draft.size === 0 && (
        <div className="mt-3 rounded-xl border border-warn/35 bg-warn/5 p-3 text-[12px] text-warn">
          No GPUs selected — the engine cannot start. Pick at least one, or use “All GPUs”.
        </div>
      )}

      {err && (
        <div className="mt-3 rounded-xl border border-bad/35 bg-bad/5 p-3 font-mono text-[11.5px] text-bad">
          {err}
        </div>
      )}

      {doc.applies_to_next_launch && dirty && (
        <div className="mt-3 rounded-xl border border-accent/30 bg-accent/[0.05] p-3 text-[12px] text-ink-dim">
          An engine is running. CUDA reads the device list once at process start, so this
          takes effect on the <strong className="text-ink">next</strong> launch — stop and
          restart the engine to apply it.
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Button tone="primary" onClick={() => apply(false)} disabled={busy || !dirty || draft.size === 0}>
          {busy ? 'Saving…' : 'Apply selection'}
        </Button>
        <Button tone="ghost" onClick={() => apply(true)} disabled={busy || doc.all_gpus}>
          Use all GPUs
        </Button>
        <Button
          tone="ghost"
          onClick={() => {
            setDraft(new Set(current))
            setSaved(false)
          }}
          disabled={!dirty}
        >
          Reset
        </Button>
      </div>
    </Panel>
  )
}
