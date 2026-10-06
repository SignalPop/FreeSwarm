'use client'

import type { EngineDetail, GpuInfo } from '@/lib/api'
import { bytesLabel } from '@/lib/format'

/**
 * Which cards a model could be loaded on right now, judged from each card's FREE VRAM.
 *
 * No splitting a model across cards (tensor parallelism needs NCCL, which has no Windows
 * build), so the question is per card: do the weights, plus what the engine needs beside
 * them, fit in what that card has free? For a mixture-of-experts model there is a second
 * route: offload the experts to host RAM and keep only the rest on the card. A card already
 * running an engine can take a second one, but only in the room that engine does not keep
 * for itself (shareRoom).
 */

export type Fit = 'fits' | 'offload' | 'shared' | 'busy' | 'no'

export type FitInput = {
  /** Bytes of weights (checkpoint size). */
  weightBytes: number | null | undefined
  isMoe?: boolean
  /** Bytes of routed experts, when known: what offload moves to host RAM. */
  expertBytes?: number | null
  /** KV-cache bytes per token of context, when known. */
  kvBytesPerToken?: number | null
}

export type GpuFit = {
  index: string
  name: string
  fit: Fit
  freeBytes: number
  needBytes: number
  /** VRAM needed with the experts offloaded (MoE only). */
  offloadBytes: number | null
  holder: string | null
  /** Whether it would fit on this card if the card were empty (for a busy card). */
  fitsEmpty: boolean
  /** On a card with an engine: what a second engine may use, and what the first one keeps. */
  shareBytes: number | null
  reserveBytes: number | null
}

// freetoken's EngineConfig.memory_ratio, which applies when a launch sets none.
export const ENGINE_DEFAULT_RATIO = 0.9
// The newcomer fills its claim with weights and KV cache; its own activations come out of
// what is left (engine.py SHARE_OWN_HEADROOM_BYTES).
const SHARE_OWN_HEADROOM_BYTES = 2 * 2 ** 30

export type ShareRoom = {
  holders: EngineDetail[]
  /** Every holder is serving; a card still loading is not shared (its use is not settled). */
  settled: boolean
  freeBytes: number
  /** Headroom the holders keep for activations and CUDA graphs: (1 - ratio) of the card each. */
  reserveBytes: number
  /** What a second engine may claim without eating into that headroom or its own. */
  roomBytes: number
}

/**
 * Room for a second engine on a card, by the same rule as the server's preflight_share:
 * an engine keeps (1 - its memory ratio) of the card as working room it allocates under
 * load, and a newcomer sized against "free" would otherwise take it.
 */
export function shareRoom(g: GpuInfo, engines: EngineDetail[] | undefined): ShareRoom {
  const index = String(g.index)
  const holders = (engines ?? []).filter((e) => e.state !== 'stopped' && String(e.gpus) === index)
  const free = Math.max(0, g.memory_total_bytes - g.memory_used_bytes)
  const reserve = holders.reduce((sum, e) => {
    const r = Number((e.options as Record<string, unknown>)?.memory_ratio) || ENGINE_DEFAULT_RATIO
    return sum + (1 - r) * g.memory_total_bytes
  }, 0)
  return {
    holders,
    settled: holders.every((e) => e.state === 'running'),
    freeBytes: free,
    reserveBytes: reserve,
    roomBytes: Math.max(0, free - reserve - SHARE_OWN_HEADROOM_BYTES),
  }
}

// Same numbers the launch form uses (app/models/page.tsx): loading overhead on the weights, and
// the CUDA context / activations / graph workspace beside them.
const WEIGHT_OVERHEAD = 1.06
const ENGINE_SLACK_BYTES = 1.5 * 2 ** 30
const FREE_USABLE = 0.95
// The smallest context worth loading an agent model with; its KV cache is reserved at load.
const MIN_CONTEXT = 16384
const KV_FALLBACK_BYTES = 1 * 2 ** 30
// Non-expert share of an MoE checkpoint when the split is unknown (attention, embeddings,
// shared experts): typically 5-15%; the high end keeps the estimate on the safe side.
const MOE_RESIDENT_SHARE = 0.15

const MOE_HINT = /moe|mixtral|gpt-?oss|deepseek-?v[23]|(?:^|[-_/])a\d+(?:\.\d+)?b(?:$|[-_])/i

/** A best guess from a name alone (Qwen3-30B-A3B, gpt-oss-20b, Mixtral...). */
export function looksMoe(name: string): boolean {
  return MOE_HINT.test(name)
}

export function gpuFits(
  m: FitInput,
  gpus: GpuInfo[] | undefined,
  engines: EngineDetail[] | undefined,
): GpuFit[] {
  const w = m.weightBytes ?? 0
  if (!w || !gpus?.length) return []
  const kv = m.kvBytesPerToken ? m.kvBytesPerToken * MIN_CONTEXT : KV_FALLBACK_BYTES
  const need = w * WEIGHT_OVERHEAD + ENGINE_SLACK_BYTES + kv
  const resident = m.isMoe
    ? m.expertBytes
      ? Math.max(0, w - m.expertBytes)
      : w * MOE_RESIDENT_SHARE
    : null
  const offloadNeed = resident != null ? resident * WEIGHT_OVERHEAD + ENGINE_SLACK_BYTES + kv : null
  return gpus.map((g) => {
    const index = String(g.index)
    const share = shareRoom(g, engines)
    const holder = share.holders[0]
    const free = share.freeBytes
    const room = free * FREE_USABLE
    const emptyRoom = g.memory_total_bytes * FREE_USABLE
    let fit: Fit
    if (holder) fit = share.settled && need <= share.roomBytes ? 'shared' : 'busy'
    else if (need <= room) fit = 'fits'
    else if (offloadNeed != null && offloadNeed <= room) fit = 'offload'
    else fit = 'no'
    return {
      index,
      name: g.name.replace(/^NVIDIA\s+(GeForce\s+)?/i, ''),
      fit,
      freeBytes: free,
      needBytes: need,
      offloadBytes: offloadNeed,
      holder: holder ? share.holders.map((e) => e.model_id || 'an engine').join(', ') : null,
      fitsEmpty: need <= emptyRoom || (offloadNeed != null && offloadNeed <= emptyRoom),
      shareBytes: holder ? share.roomBytes : null,
      reserveBytes: holder ? share.reserveBytes : null,
    }
  })
}

const TONE: Record<Fit, string> = {
  fits: 'border-good/40 bg-good/10 text-good',
  offload: 'border-warn/40 bg-warn/10 text-warn',
  shared: 'border-accent/40 bg-accent/10 text-accent',
  busy: 'border-seam bg-panel-hi text-ink-dim',
  no: 'border-seam text-ink-faint line-through decoration-ink-faint/60',
}

function title(f: GpuFit): string {
  const lines = [`GPU ${f.index} ${f.name}: ${bytesLabel(f.freeBytes)} free`, `needs ~${bytesLabel(f.needBytes)} to load whole`]
  if (f.offloadBytes != null) lines.push(`~${bytesLabel(f.offloadBytes)} with experts offloaded to host RAM`)
  if (f.holder && f.shareBytes != null && f.reserveBytes != null) {
    const room = `${f.holder} keeps ~${bytesLabel(f.reserveBytes)} as working room; ~${bytesLabel(f.shareBytes)} is left for a second engine`
    if (f.fit === 'shared') lines.push(`can share with ${f.holder}: ${room}`)
    else lines.push(`in use by ${f.holder}: ${room}${f.fitsEmpty ? ' -- would fit once it is unloaded' : ''}`)
  }
  return lines.join('\n')
}

function label(f: GpuFit): string {
  const where = `GPU ${f.index} ${f.name.replace(/^RTX\s+/i, '')}`
  if (f.fit === 'fits') return `${where} ✓`
  if (f.fit === 'offload') return `${where} · offload`
  if (f.fit === 'shared') return `${where} · shares`
  if (f.fit === 'busy') return `${where} · in use${f.fitsEmpty ? '' : ' · too small'}`
  return `${where} ✗`
}

/** One chip per card: fits / fits with offload / in use / too small (hover for the numbers). */
export function GpuFitChips({
  fits,
  className = '',
}: {
  fits: GpuFit[]
  className?: string
}) {
  if (!fits.length) return null
  return (
    <span className={`inline-flex flex-wrap items-center gap-1 ${className}`}>
      {fits.map((f) => (
        <span
          key={f.index}
          title={title(f)}
          className={`cursor-default rounded-md border px-1.5 py-0.5 font-mono text-[10.5px] ${TONE[f.fit]}`}
        >
          {label(f)}
        </span>
      ))}
    </span>
  )
}
