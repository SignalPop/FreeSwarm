'use client'

import { useEffect, useState } from 'react'
import { api, type SandboxStatus } from '@/lib/api'

/**
 * Shared sandbox availability, cached across every code block on the page.
 *
 * Two things this fixes over a plain per-component fetch:
 *
 * 1. **One probe, not one per code block.** A long reply can contain a dozen runnable
 *    blocks; each mounting its own status request hammers the control plane for an answer
 *    that is identical every time.
 * 2. **A failed probe is not permanent.** The status is fetched once on mount, so a single
 *    transient failure — the control plane still booting when the page opened is the normal
 *    case — used to leave every Run button disabled forever, with no way back short of a
 *    reload. The cache is re-checked when it goes stale and can be refreshed on demand, and
 *    an unknown state is treated as *available* so the click surfaces the real error
 *    instead of the button silently refusing.
 */
type Cache = { at: number; value: SandboxStatus }

let cache: Cache | null = null
let inflight: Promise<SandboxStatus> | null = null
const listeners = new Set<(s: SandboxStatus) => void>()

const FRESH_MS = 15_000

function publish(value: SandboxStatus) {
  cache = { at: Date.now(), value }
  listeners.forEach((fn) => fn(value))
}

export function fetchSandboxStatus(force = false): Promise<SandboxStatus> {
  if (!force && cache && Date.now() - cache.at < FRESH_MS) {
    return Promise.resolve(cache.value)
  }
  if (inflight) return inflight
  inflight = api
    .sandboxStatus()
    .then((s) => {
      publish(s)
      return s
    })
    .catch((e) => {
      // Unknown, not unavailable: the control plane may simply not be up yet. Reporting
      // `available: true` keeps the button live, and a click then shows the real reason.
      const s: SandboxStatus = {
        available: true,
        reason: e instanceof Error ? e.message : undefined,
      }
      publish(s)
      return s
    })
    .finally(() => {
      inflight = null
    })
  return inflight
}

export function useSandboxStatus(enabled = true): SandboxStatus | null {
  const [status, setStatus] = useState<SandboxStatus | null>(cache?.value ?? null)

  useEffect(() => {
    if (!enabled) return
    listeners.add(setStatus)
    void fetchSandboxStatus()
    // Re-check periodically so starting Docker (or building the image) makes the buttons
    // live again without a reload.
    const timer = setInterval(() => void fetchSandboxStatus(true), 30_000)
    return () => {
      listeners.delete(setStatus)
      clearInterval(timer)
    }
  }, [enabled])

  return status
}
