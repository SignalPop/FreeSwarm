'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

export type PollResult<T> = {
  data: T | null
  error: string | null
  loading: boolean
  refresh: () => void
}

/**
 * Poll an async source on an interval.
 *
 * Two behaviours that matter for a dashboard:
 *
 * - The next tick is scheduled *after* the previous one settles, not on a fixed timer, so
 *   a slow request (an nvidia-smi stall, a cache rebuild holding the engine) cannot pile
 *   up overlapping in-flight requests.
 * - A failed poll keeps the last good `data` and surfaces `error` alongside it, so the
 *   console shows stale-but-real numbers rather than blanking out when the engine restarts.
 */
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs: number): PollResult<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)

  // Held in a ref so changing the fetcher identity between renders (an inline arrow, say)
  // does not tear down and restart the polling loop on every render.
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const refresh = useCallback(() => setNonce((n) => n + 1), [])

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    async function tick() {
      try {
        const next = await fetcherRef.current()
        if (cancelled) return
        setData(next)
        setError(null)
      } catch (err) {
        if (cancelled) return
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        if (!cancelled) {
          setLoading(false)
          timer = setTimeout(tick, intervalMs)
        }
      }
    }

    tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [intervalMs, nonce])

  return { data, error, loading, refresh }
}
