'use client'

import { useEffect, useState } from 'react'
import { Panel } from '@/components/ui'

/**
 * Whether a disqualified result retires the library modules it was built on, and every
 * other result standing on them.
 *
 * On by default. A leak lives in the module, not in the script that imported it, so
 * without this a signal proven to look ahead stays `active`, the agents keep importing
 * it, and the swarm spends the night producing better-scoring versions of the same
 * invalid result.
 */
export default function SwarmIntegritySettings() {
  const [enabled, setEnabled] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    fetch('/api/auto-quarantine')
      .then((r) => r.json())
      .then((d) => alive && setEnabled(!!d.enabled))
      .catch((e) => alive && setErr(String(e)))
    return () => {
      alive = false
    }
  }, [])

  async function toggle() {
    if (enabled === null) return
    setBusy(true)
    setErr(null)
    try {
      const res = await fetch('/api/auto-quarantine', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !enabled }),
      })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      setEnabled(!!(await res.json()).enabled)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel className="mb-6 p-5">
      <div className="mb-2 text-[15px] font-medium text-ink">Swarm integrity</div>
      <p className="mb-3 text-[12px] text-ink-faint">
        A leak lives in the signal module, not in the script that imported it. When this is on, a
        result disqualified by the look-ahead check, a failed audit or an operator demotion also
        retires the library modules it was built on — and disqualifies every other result that
        imported them, then re-crowns from what is left.
      </p>

      <div className="flex items-start justify-between gap-4 border-t border-seam/60 pt-3">
        <div className="min-w-0">
          <div className="text-[13px] text-ink">Retire signals with look-ahead bias automatically</div>
          <div className="mt-0.5 text-[11.5px] text-ink-faint">
            Off means bad signals stay importable and the swarm keeps refining them. Leave it on
            unless you are triaging by hand.
          </div>
        </div>
        <button
          onClick={toggle}
          disabled={busy || enabled === null}
          role="switch"
          aria-checked={!!enabled}
          className={`mt-0.5 h-6 w-11 shrink-0 rounded-full border transition-colors disabled:opacity-40 ${
            enabled ? 'border-good/50 bg-good/30' : 'border-seam bg-panel-hi'
          }`}
        >
          <span
            className={`block size-4 rounded-full transition-transform ${
              enabled ? 'translate-x-5 bg-good' : 'translate-x-1 bg-ink-faint'
            }`}
          />
        </button>
      </div>

      {err && <div className="mt-2 text-[12px] text-bad">{err}</div>}
    </Panel>
  )
}
