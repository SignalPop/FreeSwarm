'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { Panel } from '@/components/ui'
import { bugsApi, type MonitorConfig, type MonitorDoc } from '@/lib/bugs'

function Switch({ on, disabled, onClick }: { on: boolean; disabled?: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      role="switch"
      aria-checked={on}
      className={`mt-0.5 h-6 w-11 shrink-0 rounded-full border transition-colors disabled:opacity-40 ${
        on ? 'border-good/50 bg-good/30' : 'border-seam bg-panel-hi'
      }`}
    >
      <span
        className={`block size-4 rounded-full transition-transform ${on ? 'translate-x-5 bg-good' : 'translate-x-1 bg-ink-faint'}`}
      />
    </button>
  )
}

/**
 * The monitoring agent: reads every agent's log and the engines' state each minute and files
 * what it finds on the Bugs page. Off by default; the backend reads the switch every tick.
 */
export default function MonitorSettings() {
  const [doc, setDoc] = useState<MonitorDoc | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    bugsApi
      .monitor()
      .then((d) => alive && setDoc(d))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [])

  async function save(changes: Partial<MonitorConfig>) {
    setBusy(true)
    setErr(null)
    try {
      setDoc(await bugsApi.configure(changes))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const cfg = doc?.config
  const models = Array.from(new Set([...(doc?.models ?? []), ...(cfg?.model ? [cfg.model] : [])]))

  return (
    <Panel className="mb-6 p-5">
      <div className="mb-2 text-[15px] font-medium text-ink">Monitoring agent</div>
      <p className="mb-3 text-[12px] text-ink-faint">
        Reads every agent&apos;s log — the same records as the agent inspector — and the engines&apos; state once a
        minute, and files problems on the <Link href="/bugs" className="text-accent hover:underline">Bugs</Link> page:
        errors raised by the platform rather than the agent, bad or missing data, stalled model replies and tool
        calls, spending limits, engine crashes. An agent&apos;s own coding mistakes are filed only once they recur.
      </p>

      <div className="flex items-start justify-between gap-4 border-t border-seam/60 pt-3">
        <div className="min-w-0">
          <div className="text-[13px] text-ink">Watch the logs and file bugs</div>
          <div className="mt-0.5 text-[11.5px] text-ink-faint">
            Rules only: no model calls, negligible cost. A scan can also be run by hand from the Bugs page.
          </div>
        </div>
        <Switch on={!!cfg?.enabled} disabled={busy || !cfg} onClick={() => save({ enabled: !cfg?.enabled })} />
      </div>

      <div className="mt-3 flex items-start justify-between gap-4 border-t border-seam/60 pt-3">
        <div className="min-w-0">
          <div className="text-[13px] text-ink">Model triage</div>
          <div className="mt-0.5 text-[11.5px] text-ink-faint">
            A loaded model writes each new bug up (likely cause, suggested fix, severity) and reads finished
            iterations for bad data no rule catches — a few short requests a minute. Uses the best local model
            unless one is picked here; an external model is only used when picked.
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-[12px] text-ink-dim">
            <label className="flex items-center gap-2">
              model
              <select
                value={cfg?.model ?? ''}
                disabled={busy || !cfg}
                onChange={(e) => save({ model: e.target.value })}
                className="rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[11.5px] text-ink outline-none focus:border-accent"
              >
                <option value="">best local model (auto)</option>
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>
            {doc?.state.llm_note && cfg?.enabled && cfg.llm_triage && (
              <span className="text-warn">{doc.state.llm_note}</span>
            )}
          </div>
        </div>
        <Switch
          on={!!cfg?.llm_triage}
          disabled={busy || !cfg}
          onClick={() => save({ llm_triage: !cfg?.llm_triage })}
        />
      </div>

      <div className="mt-3 flex items-start justify-between gap-4 border-t border-seam/60 pt-3">
        <div className="min-w-0">
          <div className="text-[13px] text-ink">Close bugs the logs show fixed</div>
          <div className="mt-0.5 text-[11.5px] text-ink-faint">
            Only on evidence: the tool, model or kind of iteration it happened in ran again that many times without
            it, and a quiet period passed. An engine crash counts as fixed when that model runs again. Bugs filed by
            hand or found by the model&apos;s review are never closed automatically. A closed bug reopens if it
            comes back; every step is written into the bug&apos;s notes.
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-[12px] text-ink-dim">
            <label className="flex items-center gap-2">
              after
              <input
                type="number"
                min={1}
                key={`fa-${cfg?.fixed_after}`}
                defaultValue={cfg?.fixed_after ?? 15}
                disabled={busy || !cfg}
                onBlur={(e) => {
                  const v = Number(e.target.value)
                  if (v >= 1 && v !== cfg?.fixed_after) save({ fixed_after: v })
                }}
                className="w-16 rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[11.5px] text-ink outline-none focus:border-accent"
              />
              clean chances
            </label>
            <label className="flex items-center gap-2">
              and
              <input
                type="number"
                min={0}
                key={`fq-${cfg?.fixed_quiet_minutes}`}
                defaultValue={cfg?.fixed_quiet_minutes ?? 30}
                disabled={busy || !cfg}
                onBlur={(e) => {
                  const v = Number(e.target.value)
                  if (v >= 0 && v !== cfg?.fixed_quiet_minutes) save({ fixed_quiet_minutes: v })
                }}
                className="w-16 rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[11.5px] text-ink outline-none focus:border-accent"
              />
              quiet minutes
            </label>
          </div>
        </div>
        <Switch
          on={cfg?.auto_close ?? true}
          disabled={busy || !cfg}
          onClick={() => save({ auto_close: !(cfg?.auto_close ?? true) })}
        />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-5 border-t border-seam/60 pt-3 text-[12px] text-ink-dim">
        <label className="flex items-center gap-2">
          stall after
          <input
            type="number"
            min={1}
            key={`stall-${cfg?.stall_minutes}`}
            defaultValue={cfg?.stall_minutes ?? 15}
            disabled={busy || !cfg}
            onBlur={(e) => {
              const v = Number(e.target.value)
              if (v >= 1 && v !== cfg?.stall_minutes) save({ stall_minutes: v })
            }}
            className="w-16 rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[11.5px] text-ink outline-none focus:border-accent"
          />
          min waiting on one reply or tool call
        </label>
        <label className="flex items-center gap-2">
          slow reply
          <input
            type="number"
            min={30}
            key={`slow-${cfg?.slow_reply_s}`}
            defaultValue={cfg?.slow_reply_s ?? 600}
            disabled={busy || !cfg}
            onBlur={(e) => {
              const v = Number(e.target.value)
              if (v >= 30 && v !== cfg?.slow_reply_s) save({ slow_reply_s: v })
            }}
            className="w-20 rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[11.5px] text-ink outline-none focus:border-accent"
          />
          s
        </label>
      </div>

      {err && <div className="mt-2 text-[12px] text-bad">{err}</div>}
    </Panel>
  )
}
