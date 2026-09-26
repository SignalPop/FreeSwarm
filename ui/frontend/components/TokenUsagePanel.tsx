'use client'

import { useState } from 'react'
import { api, type TokenSource, type TokenUsageDoc, type TokenWindow } from '@/lib/api'
import { duration, humanCount } from '@/lib/format'
import { usePoll } from '@/lib/usePoll'
import { Panel } from '@/components/ui'

// Same colour language as the rest of the console: blue = this computer's capacity, violet =
// a paired computer, amber = a hosted model that costs money.
const SOURCE: Record<TokenSource, { label: string; badge: string; text: string; bar: string }> = {
  local: { label: 'local', badge: 'border-accent/35 bg-accent/10 text-accent', text: 'text-ink', bar: 'bg-accent' },
  network: { label: 'network', badge: 'border-remote/35 bg-remote/10 text-remote', text: 'text-remote', bar: 'bg-remote' },
  external: { label: 'external', badge: 'border-warn/35 bg-warn/10 text-warn', text: 'text-warn', bar: 'bg-warn' },
}

const WINDOWS: [TokenWindow, string][] = [
  ['all', 'All time'],
  ['7d', '7d'],
  ['24h', '24h'],
]

function ago(ts: number | null, now: number): string {
  if (!ts) return '—'
  const s = now - ts
  return s < 5 ? 'just now' : `${duration(s)} ago`
}

function exact(n: number): string {
  return `${n.toLocaleString()} tokens`
}

/**
 * Tokens processed per model, from every place a model can run. Local figures come from the
 * engines' own counters (so they include requests paired computers sent here); network and
 * external figures from the usage each reply reports. Kept across restarts by the control plane.
 */
export default function TokenUsagePanel() {
  const [win, setWin] = useState<TokenWindow>('all')
  const { data, error, refresh } = usePoll<TokenUsageDoc>(() => api.tokenUsage(win), 5000)
  // A reply for the previous window can land after the switch; show nothing stale meanwhile.
  const doc = data && data.window === win ? data : null
  const rows = doc?.models ?? []
  const grand = doc?.totals.total_tokens ?? 0
  const now = doc?.ts ?? Date.now() / 1000

  return (
    <Panel className="mb-6 p-5">
      <div className="mb-4 flex flex-wrap items-baseline gap-3">
        <span className="text-[15px] font-medium text-ink">Tokens by model</span>
        <span className="font-mono text-[11px] text-ink-faint">
          local engines, paired computers and external providers — kept across restarts
        </span>
        <div className="ml-auto flex overflow-hidden rounded-lg border border-seam">
          {WINDOWS.map(([w, label]) => (
            <button
              key={w}
              onClick={() => {
                setWin(w)
                refresh()
              }}
              className={`px-2.5 py-1 font-mono text-[11px] transition-colors ${
                win === w ? 'bg-accent/15 text-accent' : 'text-ink-faint hover:bg-panel-hi hover:text-ink-dim'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {doc && (
        <div className="mb-3 flex flex-wrap items-baseline gap-x-5 gap-y-1 font-mono text-[12px]">
          <span className="text-ink" title={exact(grand)}>
            {humanCount(grand)} <span className="text-ink-faint">total</span>
          </span>
          {(Object.keys(SOURCE) as TokenSource[]).map((s) => {
            const t = doc.by_source[s]
            return (
              <span key={s} className="text-ink-dim" title={exact(t.total_tokens)}>
                <span className={`mr-1.5 inline-block h-1.5 w-1.5 rounded-full ${SOURCE[s].bar}`} />
                {SOURCE[s].label} {humanCount(t.total_tokens)}
                <span className="text-ink-faint"> · {t.requests.toLocaleString()} req</span>
              </span>
            )
          })}
        </div>
      )}

      {error && !doc && <div className="text-[13px] text-bad">Could not read token usage: {error}</div>}

      {doc && rows.length === 0 && (
        <div className="rounded-xl border border-seam bg-panel-hi/40 px-4 py-6 text-center text-[13px] text-ink-faint">
          {win === 'all'
            ? 'No tokens counted yet — they appear here as models answer.'
            : 'No model answered in this window.'}
        </div>
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-[12.5px]">
            <thead>
              <tr className="border-b border-seam text-left font-mono text-[10px] uppercase tracking-wide text-ink-faint">
                <th className="py-1.5 pr-3 font-normal">model</th>
                <th className="px-3 font-normal">source</th>
                <th className="px-3 text-right font-normal">prompt</th>
                <th className="px-3 text-right font-normal">completion</th>
                <th className="px-3 text-right font-normal">total</th>
                <th className="px-3 text-right font-normal">requests</th>
                <th className="pl-3 text-right font-normal">last used</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const src = SOURCE[r.source]
                const share = grand > 0 ? (r.total_tokens / grand) * 100 : 0
                return (
                  <tr key={`${r.source}:${r.model}`} className="border-b border-seam/50 last:border-b-0">
                    <td className={`max-w-[340px] truncate py-2 pr-3 font-mono ${src.text}`} title={r.model}>
                      {r.model}
                    </td>
                    <td className="px-3">
                      <span className={`rounded-full border px-2 py-0.5 font-mono text-[10.5px] ${src.badge}`}>
                        {src.label}
                      </span>
                    </td>
                    <td className="px-3 text-right font-mono text-ink-dim" title={exact(r.prompt_tokens)}>
                      {humanCount(r.prompt_tokens)}
                    </td>
                    <td className="px-3 text-right font-mono text-ink-dim" title={exact(r.completion_tokens)}>
                      {humanCount(r.completion_tokens)}
                    </td>
                    <td className="px-3 text-right font-mono text-ink" title={`${exact(r.total_tokens)} · ${share.toFixed(1)}% of all`}>
                      <div>{humanCount(r.total_tokens)}</div>
                      <div className="ml-auto mt-1 h-[3px] w-16 overflow-hidden rounded-full bg-seam">
                        <div className={`ml-auto h-full rounded-full ${src.bar}`} style={{ width: `${Math.max(share, 2)}%` }} />
                      </div>
                    </td>
                    <td
                      className="px-3 text-right font-mono text-ink-dim"
                      title={
                        r.unmetered
                          ? `${r.unmetered} of these replies reported no token usage, so their tokens are not included`
                          : undefined
                      }
                    >
                      {r.requests.toLocaleString()}
                      {r.unmetered > 0 && <span className="text-warn">*</span>}
                    </td>
                    <td
                      className="pl-3 text-right font-mono text-[11.5px] text-ink-faint"
                      title={r.last_seen ? new Date(r.last_seen * 1000).toLocaleString() : undefined}
                    >
                      {ago(r.last_seen, now)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}
