'use client'

import { useEffect, useRef, useState } from 'react'
import { api, type LogEntry } from '@/lib/api'
import { clockTime } from '@/lib/format'
import { Button, PageHeader, Panel, Pill } from '@/components/ui'

export default function LogsPage() {
  const [entries, setEntries] = useState<LogEntry[]>([])
  const [follow, setFollow] = useState(true)
  const [filter, setFilter] = useState('')
  const [err, setErr] = useState<string | null>(null)

  const boxRef = useRef<HTMLDivElement>(null)
  const cursorRef = useRef(0)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    async function tick() {
      try {
        const res = await api.logs(cursorRef.current)
        if (cancelled) return
        if (res.entries.length) {
          // Cap retained lines: an engine that has been up for hours produces more than a
          // DOM should hold, and the server ring is the durable copy anyway.
          setEntries((prev) => [...prev, ...res.entries].slice(-5000))
          cursorRef.current = res.next_cursor
        }
        setErr(null)
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e))
      } finally {
        if (!cancelled) timer = setTimeout(tick, 1000)
      }
    }

    tick()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    if (follow) boxRef.current?.scrollTo({ top: boxRef.current.scrollHeight })
  }, [entries, follow])

  const needle = filter.trim().toLowerCase()
  const shown = needle
    ? entries.filter((e) => e.text.toLowerCase().includes(needle))
    : entries

  return (
    <div className="mx-auto flex h-full max-w-[1180px] flex-col px-8 py-8">
      <PageHeader
        title="Logs"
        subtitle="stdout and stderr from the engine process, captured by the control plane"
        right={
          <div className="flex items-center gap-2">
            <Pill tone={err ? 'bad' : 'good'} pulse={!err}>
              {err ? 'Disconnected' : `${entries.length} lines`}
            </Pill>
            <Button tone="ghost" onClick={() => setEntries([])}>
              Clear view
            </Button>
          </div>
        }
      />

      <Panel className="mb-3 flex flex-wrap items-center gap-4 px-4 py-2.5">
        <input
          className="min-w-[220px] flex-1 rounded-lg border border-seam bg-panel-hi px-3 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent"
          placeholder="Filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <label className="flex items-center gap-2 text-[12px] text-ink-dim">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          Follow tail
        </label>
      </Panel>

      <div
        ref={boxRef}
        onScroll={(e) => {
          // Scrolling away from the bottom turns off follow, the way a terminal pager does.
          const el = e.currentTarget
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
          if (!atBottom && follow) setFollow(false)
        }}
        className="min-h-0 flex-1 overflow-y-auto rounded-2xl border border-seam bg-panel p-4"
      >
        {shown.length === 0 && (
          <div className="grid h-full place-items-center text-[13px] text-ink-faint">
            {entries.length === 0 ? 'No output yet — start an engine.' : 'Nothing matches the filter.'}
          </div>
        )}
        {shown.map((e) => (
          <div key={e.seq} className="flex gap-3 font-mono text-[11.5px] leading-relaxed">
            <span className="shrink-0 text-ink-faint">{clockTime(e.ts)}</span>
            <span
              className={`shrink-0 ${
                e.stream === 'stderr'
                  ? 'text-warn'
                  : e.stream === 'ui'
                    ? 'text-accent'
                    : 'text-ink-faint'
              }`}
            >
              {e.stream.padEnd(6)}
            </span>
            <span className="whitespace-pre-wrap break-all text-ink-dim">{e.text}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
