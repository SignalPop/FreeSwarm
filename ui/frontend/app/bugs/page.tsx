'use client'

import Link from 'next/link'
import { useCallback, useEffect, useState } from 'react'
import { Button, EmptyState, PageHeader, Panel, Pill } from '@/components/ui'
import {
  bugAsText,
  bugsApi,
  PRIORITIES,
  SEVERITIES,
  STATUSES,
  type Bug,
  type BugCounts,
  type BugStatus,
  type BugSummary,
  type MonitorDoc,
  type Priority,
  type Recheck,
  type Severity,
} from '@/lib/bugs'
import { duration } from '@/lib/format'
import { usePoll } from '@/lib/usePoll'

const REFRESH_MS = 10_000

function ago(ts: number | null | undefined): string {
  return ts ? `${duration(Date.now() / 1000 - ts)} ago` : '—'
}

function stamp(ts: number | null | undefined): string {
  return ts ? new Date(ts * 1000).toLocaleString(undefined, { hour12: false }) : '—'
}

const SEV_TONE: Record<Severity, 'bad' | 'warn' | 'neutral'> = {
  critical: 'bad',
  high: 'bad',
  medium: 'warn',
  low: 'neutral',
}

const PRI_TEXT: Record<Priority, string> = {
  P1: 'border-bad/40 bg-bad/10 text-bad',
  P2: 'border-warn/40 bg-warn/10 text-warn',
  P3: 'border-seam bg-panel-hi text-ink-dim',
  P4: 'border-seam bg-panel-hi text-ink-faint',
}

const STATUS_TONE: Record<BugStatus, 'bad' | 'warn' | 'good'> = { open: 'bad', pending: 'warn', closed: 'good' }

async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    // No clipboard API (or permission refused): the old way still works in every browser.
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    document.execCommand('copy')
    ta.remove()
  }
}

/**
 * Copy the whole bug as Markdown. The icon is the usual two overlapping boxes: the front one
 * lifts on hover; on click they fold together into a check mark that draws itself, with a ring
 * pulsing out, so it is obvious from across the desk that the copy happened.
 */
function CopyBugButton({ load, size = 'md' }: { load: () => Promise<Bug>; size?: 'sm' | 'md' }) {
  const [state, setState] = useState<'idle' | 'busy' | 'done' | 'error'>('idle')
  const done = state === 'done'

  async function copy(e: React.MouseEvent) {
    e.stopPropagation()
    if (state === 'busy') return
    setState('busy')
    try {
      await copyText(bugAsText(await load()))
      setState('done')
    } catch {
      setState('error')
    }
    setTimeout(() => setState('idle'), 1600)
  }

  const box = size === 'sm' ? 'h-7 w-7' : 'h-8 w-8'
  const icon = size === 'sm' ? 'h-[15px] w-[15px]' : 'h-4 w-4'
  return (
    <button
      type="button"
      onClick={copy}
      title={done ? 'Copied the whole bug' : state === 'error' ? 'Copy failed' : 'Copy the whole bug (Markdown)'}
      aria-label="Copy the whole bug"
      className={`group relative grid ${box} shrink-0 place-items-center rounded-lg border transition-colors duration-300 ${
        done
          ? 'border-good/50 bg-good/10 text-good'
          : state === 'error'
            ? 'border-bad/50 text-bad'
            : 'border-seam text-ink-dim hover:border-accent/60 hover:text-ink'
      }`}
    >
      {done && <span className="pointer-events-none absolute inset-0 animate-ping rounded-lg border border-good/60" />}
      <svg viewBox="0 0 24 24" className={`${icon} overflow-visible`} fill="none" stroke="currentColor" strokeWidth={1.8}
        strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        {/* back box */}
        <rect x="8" y="8" width="12" height="12" rx="2.5" style={{ transformBox: 'fill-box' }}
          className={`origin-center transition-all duration-300 ${done ? 'scale-50 opacity-0' : 'opacity-100'}`} />
        {/* front box: lifts on hover, slides onto the back one when copied */}
        <path d="M16 8V6.5A2.5 2.5 0 0 0 13.5 4h-7A2.5 2.5 0 0 0 4 6.5v7A2.5 2.5 0 0 0 6.5 16H8"
          className={`transition-all duration-300 ${
            done ? 'translate-x-1 translate-y-1 opacity-0' : state === 'busy' ? 'translate-x-0.5 translate-y-0.5'
              : 'group-hover:-translate-x-0.5 group-hover:-translate-y-0.5'
          }`} />
        {/* check mark, drawn in */}
        <path d="M6 12.5l4 4 8-9" strokeWidth={2.2}
          style={{ strokeDasharray: 24, strokeDashoffset: done ? 0 : 24, transition: 'stroke-dashoffset 380ms ease-out 120ms' }} />
      </svg>
    </button>
  )
}

function PriorityChip({ p }: { p: Priority }) {
  return <span className={`rounded-md border px-1.5 py-0.5 font-mono text-[11px] ${PRI_TEXT[p]}`}>{p}</span>
}

function Section({ title, right, children }: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="border-t border-seam/60 pt-3">
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <h3 className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-ink-faint">{title}</h3>
        {right}
      </div>
      {children}
    </div>
  )
}

function CodeBlock({ text, empty }: { text: string; empty: string }) {
  const [copied, setCopied] = useState(false)
  if (!text) return <div className="text-[12px] text-ink-faint">{empty}</div>
  return (
    <div className="relative">
      <button
        onClick={() => {
          navigator.clipboard?.writeText(text).then(() => {
            setCopied(true)
            setTimeout(() => setCopied(false), 1200)
          })
        }}
        className="absolute right-2 top-2 rounded-md border border-seam bg-panel px-2 py-0.5 font-mono text-[10.5px] text-ink-dim hover:text-ink"
      >
        {copied ? 'copied' : 'copy'}
      </button>
      <pre className="max-h-[360px] overflow-auto whitespace-pre-wrap break-all rounded-xl border border-seam bg-canvas p-3 pr-16 font-mono text-[11.5px] leading-relaxed text-ink-dim">
        {text}
      </pre>
    </div>
  )
}

const selectCls =
  'rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[12px] text-ink outline-none focus:border-accent disabled:opacity-40'

/** One bug in full: what was found, where, the script and evidence, and its triage fields. */
function BugDetail({ id, onChanged, onClose }: { id: number; onChanged: () => void; onClose: () => void }) {
  const [bug, setBug] = useState<Bug | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [notes, setNotes] = useState('')
  const [rechecking, setRechecking] = useState(false)
  const [verdict, setVerdict] = useState<Recheck | null>(null)

  const load = useCallback(() => {
    bugsApi
      .get(id)
      .then((b) => {
        setBug(b)
        setNotes(b.notes)
        setErr(null)
      })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)))
  }, [id])

  useEffect(() => {
    setBug(null)
    load()
  }, [load])

  async function patch(changes: Parameters<typeof bugsApi.patch>[1]) {
    setBusy(true)
    try {
      const b = await bugsApi.patch(id, changes)
      setBug(b)
      setNotes(b.notes)
      setErr(null)
      onChanged()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function recheck() {
    setRechecking(true)
    setVerdict(null)
    try {
      const r = await bugsApi.recheck(id)
      setVerdict(r)
      setBug(r.bug)
      setNotes(r.bug.notes)
      setErr(null)
      onChanged()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setRechecking(false)
    }
  }

  async function remove() {
    if (!confirm('Delete this bug? If the monitor sees the problem again it is filed afresh — close it to keep it closed.'))
      return
    setBusy(true)
    try {
      await bugsApi.remove(id)
      onChanged()
      onClose()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  if (!bug) {
    return (
      <Panel className="p-5 text-[13px] text-ink-faint">{err ? <span className="text-bad">{err}</span> : 'Loading…'}</Panel>
    )
  }

  const ctx = Object.entries(bug.context ?? {}).filter(([, v]) => v !== null && v !== undefined && v !== '')

  return (
    <Panel className="space-y-4 p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center gap-2 font-mono text-[11px] text-ink-faint">
            <span>#{bug.id}</span>
            <span>·</span>
            <span>{bug.category.replace('_', ' ')}</span>
            <span>·</span>
            <span>{bug.source === 'manual' ? 'filed by hand' : bug.triaged ? 'monitor · triaged' : 'monitor'}</span>
          </div>
          {/* A textarea that grows with its text: a one-line input showed only what fit and
              made long titles look cut off. Enter saves; Shift+Enter is not needed in a title. */}
          <textarea
            key={`t-${bug.id}-${bug.updated_at}`}
            defaultValue={bug.title}
            rows={1}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                e.currentTarget.blur()
              }
            }}
            onBlur={(e) => {
              const v = e.target.value.replace(/\s+/g, ' ').trim()
              if (v && v !== bug.title) patch({ title: v })
            }}
            className="field-sizing-content block w-full resize-none rounded-lg border border-transparent bg-transparent px-1 py-0.5 text-[16px] leading-snug font-medium text-ink outline-none hover:border-seam focus:border-accent"
          />
        </div>
        <span className="flex shrink-0 items-center gap-2">
          <CopyBugButton load={async () => bug} />
          <button
            onClick={onClose}
            className="rounded-lg border border-seam px-2.5 py-1 font-mono text-[11px] text-ink-dim hover:text-ink"
          >
            close
          </button>
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-3 text-[12px] text-ink-dim">
        <label className="flex items-center gap-1.5">
          status
          <select
            className={selectCls}
            value={bug.status}
            disabled={busy}
            onChange={(e) => patch({ status: e.target.value as BugStatus })}
          >
            {STATUSES.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          priority
          <select
            className={selectCls}
            value={bug.priority}
            disabled={busy}
            onChange={(e) => patch({ priority: e.target.value as Priority })}
          >
            {PRIORITIES.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          severity
          <select
            className={selectCls}
            value={bug.severity}
            disabled={busy}
            onChange={(e) => patch({ severity: e.target.value as Severity })}
          >
            {SEVERITIES.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <span className="ml-auto flex gap-2">
          {bug.status !== 'closed' && (
            <Button
              tone="ghost"
              disabled={busy || rechecking}
              onClick={recheck}
              className={rechecking ? 'animate-pulse' : ''}
            >
              {rechecking ? 'Rechecking…' : 'Recheck'}
            </Button>
          )}
          {bug.status !== 'closed' ? (
            <Button tone="ghost" disabled={busy} onClick={() => patch({ status: 'closed' })}>
              Close
            </Button>
          ) : (
            <Button tone="ghost" disabled={busy} onClick={() => patch({ status: 'open' })}>
              Reopen
            </Button>
          )}
          <Button tone="danger" disabled={busy} onClick={remove}>
            Delete
          </Button>
        </span>
      </div>

      {verdict && (
        <div
          className={`rounded-xl border px-3 py-2 text-[12.5px] leading-relaxed ${
            verdict.verdict === 'fixed'
              ? 'border-good/40 bg-good/10 text-good'
              : verdict.verdict === 'not_yet'
                ? 'border-warn/40 bg-warn/10 text-warn'
                : 'border-seam bg-panel-hi text-ink-dim'
          }`}
        >
          <div className="flex items-start gap-2">
            <span className="flex-1">{verdict.message}</span>
            {verdict.needed ? (
              <span className="shrink-0 font-mono text-[11px]">
                {verdict.chances}/{verdict.needed}
              </span>
            ) : null}
          </div>
          {verdict.verdict === 'not_yet' && verdict.needed ? (
            <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-seam">
              <div
                className="h-full rounded-full bg-warn transition-all duration-700"
                style={{ width: `${Math.min(100, ((verdict.chances ?? 0) / verdict.needed) * 100)}%` }}
              />
            </div>
          ) : null}
        </div>
      )}

      <div className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-[11.5px] sm:grid-cols-4">
        {[
          ['seen', `${bug.occurrences}×`],
          ['first', ago(bug.first_seen)],
          ['last', ago(bug.last_seen)],
          ['closed', bug.closed_at ? `${ago(bug.closed_at)}${bug.closed_by ? ` · by ${bug.closed_by}` : ''}` : '—'],
        ].map(([k, v]) => (
          <div key={k}>
            <span className="text-ink-faint">{k} </span>
            <span className="text-ink">{v}</span>
          </div>
        ))}
      </div>

      <Section title="Description">
        <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-ink-dim">{bug.description || '—'}</p>
      </Section>

      {bug.suggestion && (
        <Section title="Suggested fix">
          <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-ink-dim">{bug.suggestion}</p>
        </Section>
      )}

      <Section title="Where">
        <div className="grid grid-cols-1 gap-y-1 font-mono text-[11.5px] sm:grid-cols-2">
          {(
            [
              ['agent', bug.agent],
              ['model', bug.model],
              ['tool', bug.tool],
              ['objective', bug.objective_title],
              ['project', bug.project_id],
              ...ctx.filter(([k]) => !['agent', 'model', 'project_id', 'objective_title'].includes(k)),
            ] as [string, unknown][]
          )
            .filter(([, v]) => v !== null && v !== undefined && v !== '')
            .map(([k, v]) => (
              <div key={k} className="min-w-0 truncate">
                <span className="text-ink-faint">{k} </span>
                <span className="text-ink">{typeof v === 'string' ? v : JSON.stringify(v)}</span>
              </div>
            ))}
        </div>
      </Section>

      <Section title="Script / inputs (latest)">
        <CodeBlock text={bug.script} empty="No script: this was not a code-running tool." />
      </Section>

      <Section title="Evidence (latest)">
        <CodeBlock text={bug.evidence} empty="—" />
      </Section>

      <Section
        title="Notes"
        right={
          notes !== bug.notes && (
            <Button tone="ghost" disabled={busy} onClick={() => patch({ notes })} className="px-3 py-1 text-[12px]">
              Save notes
            </Button>
          )
        }
      >
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={3}
          placeholder="What you found, what was changed…"
          className="w-full rounded-xl border border-seam bg-panel-hi px-3 py-2 text-[12.5px] text-ink outline-none focus:border-accent"
        />
      </Section>

      <Section title={`Sightings (${bug.sightings.length}${bug.occurrences > bug.sightings.length ? ` of ${bug.occurrences}` : ''})`}>
        <ol className="max-h-[260px] space-y-1 overflow-auto">
          {bug.sightings.map((s, i) => (
            <li key={i} className="rounded-lg px-2 py-1 font-mono text-[11px] hover:bg-panel-hi/60">
              <details>
                <summary className="cursor-pointer list-none text-ink-dim">
                  <span className="text-ink-faint">{stamp(s.at)}</span> · {s.agent ?? s.model ?? '—'}
                  {s.record_id && <span className="text-ink-faint"> · record {s.record_id}</span>}
                </summary>
                <pre className="mt-1 max-h-[200px] overflow-auto whitespace-pre-wrap break-all text-ink-faint">
                  {s.evidence || '—'}
                </pre>
              </details>
            </li>
          ))}
        </ol>
      </Section>

      {err && <div className="text-[12px] text-bad">{err}</div>}
    </Panel>
  )
}

function NewBug({ onDone }: { onDone: (id?: number) => void }) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [script, setScript] = useState('')
  const [severity, setSeverity] = useState<Severity>('medium')
  const [priority, setPriority] = useState<Priority>('P3')
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit() {
    setBusy(true)
    try {
      const b = await bugsApi.create({ title, description, script, severity, priority })
      onDone(b.id)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  const field = 'w-full rounded-xl border border-seam bg-panel-hi px-3 py-2 text-[12.5px] text-ink outline-none focus:border-accent'
  return (
    <Panel className="mb-4 space-y-3 p-5">
      <div className="text-[15px] font-medium text-ink">New bug</div>
      <input className={field} placeholder="Title" value={title} onChange={(e) => setTitle(e.target.value)} />
      <textarea
        className={field}
        rows={3}
        placeholder="What happens, and what should happen"
        value={description}
        onChange={(e) => setDescription(e.target.value)}
      />
      <textarea
        className={`${field} font-mono text-[11.5px]`}
        rows={3}
        placeholder="Script or input that shows it (optional)"
        value={script}
        onChange={(e) => setScript(e.target.value)}
      />
      <div className="flex flex-wrap items-center gap-3 text-[12px] text-ink-dim">
        <select className={selectCls} value={priority} onChange={(e) => setPriority(e.target.value as Priority)}>
          {PRIORITIES.map((p) => (
            <option key={p}>{p}</option>
          ))}
        </select>
        <select className={selectCls} value={severity} onChange={(e) => setSeverity(e.target.value as Severity)}>
          {SEVERITIES.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <span className="ml-auto flex gap-2">
          <Button tone="ghost" onClick={() => onDone()}>
            Cancel
          </Button>
          <Button tone="primary" disabled={busy || title.trim().length < 3} onClick={submit}>
            File bug
          </Button>
        </span>
      </div>
      {err && <div className="text-[12px] text-bad">{err}</div>}
    </Panel>
  )
}

export default function BugsPage() {
  const [status, setStatus] = useState<BugStatus | 'all'>('open')
  const [q, setQ] = useState('')
  const [selected, setSelected] = useState<number | null>(null)
  const [creating, setCreating] = useState(false)
  const [scanMsg, setScanMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const list = usePoll(() => bugsApi.list(status, q.trim()), REFRESH_MS)
  const mon = usePoll<MonitorDoc>(bugsApi.monitor, REFRESH_MS)
  const { refresh } = list

  useEffect(() => {
    const t = setTimeout(refresh, 250) // debounce typing
    return () => clearTimeout(t)
  }, [status, q, refresh])

  const counts: BugCounts = list.data?.counts ?? { open: 0, pending: 0, closed: 0 }
  const bugs: BugSummary[] = list.data?.bugs ?? []
  const cfg = mon.data?.config
  const state = mon.data?.state

  async function toggleMonitor() {
    if (!cfg) return
    setBusy(true)
    try {
      await bugsApi.configure({ enabled: !cfg.enabled })
      mon.refresh()
    } finally {
      setBusy(false)
    }
  }

  async function scanNow() {
    setBusy(true)
    setScanMsg(null)
    try {
      const r = await bugsApi.scan()
      setScanMsg(
        `${r.findings} findings · ${r.new_bugs} new bug${r.new_bugs === 1 ? '' : 's'}` +
          (r.reopened ? ` · ${r.reopened} reopened` : '') +
          (r.closed_fixed ? ` · ${r.closed_fixed} closed as fixed` : ''),
      )
      list.refresh()
      mon.refresh()
    } catch (e) {
      setScanMsg(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const tabs: { key: BugStatus | 'all'; label: string; n?: number }[] = [
    { key: 'open', label: 'Open', n: counts.open },
    { key: 'pending', label: 'Pending', n: counts.pending },
    { key: 'closed', label: 'Closed', n: counts.closed },
    { key: 'all', label: 'All' },
  ]

  return (
    <div className="mx-auto max-w-[1440px] px-8 py-8">
      <PageHeader
        title="Bugs"
        subtitle={
          <>
            Filed by the monitoring agent from the agents&apos; logs and the engines — and by hand.{' '}
            <Link href="/settings" className="text-accent hover:underline">
              Monitor settings
            </Link>
          </>
        }
        right={
          <div className="flex flex-wrap items-center justify-end gap-2">
            <Pill tone={cfg?.enabled ? 'good' : 'neutral'} pulse={!!cfg?.enabled}>
              {cfg?.enabled
                ? `Monitoring · ${state?.last_scan ? `scanned ${ago(state.last_scan)}` : 'first scan within a minute'}` +
                  (state?.triage_running ? ` · ${state.model ?? 'model'} writing up` : '')
                : 'Monitor off'}
            </Pill>
            <Button tone="ghost" disabled={busy || !cfg} onClick={toggleMonitor}>
              {cfg?.enabled ? 'Turn off' : 'Turn on'}
            </Button>
            <Button tone="ghost" disabled={busy} onClick={scanNow}>
              Scan now
            </Button>
            <Button
              tone="ghost"
              onClick={() => window.open('/bugs/report?status=all&print=1', '_blank', 'noopener')}
            >
              Export PDF
            </Button>
            <Button onClick={() => setCreating(true)}>New bug</Button>
          </div>
        }
      />

      {(scanMsg || state?.last_error || (cfg?.enabled && cfg.llm_triage && state?.llm_note)) && (
        <div className="-mt-4 mb-4 space-y-1 font-mono text-[11.5px]">
          {scanMsg && <div className="text-ink-dim">scan: {scanMsg}</div>}
          {state?.last_error && <div className="text-bad">last scan failed: {state.last_error}</div>}
          {cfg?.enabled && cfg.llm_triage && state?.llm_note && <div className="text-warn">triage: {state.llm_note}</div>}
        </div>
      )}

      {creating && (
        <NewBug
          onDone={(id) => {
            setCreating(false)
            if (id) {
              setStatus('open')
              setSelected(id)
              list.refresh()
            }
          }}
        />
      )}

      <Panel className="mb-3 flex flex-wrap items-center gap-2 px-4 py-2.5">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setStatus(t.key)}
            className={`rounded-lg px-3 py-1.5 text-[12.5px] transition-colors ${
              status === t.key ? 'bg-panel-hi text-ink shadow-[inset_0_0_0_1px_var(--color-seam)]' : 'text-ink-dim hover:text-ink'
            }`}
          >
            {t.label}
            {t.n !== undefined && <span className="ml-1.5 font-mono text-[11px] text-ink-faint">{t.n}</span>}
          </button>
        ))}
        <input
          className="ml-auto min-w-[220px] rounded-lg border border-seam bg-panel-hi px-3 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent"
          placeholder="Search title, evidence, model, tool…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      </Panel>

      {list.error && <div className="mb-3 text-[12px] text-bad">{list.error}</div>}

      <div className={`grid gap-4 ${selected !== null ? 'xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]' : ''}`}>
        <div className="min-w-0">
          {bugs.length === 0 && !list.loading ? (
            <EmptyState
              title={status === 'open' ? 'No open bugs' : `No ${status === 'all' ? '' : status + ' '}bugs`}
              hint={
                cfg?.enabled
                  ? 'The monitor scans every minute; problems it finds show up here.'
                  : 'The monitor is off. Turn it on, or run a scan by hand.'
              }
            />
          ) : (
            <Panel className="overflow-hidden">
              <div className="flex items-center border-b border-seam pr-3">
                <div className="grid flex-1 grid-cols-[48px_76px_minmax(0,1fr)_56px_84px_72px] gap-3 px-4 py-2 font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-faint">
                  <span>pri</span>
                  <span>severity</span>
                  <span>bug</span>
                  <span className="text-right">seen</span>
                  <span className="text-right">last</span>
                  <span>status</span>
                </div>
                <span className="w-7" />
              </div>
              <ol>
                {bugs.map((b) => (
                  <li
                    key={b.id}
                    className={`flex items-center border-b border-seam/50 pr-3 transition-colors last:border-0 ${
                      b.id === selected ? 'bg-panel-hi' : 'hover:bg-panel-hi/50'
                    }`}
                  >
                    <button
                      onClick={() => setSelected(b.id === selected ? null : b.id)}
                      className="grid min-w-0 flex-1 grid-cols-[48px_76px_minmax(0,1fr)_56px_84px_72px] items-center gap-3 px-4 py-2.5 text-left"
                    >
                      <span>
                        <PriorityChip p={b.priority} />
                      </span>
                      <span
                        className={`font-mono text-[11.5px] ${
                          SEV_TONE[b.severity] === 'bad'
                            ? 'text-bad'
                            : SEV_TONE[b.severity] === 'warn'
                              ? 'text-warn'
                              : 'text-ink-faint'
                        }`}
                      >
                        {b.severity}
                      </span>
                      <span className="min-w-0">
                        <span className="line-clamp-2 break-words text-[13px] leading-snug text-ink" title={b.title}>
                          {b.title}
                        </span>
                        <span className="block truncate font-mono text-[10.5px] text-ink-faint">
                          #{b.id} · {b.category.replace('_', ' ')}
                          {b.model && ` · ${b.model}`}
                          {b.tool && ` · ${b.tool}`}
                          {b.objective_title && ` · ${b.objective_title}`}
                        </span>
                      </span>
                      <span className="text-right font-mono text-[12px] text-ink-dim">{b.occurrences}×</span>
                      <span className="text-right font-mono text-[11px] text-ink-faint">{ago(b.last_seen)}</span>
                      <span>
                        <span
                          className={`font-mono text-[11px] ${
                            STATUS_TONE[b.status] === 'bad'
                              ? 'text-bad'
                              : STATUS_TONE[b.status] === 'warn'
                                ? 'text-warn'
                                : 'text-good'
                          }`}
                        >
                          {b.status}
                        </span>
                        {b.status === 'closed' && b.closed_by === 'monitor' && (
                          <span className="block font-mono text-[9.5px] text-ink-faint" title="The monitor saw it fixed">
                            auto
                          </span>
                        )}
                      </span>
                    </button>
                    <CopyBugButton size="sm" load={() => bugsApi.get(b.id)} />
                  </li>
                ))}
              </ol>
            </Panel>
          )}
        </div>

        {selected !== null && (
          <div className="min-w-0">
            <BugDetail
              id={selected}
              onChanged={() => list.refresh()}
              onClose={() => setSelected(null)}
            />
          </div>
        )}
      </div>
    </div>
  )
}
