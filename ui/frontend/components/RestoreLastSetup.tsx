'use client'

import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { api, type ConsoleDoc, type LastSetupDoc, type RestoreItemStatus } from '@/lib/api'
import { bytesLabel, duration } from '@/lib/format'
import { Button, Panel, Pill } from '@/components/ui'

const IDLE_POLL_MS = 10_000
const JOB_POLL_MS = 1_500
/** A finished job's report stays up this long (or until dismissed); the job lives on the server. */
const REPORT_KEEP_S = 15 * 60

const STATUS_TONE: Record<RestoreItemStatus, 'good' | 'warn' | 'bad' | 'accent' | 'neutral'> = {
  queued: 'neutral',
  loading: 'warn',
  ready: 'good',
  failed: 'bad',
  skipped: 'neutral',
}

const kindLabel = (kind: 'llm' | 'ts') => (kind === 'llm' ? 'LLM' : 'forecaster')

/**
 * "Start last setup": bring back the models that were running before a crash or restart.
 *
 * The control plane remembers every model the user launched (and forgets the ones they
 * unloaded) in last_setup.json. This shows what it would start, in the order it would start
 * them -- LLMs one at a time, largest first, then forecasters -- then follows the job.
 *
 * Returns the button and the panel separately so each page places them, like useUnloadAll.
 */
export function useRestoreLastSetup(data: ConsoleDoc | null | undefined, onChanged: () => void) {
  const [doc, setDoc] = useState<LastSetupDoc | null>(null)
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [dismissedJob, setDismissedJob] = useState<string | null>(null)
  const [now, setNow] = useState(() => Date.now())

  const job = doc?.job ?? null
  const running = job?.state === 'running'

  const load = useCallback(async () => {
    try {
      setDoc(await api.lastSetup())
    } catch {
      /* the preview is advisory; a failed poll keeps the last one */
    }
  }, [])

  // Poll: slowly while idle (keeps the disabled state honest), quickly while a job runs.
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    async function tick() {
      if (running) {
        try {
          const r = await api.restoreJob()
          if (!cancelled) {
            setDoc((d) => (d ? { ...d, job: r.job } : d))
            if (r.job?.state !== 'running') {
              await load()
              onChanged()
            }
          }
        } catch {
          /* keep polling */
        }
      } else {
        await load()
      }
      if (!cancelled) timer = setTimeout(tick, running ? JOB_POLL_MS : IDLE_POLL_MS)
    }
    tick()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
    // onChanged is a page refresh callback; re-subscribing on its identity is not wanted.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, load])

  // A ticking clock for the "loading for 3m" label.
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [running])

  async function start() {
    setBusy(true)
    setErr(null)
    try {
      const r = await api.restoreLastSetup()
      setDoc((d) => (d ? { ...d, job: r.job } : { entries: [], to_start: 0, updated_at: null, job: r.job }))
      setDismissedJob(null)
      setOpen(false)
      onChanged()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function forget(kind: 'llm' | 'ts', model: string) {
    try {
      await api.forgetLastSetup(kind, model)
      await load()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const entries = doc?.entries ?? []
  const toStart = doc?.to_start ?? 0
  const disabledReason = running
    ? 'The last setup is being started'
    : !doc
      ? 'Reading the last setup…'
      : entries.length === 0
        ? 'Nothing remembered yet: models you load are remembered here, and brought back by this button after a restart or crash.'
        : toStart === 0
          ? 'Everything in the last setup is already running (or cannot be started — open the list to see why).'
          : null
  const title =
    disabledReason ??
    `Start ${toStart} model${toStart === 1 ? '' : 's'} from the last setup: ` +
      entries
        .filter((e) => e.action === 'start')
        .map((e) => e.model)
        .join(', ')

  const button = (
    <span title={title} className="inline-flex">
      <Button
        tone="ghost"
        onClick={() => {
          setErr(null)
          setOpen((o) => !o)
          load()
        }}
        // Openable whenever something is remembered, so an all-running or all-invalid list
        // can still be inspected (and pruned); starting is gated inside the panel.
        disabled={running || !doc || entries.length === 0}
      >
        {running
          ? 'Starting last setup…'
          : toStart > 0
            ? `Start last setup (${toStart})`
            : 'Start last setup'}
      </Button>
    </span>
  )

  // Loading progress of an LLM, from the console's own per-engine load estimate.
  function loadPct(instanceId: string | null): string | null {
    if (!instanceId) return null
    const eng = data?.engines?.find((e) => e.instance_id === instanceId)
    return eng?.load ? `${eng.load.label} · ${eng.load.overall_pct.toFixed(0)}%` : null
  }

  const recent = job?.finished_at != null && now / 1000 - job.finished_at < REPORT_KEEP_S
  const showJob = job != null && (running || (recent && dismissedJob !== job.id)) && !open
  let panel: ReactNode = null

  if (open) {
    panel = (
      <Panel className="mb-6 p-4 text-[12.5px] text-ink-dim">
        <div className="flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <div className="text-[14px] font-medium text-ink">Start last setup</div>
            <div className="mt-1 text-[12px] text-ink-faint">
              LLMs load one at a time, largest first, each waiting for the previous to be ready;
              forecasters go last. Models already running are skipped, and a failure does not
              stop the rest.
              {doc?.updated_at ? ` Remembered ${duration(Date.now() / 1000 - doc.updated_at)} ago.` : ''}
            </div>
            <ol className="mt-3 space-y-1.5">
              {entries.map((e, i) => (
                <li key={`${e.kind}:${e.model}`} className="flex flex-wrap items-center gap-2">
                  <span className="w-5 text-right font-mono text-[11px] text-ink-faint">{i + 1}.</span>
                  <span className={e.action === 'start' ? 'text-ink' : 'text-ink-faint'}>{e.model}</span>
                  <span className="font-mono text-[11px] text-ink-faint">
                    {kindLabel(e.kind)} · GPU {e.launch_gpus ?? 'auto'}
                    {e.size_bytes ? ` · ${bytesLabel(e.size_bytes)}` : ''}
                    {e.kind === 'llm' && e.options?.moe_backend ? ` · ${String(e.options.moe_backend)}` : ''}
                    {e.kind === 'llm' && e.options?.num_tokens ? ` · ctx ${Number(e.options.num_tokens).toLocaleString()}` : ''}
                  </span>
                  {e.action === 'running' && <Pill tone="good">already running</Pill>}
                  {e.action === 'invalid' && <Pill tone="bad">cannot start</Pill>}
                  {e.reason && e.action !== 'running' && (
                    <span className={e.action === 'invalid' ? 'text-bad' : 'text-warn'}>{e.reason}</span>
                  )}
                  <button
                    onClick={() => forget(e.kind, e.model)}
                    title="Forget this model: it will not be started by this button"
                    className="ml-auto font-mono text-[11px] text-ink-faint hover:text-bad"
                  >
                    forget
                  </button>
                </li>
              ))}
            </ol>
            {err && <div className="mt-3 text-bad">{err}</div>}
            <div className="mt-4 flex gap-2">
              <Button tone="primary" onClick={start} disabled={busy || toStart === 0}>
                {busy
                  ? 'Starting…'
                  : toStart === 0
                    ? 'Nothing to start'
                    : `Start ${toStart} model${toStart === 1 ? '' : 's'}`}
              </Button>
              <Button tone="ghost" onClick={() => setOpen(false)}>
                Cancel
              </Button>
            </div>
          </div>
        </div>
      </Panel>
    )
  } else if (showJob && job) {
    const done = job.items.filter((i) => i.status === 'ready').length
    const failed = job.items.filter((i) => i.status === 'failed').length
    const toLoad = job.items.filter((i) => i.status !== 'skipped').length
    panel = (
      <Panel className="mb-6 p-4 text-[12.5px] text-ink-dim">
        <div className="flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <div className="text-ink">
              {running
                ? `Starting last setup: ${done} of ${toLoad} ready${failed ? `, ${failed} failed` : ''}…`
                : `Last setup started: ${done} of ${toLoad} ready${failed ? `, ${failed} failed` : ''}.`}
            </div>
            <ol className="mt-2 space-y-1.5">
              {job.items.map((it) => {
                const pct = it.status === 'loading' ? loadPct(it.instance_id) : null
                const elapsed =
                  it.status === 'loading' && it.started_at ? duration(now / 1000 - it.started_at) : null
                return (
                  <li key={`${it.kind}:${it.model}`} className="flex flex-wrap items-center gap-2">
                    <Pill tone={STATUS_TONE[it.status]} pulse={it.status === 'loading'}>
                      {it.status}
                    </Pill>
                    <span className="text-ink">{it.model}</span>
                    <span className="font-mono text-[11px] text-ink-faint">
                      {kindLabel(it.kind)} · GPU {it.gpus ?? 'auto'}
                      {pct ? ` · ${pct}` : ''}
                      {elapsed ? ` · ${elapsed}` : ''}
                    </span>
                    {it.status === 'skipped' && <span className="text-ink-faint">already running</span>}
                    {it.note && it.status !== 'skipped' && <span className="text-warn">{it.note}</span>}
                    {it.error && <span className="basis-full pl-1 font-mono text-[11.5px] text-bad">{it.error}</span>}
                  </li>
                )
              })}
            </ol>
          </div>
          {!running && (
            <button
              onClick={() => setDismissedJob(job.id)}
              className="shrink-0 font-mono text-[11px] text-ink-faint hover:text-ink"
            >
              dismiss
            </button>
          )}
        </div>
      </Panel>
    )
  }

  return { button, panel }
}
