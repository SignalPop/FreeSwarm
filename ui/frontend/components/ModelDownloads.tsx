'use client'

import { useEffect, useState } from 'react'
import { bytesLabel, duration } from '@/lib/format'
import { Button, Panel, Pill } from '@/components/ui'
import { RatingChips, type RatingSort, SortByRating, sortByRating, useRatings } from '@/lib/ratings'

type Job = {
  id: string
  key: string | null
  name: string
  repo: string
  phase: 'queued' | 'starting' | 'downloading' | 'verifying' | 'done' | 'error' | 'cancelled'
  bytes: number
  files: number
  done_bytes?: number
  rate_bytes_s?: number
  eta_s?: number | null
  verify_done?: number
  verify_files?: number
  error: string | null
}

type Plan = {
  repo: string
  revision: string
  dest: string
  target: string
  files: number
  bytes: number
  present_bytes: number
  needed_bytes: number
  free_bytes: number
  drive: string
  fits: boolean
  code_files: string[]
  code_note: string | null
  excluded_unsafe: string[]
  excluded_unsafe_count: number
  skipped_bytes: number
  installed: boolean
  complete: boolean
  missing_extras: string[]
}

type Known = Partial<Plan> & {
  key: string
  name: string
  repo: string
  revision: string
  role: string
  state: 'installed' | 'partial' | 'missing' | 'unknown'
  offline?: boolean
  error?: string
  job?: Job
}

type Doc = { known: Known[]; jobs: Job[]; token_set: boolean; models_dir: string; hub_dir: string }

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) } })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const b = await res.json()
      if (b?.detail) detail = typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail)
    } catch {
      /* non-JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

const ACTIVE = new Set(['queued', 'starting', 'downloading', 'verifying'])

/** Live progress of one download. */
function JobBar({ job, onCancel }: { job: Job; onCancel: () => void }) {
  const pct = job.phase === 'verifying'
    ? ((job.verify_done ?? 0) / Math.max(1, job.verify_files ?? 1)) * 100
    : ((job.done_bytes ?? 0) / Math.max(1, job.bytes)) * 100
  return (
    <div className="mt-1.5">
      <div className="flex items-center gap-2 font-mono text-[10.5px] text-ink-dim">
        <span className={job.phase === 'error' ? 'text-bad' : ''}>{job.phase}</span>
        {job.phase === 'downloading' && (
          <span>
            {bytesLabel(job.done_bytes ?? 0)} / {bytesLabel(job.bytes)}
            {job.rate_bytes_s ? ` · ${bytesLabel(job.rate_bytes_s)}/s` : ''}
            {job.eta_s ? ` · ${duration(job.eta_s)} left` : ''}
          </span>
        )}
        {job.phase === 'verifying' && (
          <span>
            checksums {job.verify_done ?? 0}/{job.verify_files ?? '?'}
          </span>
        )}
        {ACTIVE.has(job.phase) && (
          <button onClick={onCancel} className="ml-auto text-ink-faint hover:text-bad">
            cancel
          </button>
        )}
      </div>
      {ACTIVE.has(job.phase) && (
        <div className="mt-1 h-1.5 overflow-hidden rounded bg-seam">
          <div className={`h-full ${job.phase === 'verifying' ? 'bg-good' : 'bg-accent'} transition-all`} style={{ width: `${Math.min(100, pct)}%` }} />
        </div>
      )}
      {job.error && <div className="mt-0.5 text-[11.5px] text-bad">{job.error}</div>}
    </div>
  )
}

type TokenStatus = { token_set: boolean; token_user: string | null; token_invalid?: boolean }

/** What is wrong with a value typed into the token box, before it is sent anywhere. */
function tokenProblem(t: string): string | null {
  if (!t) return null
  if (t.includes('@')) return "That's an email address. Hugging Face doesn't let apps sign in with your email and password. Paste an access token instead (see the steps above)."
  if (!t.startsWith('hf_')) return "That isn't an access token. Tokens start with hf_. Your Hugging Face password won't work here."
  return null
}

/**
 * Connect a Hugging Face account, needed only for gated models (e.g. Meta's Llama). Hugging Face
 * has no password login for apps, so this walks through creating an access token and reports,
 * right here, whether it is checking, connected (and as whom), or what went wrong.
 */
function HfAccount({ onChange }: { onChange: () => void }) {
  const [status, setStatus] = useState<TokenStatus | null>(null)
  const [token, setToken] = useState('')
  const [state, setState] = useState<'idle' | 'checking' | 'ok' | 'error'>('idle')
  const [msg, setMsg] = useState<string | null>(null)

  async function refresh() {
    try {
      setStatus(await req<TokenStatus>('/api/downloads/token'))
    } catch {
      /* the panel's own error line covers an unreachable backend */
    }
  }
  useEffect(() => { void refresh() }, [])

  const problem = tokenProblem(token.trim())

  async function connect() {
    setState('checking')
    setMsg(null)
    try {
      const r = await req<TokenStatus & { verified: boolean }>('/api/downloads/token', { method: 'PUT', body: JSON.stringify({ token: token.trim() }) })
      setToken('')
      setState('ok')
      setMsg(r.verified ? `Connected as ${r.token_user}.` : "Saved, but Hugging Face couldn't be reached to check it. It will be used on the next download.")
      await refresh()
      onChange()
    } catch (e) {
      setState('error')
      setMsg(e instanceof Error ? e.message : String(e))
    }
  }

  async function disconnect() {
    setState('checking')
    setMsg(null)
    try {
      await req('/api/downloads/token', { method: 'PUT', body: JSON.stringify({ token: '' }) })
      setState('idle')
      await refresh()
      onChange()
    } catch (e) {
      setState('error')
      setMsg(e instanceof Error ? e.message : String(e))
    }
  }

  const connected = status?.token_set && !status.token_invalid
  return (
    <div className="rounded-xl border border-seam p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[12.5px] font-medium text-ink">Hugging Face account</span>
        {status == null ? null : status.token_invalid ? (
          <Pill tone="bad">token no longer works</Pill>
        ) : connected ? (
          <Pill tone="good">{status.token_user ? `connected as ${status.token_user}` : 'connected'}</Pill>
        ) : (
          <Pill tone="neutral">not connected</Pill>
        )}
        <span className="text-[11.5px] text-ink-faint">
          Only needed for gated models (Hugging Face marks them &quot;gated&quot;; Meta&apos;s Llama repos are an example). Everything else downloads without it.
        </span>
      </div>

      {connected ? (
        <div className="mt-2 flex items-center gap-3 text-[12px] text-ink-dim">
          <span>Your token is saved on this computer.</span>
          <button onClick={disconnect} disabled={state === 'checking'} className="font-mono text-[11px] text-ink-faint hover:text-bad">
            disconnect
          </button>
        </div>
      ) : (
        <>
          <ol className="mt-2 list-decimal space-y-0.5 pl-5 text-[12px] text-ink-dim">
            <li>
              Hugging Face doesn&apos;t let apps sign in with your email and password, so you need an access token.
              Open{' '}
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="text-accent underline">
                huggingface.co/settings/tokens
              </a>{' '}
              (sign in there if asked).
            </li>
            <li>Click <b>Create new token</b>, choose the <b>Read</b> type, give it any name and create it.</li>
            <li>Copy the token (it starts with <span className="font-mono">hf_</span>), paste it below and click Connect.</li>
          </ol>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <input type="password" value={token} placeholder="hf_…" autoComplete="off"
              onChange={(e) => { setToken(e.target.value); setState('idle'); setMsg(null) }}
              onKeyDown={(e) => { if (e.key === 'Enter' && token.trim() && !problem) void connect() }}
              className={`w-[320px] rounded-lg border bg-panel-hi px-2.5 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent ${problem ? 'border-bad' : 'border-seam'}`} />
            <Button tone="primary" disabled={state === 'checking' || !token.trim() || !!problem} onClick={connect}>
              {state === 'checking' ? 'Checking…' : 'Connect'}
            </Button>
            <span className="text-[11px] text-ink-faint">Stored on this computer only, never shown again.</span>
          </div>
        </>
      )}
      {problem && <div className="mt-1.5 text-[12px] text-bad">{problem}</div>}
      {state === 'checking' && <div className="mt-1.5 text-[12px] text-ink-dim">Checking the token with Hugging Face…</div>}
      {msg && <div className={`mt-1.5 text-[12px] ${state === 'error' ? 'text-bad' : 'text-good'}`}>{state === 'ok' ? '✓ ' : '✗ '}{msg}</div>}
    </div>
  )
}

/** What is wrong with a value typed into the model box, or null when it looks like a model ID. */
function repoProblem(r: string): string | null {
  if (!r) return null
  if (r.includes('@')) return "That's an email address. This box takes a model ID. To sign in to Hugging Face, use the account box above."
  if (/^https?:\/\//.test(r)) return null // a pasted model URL is converted below
  if (!/^[A-Za-z0-9][\w.-]*\/[\w.-]+$/.test(r)) return 'Enter a model ID as owner/model-name, e.g. google/gemma-4-26B-A4B-it. You can copy it from the top of the model page on huggingface.co.'
  return null
}

/** Accept a pasted huggingface.co URL as well as a bare owner/model ID. */
function normalizeRepo(r: string): string {
  const m = r.match(/^https?:\/\/(?:www\.)?huggingface\.co\/([^/\s]+\/[^/\s?#]+)/)
  return m ? m[1] : r
}

/**
 * Download the models this install uses -- pinned to the exact revisions in use -- or any
 * Hugging Face model. Shows what will be fetched (and what is skipped, and any code that
 * would run) before starting; checks disk space; verifies checksums; resumes after cancel.
 */
export default function ModelDownloads({ onInstalled }: { onInstalled?: () => void }) {
  const [doc, setDoc] = useState<Doc | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const [repo, setRepo] = useState('')
  const [dest, setDest] = useState<'hf-cache' | 'models'>('hf-cache')
  const [includeCode, setIncludeCode] = useState(false)
  const [plan, setPlan] = useState<Plan | null>(null)
  const [planErr, setPlanErr] = useState<string | null>(null)
  const [sortBy, setSortBy] = useState<RatingSort>('default')
  const ratingFor = useRatings()
  const [checking, setChecking] = useState(false)
  const [busy, setBusy] = useState(false)
  const [doneSeen, setDoneSeen] = useState<Set<string>>(new Set())

  async function load() {
    try {
      const d = await req<Doc>('/api/downloads')
      setDoc(d)
      setErr(null)
      // A download that just finished: let the page re-read the model list.
      const finished = d.jobs.filter((j) => j.phase === 'done' && !doneSeen.has(j.id))
      if (finished.length) {
        setDoneSeen((s) => new Set([...s, ...finished.map((j) => j.id)]))
        onInstalled?.()
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }
  useEffect(() => {
    void load()
    const active = doc?.jobs.some((j) => ACTIVE.has(j.phase))
    const t = setInterval(load, active ? 1500 : 10_000)
    return () => clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc?.jobs.some((j) => ACTIVE.has(j.phase))])

  async function act(fn: () => Promise<unknown>) {
    setBusy(true)
    setErr(null)
    try {
      await fn()
      await load()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const missing = (doc?.known ?? []).filter((k) => k.state !== 'installed')
  const activeJobs = (doc?.jobs ?? []).filter((j) => ACTIVE.has(j.phase))
  const customJobs = (doc?.jobs ?? []).filter((j) => !j.key)
  const body = { repo: normalizeRepo(repo.trim()), dest, include_code: includeCode }
  const badRepo = repoProblem(repo.trim())

  async function check() {
    setChecking(true)
    setPlanErr(null)
    try {
      setPlan(await req<Plan>('/api/downloads/plan', { method: 'POST', body: JSON.stringify(body) }))
    } catch (e) {
      setPlanErr(e instanceof Error ? e.message : String(e))
    } finally {
      setChecking(false)
    }
  }

  return (
    <Panel className="mb-6 p-4">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 text-left">
        <span className="text-[14px] font-medium text-ink">Download models</span>
        {doc && (
          <span className="font-mono text-[11px] text-ink-faint">
            {doc.known.length - missing.length}/{doc.known.length} of the models in use installed
            {activeJobs.length ? ` · ${activeJobs.length} downloading` : ''}
          </span>
        )}
        {missing.length > 0 && <Pill tone="warn">{missing.length} missing</Pill>}
        <span className="ml-auto font-mono text-[11px] text-accent">{open ? 'hide' : 'show'}</span>
      </button>
      {err && <div className="mt-2 text-[12px] text-bad">{err}</div>}

      {open && doc && (
        <div className="mt-3 space-y-4">
          <div className="space-y-2">
            <div className="flex justify-end"><SortByRating value={sortBy} onChange={setSortBy} /></div>
            {sortByRating(doc.known, sortBy, (k) => ratingFor(k.name, k.repo)).map((k) => {
              const job = k.job
              const running = job && ACTIVE.has(job.phase)
              return (
                <div key={k.key} className="rounded-xl border border-seam bg-panel-hi/40 p-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-[12.5px] text-ink">{k.name}</span>
                    <Pill tone={k.state === 'installed' ? 'good' : k.state === 'partial' ? 'warn' : k.state === 'missing' ? 'bad' : 'neutral'}>
                      {k.state}
                    </Pill>
                    <RatingChips rating={ratingFor(k.name, k.repo)} />
                    <span className="text-[11.5px] text-ink-faint">{k.role}</span>
                    <span className="ml-auto flex items-center gap-2">
                      {k.bytes != null && <span className="font-mono text-[11px] text-ink-dim">{bytesLabel(k.bytes)}</span>}
                      {k.state !== 'installed' && !running && (
                        <Button tone="primary" disabled={busy || k.fits === false} onClick={() => act(() => req(`/api/downloads/known/${k.key}`, { method: 'POST' }))}>
                          {k.state === 'partial' ? 'Resume' : 'Download'}
                        </Button>
                      )}
                    </span>
                  </div>
                  <div className="mt-0.5 font-mono text-[10.5px] text-ink-faint">
                    {k.repo} @ {k.revision.slice(0, 10)} → {k.target ?? (k.dest === 'models' ? doc.models_dir : doc.hub_dir)}
                    {k.skipped_bytes ? ` · skips ${bytesLabel(k.skipped_bytes)} of duplicate formats` : ''}
                  </div>
                  {k.state !== 'installed' && k.fits === false && (
                    <div className="mt-1 text-[11.5px] text-bad">
                      Not enough space on {k.drive}: needs {bytesLabel(k.needed_bytes ?? 0)}, {bytesLabel(k.free_bytes ?? 0)} free.
                    </div>
                  )}
                  {k.state !== 'installed' && (k.code_files?.length ?? 0) > 0 && (
                    <div className="mt-1 text-[11.5px] text-warn">
                      Includes Python that runs on this computer: {k.code_files!.join(', ')}
                      {k.code_note ? ` — ${k.code_note}` : ''}. Pinned to the reviewed revision above.
                    </div>
                  )}
                  {k.offline && <div className="mt-1 text-[11px] text-ink-faint">Hugging Face unreachable — showing what is on disk.</div>}
                  {job && <JobBar job={job} onCancel={() => act(() => req(`/api/downloads/jobs/${job.id}/cancel`, { method: 'POST' }))} />}
                </div>
              )
            })}
          </div>

          <HfAccount onChange={load} />

          {/* ---- Any Hugging Face model ---- */}
          <div className="rounded-xl border border-seam p-3">
            <div className="mb-2 text-[12.5px] font-medium text-ink">Another model from Hugging Face <span className="font-normal text-ink-faint">— paste its model ID or its huggingface.co link</span></div>
            <div className="flex flex-wrap items-center gap-2">
              <input
                value={repo}
                onChange={(e) => {
                  setRepo(e.target.value)
                  setPlan(null)
                  setPlanErr(null)
                }}
                placeholder="owner/model-name   e.g. google/gemma-4-26B-A4B-it"
                className="min-w-[260px] flex-1 rounded-lg border border-seam bg-panel-hi px-2.5 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent"
              />
              <select value={dest} onChange={(e) => { setDest(e.target.value as 'hf-cache' | 'models'); setPlan(null) }}
                className="rounded-lg border border-seam bg-panel-hi px-2 py-1.5 font-mono text-[11.5px] text-ink">
                <option value="hf-cache">Hugging Face cache</option>
                <option value="models">models folder</option>
              </select>
              <Button tone="primary" disabled={busy || !repo.trim() || !!badRepo} onClick={check}>
                {checking ? 'Checking…' : 'Check'}
              </Button>
            </div>
            {badRepo && <div className="mt-1.5 text-[12px] text-bad">{badRepo}</div>}
            {checking && <div className="mt-1.5 text-[12px] text-ink-dim">Looking up {body.repo} on Hugging Face…</div>}
            {planErr && <div className="mt-1.5 text-[12px] text-bad">✗ {planErr}</div>}
            {plan && (
              <div className="mt-2 space-y-1 text-[12px] text-ink-dim">
                <div className="font-mono text-[11px]">
                  {plan.files} files · {bytesLabel(plan.bytes)} · revision {plan.revision.slice(0, 10)} → {plan.target}
                  {plan.skipped_bytes ? ` · skipping ${bytesLabel(plan.skipped_bytes)} (other formats, subfolders)` : ''}
                </div>
                {plan.excluded_unsafe_count > 0 && (
                  <div className="text-[11.5px] text-ink-faint">
                    Not downloaded, can run code when loaded: {plan.excluded_unsafe.slice(0, 6).join(', ')}
                    {plan.excluded_unsafe_count > 6 ? ` +${plan.excluded_unsafe_count - 6} more` : ''}
                  </div>
                )}
                {plan.code_files.length > 0 && (
                  <div className="text-[11.5px] text-warn">Python files that would run on this computer: {plan.code_files.join(', ')}</div>
                )}
                <label className="flex items-center gap-2 text-[11.5px]">
                  <input type="checkbox" checked={includeCode} onChange={(e) => { setIncludeCode(e.target.checked); setPlan(null) }} />
                  Include the repo&apos;s Python code (only if you trust it — FreeToken executes it when loading the model)
                </label>
                {!plan.fits && (
                  <div className="text-bad">Not enough space on {plan.drive}: needs {bytesLabel(plan.needed_bytes)}, {bytesLabel(plan.free_bytes)} free.</div>
                )}
                {plan.bytes === 0 && <div className="text-warn">Nothing loadable here: no safetensors weights at the top level of this repo.</div>}
                <Button tone="primary" disabled={busy || !plan.fits || plan.installed || plan.bytes === 0}
                  onClick={() => act(async () => { await req('/api/downloads/custom', { method: 'POST', body: JSON.stringify(body) }); setPlan(null) })}>
                  {plan.installed ? 'Already downloaded' : `Download ${bytesLabel(plan.needed_bytes)}`}
                </Button>
              </div>
            )}
            {customJobs.map((j) => (
              <div key={j.id} className="mt-2 border-t border-seam/60 pt-2">
                <div className="font-mono text-[11.5px] text-ink">{j.repo}</div>
                <JobBar job={j} onCancel={() => act(() => req(`/api/downloads/jobs/${j.id}/cancel`, { method: 'POST' }))} />
              </div>
            ))}
          </div>

        </div>
      )}
    </Panel>
  )
}
