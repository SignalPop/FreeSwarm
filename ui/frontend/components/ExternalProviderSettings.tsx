'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { external, type Escalation, type Overview, usd } from '@/lib/external'
import { Panel, Pill } from '@/components/ui'

/**
 * External models: the Groq and OpenRouter API keys, the daily spending limit, and when a
 * stuck search asks stronger models for ideas. Keys are write-only -- stored with the other
 * secrets on the backend, never sent back to this page.
 */
export default function ExternalProviderSettings() {
  const [doc, setDoc] = useState<Overview | null>(null)
  const [keys, setKeys] = useState<Record<string, string>>({})
  const [limit, setLimit] = useState('')
  const [esc, setEsc] = useState<Escalation | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [msg, setMsg] = useState<Record<string, { ok: boolean; text: string }>>({})

  function take(d: Overview) {
    setDoc(d)
    setLimit(String(d.daily_limit_usd))
    setEsc(d.escalation)
  }
  useEffect(() => {
    external.overview().then(take).catch((e) => setMsg({ _: { ok: false, text: String(e instanceof Error ? e.message : e) } }))
  }, [])

  async function run(tag: string, fn: () => Promise<string | void>) {
    setBusy(tag)
    setMsg((m) => ({ ...m, [tag]: undefined as never }))
    try {
      const text = await fn()
      if (text) setMsg((m) => ({ ...m, [tag]: { ok: true, text } }))
    } catch (e) {
      setMsg((m) => ({ ...m, [tag]: { ok: false, text: e instanceof Error ? e.message : String(e) } }))
    } finally {
      setBusy(null)
    }
  }

  if (!doc || !esc) {
    return (
      <Panel className="mb-6 p-5">
        <div className="mb-2 text-[15px] font-medium text-ink">External models</div>
        <div className="text-[12px] text-ink-faint">{msg._?.text ?? 'Loading…'}</div>
      </Panel>
    )
  }

  const note = (tag: string) =>
    msg[tag] ? (
      <div className={`mt-1 text-[12px] ${msg[tag].ok ? 'text-good' : 'text-bad'}`}>{msg[tag].ok ? '✓ ' : '✗ '}{msg[tag].text}</div>
    ) : null

  const num = (k: keyof Escalation, label: string, unit: string) => (
    <label className="flex items-center gap-2 text-[12px] text-ink-dim">
      {label}
      <input type="number" min={1} value={Number(esc[k])} disabled={busy !== null}
        onChange={(e) => setEsc({ ...esc, [k]: Number(e.target.value) })}
        className="w-[72px] rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[12px] text-ink outline-none focus:border-accent" />
      {unit}
    </label>
  )

  return (
    <Panel className="mb-6 p-5">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-[15px] font-medium text-ink">External models</span>
        <Pill tone={doc.providers.some((p) => p.key_set) ? 'good' : 'neutral'}>
          {doc.providers.filter((p) => p.key_set).map((p) => p.label).join(' + ') || 'no keys'}
        </Pill>
        <Link href="/external" className="ml-auto font-mono text-[11.5px] text-accent hover:opacity-80">choose models →</Link>
      </div>
      <p className="mb-4 text-[12px] leading-relaxed text-ink-faint">
        Hosted models you pay for per token, used alongside the models on this computer. Nothing is used until you
        enable it on the <Link href="/external" className="text-accent">External</Link> page and tick it for a project.
      </p>

      {doc.providers.map((p) => (
        <div key={p.id} className="mb-4">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-ink-faint">
            {p.label} API key
            <Pill tone={p.key_set ? 'good' : 'neutral'}>{p.key_set ? 'set' : 'not set'}</Pill>
            <a href={p.keys_url} target="_blank" rel="noreferrer" className="normal-case tracking-normal text-accent hover:opacity-80">
              get a key ↗
            </a>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2">
            <input type="password" autoComplete="off" value={keys[p.id] ?? ''}
              onChange={(e) => setKeys({ ...keys, [p.id]: e.target.value })}
              placeholder={p.key_set ? '•••••••••••••••• (stored)' : p.id === 'groq' ? 'gsk_…' : 'sk-or-…'}
              className="min-w-0 flex-1 rounded-lg border border-seam bg-panel-hi px-3 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent" />
            <button disabled={busy !== null || !(keys[p.id] ?? '').trim()}
              onClick={() => run(p.id, async () => {
                const d = await external.save({ [`${p.id}_api_key`]: keys[p.id].trim() })
                take(d)
                setKeys({ ...keys, [p.id]: '' })
                const t = await external.test(p.id)
                return `Saved. ${t.detail}`
              })}
              className="rounded-lg border border-accent/45 px-3 py-1.5 font-mono text-[12px] text-accent disabled:opacity-40">
              {busy === p.id ? 'Checking…' : 'Save key'}
            </button>
            {p.key_set && (
              <>
                <button disabled={busy !== null} onClick={() => run(p.id, async () => (await external.test(p.id)).detail)}
                  className="rounded-lg border border-seam px-3 py-1.5 font-mono text-[12px] text-ink-dim hover:text-ink">
                  Test
                </button>
                <button disabled={busy !== null}
                  onClick={() => run(p.id, async () => { take(await external.save({ [`${p.id}_api_key`]: '' })); return 'Key removed.' })}
                  className="font-mono text-[11.5px] text-ink-faint hover:text-bad">
                  remove
                </button>
              </>
            )}
          </div>
          {note(p.id)}
        </div>
      ))}
      <div className="-mt-2 mb-4 font-mono text-[10.5px] text-ink-faint">
        Stored in ui\backend\auth\secrets.json with the other secrets — owner-only, never sent to this page. Test makes a
        free call (lists models / reads the key&apos;s credit), not a paid one.
      </div>

      {/* ---- Spending ---- */}
      <div className="text-[11px] uppercase tracking-wide text-ink-faint">Daily spending limit</div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-[12px] text-ink-dim">
        $
        <input type="number" min={0} step={0.5} value={limit} onChange={(e) => setLimit(e.target.value)}
          className="w-[96px] rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[12px] text-ink outline-none focus:border-accent" />
        per day
        <button disabled={busy !== null || limit === String(doc.daily_limit_usd) || !(Number(limit) >= 0)}
          onClick={() => run('limit', async () => { take(await external.save({ daily_limit_usd: Number(limit) })); return 'Limit saved.' })}
          className="rounded-lg border border-accent/45 px-3 py-1 font-mono text-[12px] text-accent disabled:opacity-40">
          Save
        </button>
        <span className="text-ink-faint">
          spent today {usd(doc.spend.today)} of {usd(doc.spend.limit)} — once reached, every external call is refused until midnight.
        </span>
      </div>
      {note('limit')}

      <div className="mt-4 text-[11px] uppercase tracking-wide text-ink-faint">Parallel agents per hosted model</div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-[12px] text-ink-dim">
        <select value={doc.parallel_agents} disabled={busy !== null}
          onChange={(e) => run('par', async () => { take(await external.save({ parallel_agents: Number(e.target.value) })); return `Each hosted model that searches now runs ${e.target.value} agents.` })}
          className="rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[12px] text-ink">
          {[1, 2, 3, 4, 5, 6, 8].map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        <span className="text-ink-faint">
          A hosted model answers many requests at once, so each one that searches for a project runs this many agents side
          by side (a local engine runs one — its GPU is the limit). More agents means more candidates per hour and more spend.
        </span>
      </div>
      {note('par')}

      <div className="mt-3 text-[11px] uppercase tracking-wide text-ink-faint">Parallel agents per local / network model</div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-[12px] text-ink-dim">
        <select value={doc.parallel_local_agents} disabled={busy !== null}
          onChange={(e) => run('parl', async () => { take(await external.save({ parallel_local_agents: Number(e.target.value) })); return `Each free model that searches now runs ${e.target.value} agent(s).` })}
          className="rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[12px] text-ink">
          {[1, 2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        <span className="text-ink-faint">
          An engine answers several requests at once, so 2 agents can search faster on the same GPU — each one uses
          more of the engine&apos;s context memory (keep Max running requests on the Models page at least this high).
        </span>
      </div>
      {note('parl')}

      {/* ---- Escalation ---- */}
      <div className="mt-4 flex items-center gap-2 text-[11px] uppercase tracking-wide text-ink-faint">
        When the swarm is stuck
        <label className="ml-2 flex items-center gap-1.5 normal-case tracking-normal text-[12px] text-ink-dim">
          <input type="checkbox" checked={esc.enabled} disabled={busy !== null}
            onChange={(e) => run('esc', async () => { take(await external.save({ escalation: { enabled: e.target.checked } })); return e.target.checked ? 'On.' : 'Off.' })} />
          ask stronger models for ideas
        </label>
      </div>
      <p className="mt-1 text-[12px] leading-relaxed text-ink-faint">
        An objective counts as stuck after this many candidates <i>and</i> this long without a new best. The first ask
        goes to the free model with the highest AA Intelligence score; if the search is still stuck after the step
        below, the next ask goes to the cheapest stronger external model ticked for the project, and so on up.
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-2">
        {num('stuck_candidates', 'stuck after', 'candidates')}
        {num('stuck_minutes', 'and', 'minutes')}
        {num('step_candidates', 'next step after', 'more candidates')}
        {num('step_minutes', 'and', 'minutes')}
        <button disabled={busy !== null || JSON.stringify(esc) === JSON.stringify(doc.escalation)}
          onClick={() => run('esc', async () => { take(await external.save({ escalation: esc })); return 'Saved.' })}
          className="rounded-lg border border-accent/45 px-3 py-1 font-mono text-[12px] text-accent disabled:opacity-40">
          Save
        </button>
      </div>
      {note('esc')}
    </Panel>
  )
}
