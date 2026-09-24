'use client'

import Link from 'next/link'
import { useEffect, useMemo, useState } from 'react'
import { external, type Catalog, type ExternalModel, type Overview, usd } from '@/lib/external'
import { PageHeader, Panel, Pill } from '@/components/ui'

type SortKey = 'name' | 'context' | 'price_in' | 'price_out' | 'price_blended' | 'speed' | 'swe' | 'aa'

const COLUMNS: { key: SortKey; label: string; tip: string; right?: boolean }[] = [
  { key: 'name', label: 'Model', tip: 'Name and the id used by the provider' },
  { key: 'context', label: 'Context', tip: 'Context window, tokens', right: true },
  { key: 'price_in', label: '$ in', tip: 'USD per million input tokens', right: true },
  { key: 'price_out', label: '$ out', tip: 'USD per million output tokens', right: true },
  { key: 'price_blended', label: '$ blended', tip: 'USD per million tokens at 3 input : 1 output — the usual agent mix', right: true },
  { key: 'speed', label: 'Speed', tip: 'Output tokens per second (Groq publishes it; OpenRouter does not)', right: true },
  { key: 'swe', label: 'SWE', tip: 'SWE-bench Verified, % of real GitHub issues resolved (coding)', right: true },
  { key: 'aa', label: 'AA', tip: 'Artificial Analysis Intelligence Index (reasoning, knowledge, math, coding)', right: true },
]

const ctxLabel = (n: number | null) => (n == null ? '—' : n >= 1_000_000 ? `${(n / 1_048_576).toFixed(n % 1_048_576 ? 1 : 0)}M` : `${Math.round(n / 1024)}K`)

export default function ExternalPage() {
  const [doc, setDoc] = useState<Overview | null>(null)
  const [tab, setTab] = useState<'groq' | 'openrouter'>('groq')
  const [cats, setCats] = useState<Record<string, Catalog>>({})
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [onlyEnabled, setOnlyEnabled] = useState(false)
  const [onlyRated, setOnlyRated] = useState(false)
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: 'aa', desc: true })
  const [busy, setBusy] = useState<string | null>(null)
  const [rowErr, setRowErr] = useState<Record<string, string>>({})

  async function loadCatalog(p: string, refresh = false) {
    setLoading(true)
    setErr(null)
    try {
      const c = await external.models(p, refresh)
      setCats((m) => ({ ...m, [p]: c }))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => {
    external.overview().then(setDoc).catch((e) => setErr(e instanceof Error ? e.message : String(e)))
  }, [])
  useEffect(() => {
    if (!cats[tab]) void loadCatalog(tab)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab])

  const cat = cats[tab]
  const provider = doc?.providers.find((p) => p.id === tab)
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    const list = (cat?.rows ?? []).filter(
      (r) =>
        (!q || r.id.toLowerCase().includes(q) || r.name.toLowerCase().includes(q)) &&
        (!onlyEnabled || r.enabled) &&
        (!onlyRated || r.swe != null || r.aa != null),
    )
    const val = (r: ExternalModel) => (sort.key === 'name' ? r.name.toLowerCase() : r[sort.key])
    return [...list].sort((a, b) => {
      const x = val(a)
      const y = val(b)
      if (x == null && y == null) return 0
      if (x == null) return 1 // unknowns last, whichever way
      if (y == null) return -1
      const c = x < y ? -1 : x > y ? 1 : 0
      return sort.desc ? -c : c
    })
  }, [cat, query, onlyEnabled, onlyRated, sort])

  async function toggle(r: ExternalModel) {
    setBusy(r.model)
    setRowErr((m) => ({ ...m, [r.model]: '' }))
    try {
      const { enabled } = await external.setEnabled(r.model, !r.enabled)
      setCats((m) => ({ ...m, [tab]: { ...m[tab], rows: m[tab].rows.map((x) => ({ ...x, enabled: enabled.includes(x.model) })) } }))
      setDoc((d) => (d ? { ...d, enabled } : d))
    } catch (e) {
      setRowErr((m) => ({ ...m, [r.model]: e instanceof Error ? e.message : String(e) }))
    } finally {
      setBusy(null)
    }
  }

  const enabledCount = (p: string) => (doc?.enabled ?? []).filter((n) => n.endsWith(`@${p}`)).length

  return (
    <div className="mx-auto max-w-[1200px] px-8 py-8">
      <PageHeader
        title="External"
        subtitle="Hosted models you pay for per token — enable one here, then tick it for a project"
        right={doc && (
          <Pill tone={doc.spend.today >= doc.spend.limit ? 'bad' : 'neutral'}>
            today {usd(doc.spend.today)} of {usd(doc.spend.limit)}
          </Pill>
        )}
      />

      <div className="mb-4 flex flex-wrap items-center gap-2">
        {(doc?.providers ?? []).map((p) => (
          <button key={p.id} onClick={() => setTab(p.id)}
            className={`rounded-xl border px-4 py-2 text-[13px] ${tab === p.id ? 'border-accent/50 bg-accent/[0.08] text-ink' : 'border-seam text-ink-dim hover:text-ink'}`}>
            {p.label}
            <span className="ml-2 font-mono text-[11px] text-ink-faint">
              {p.key_set ? `${enabledCount(p.id)} enabled` : 'no key'}
            </span>
          </button>
        ))}
      </div>

      {provider && !provider.key_set && (
        <Panel className="mb-4 border-warn/35 bg-warn/5 p-4 text-[12.5px] text-ink-dim">
          No {provider.label} API key yet. You can browse and enable models, but they cannot answer until you{' '}
          <a href={provider.keys_url} target="_blank" rel="noreferrer" className="text-accent">create a key ↗</a> and paste it in{' '}
          <Link href="/settings" className="text-accent">Settings → External models</Link>.
        </Panel>
      )}

      <Panel className="p-4">
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search models…"
            className="min-w-[220px] flex-1 rounded-lg border border-seam bg-panel-hi px-3 py-1.5 text-[12.5px] text-ink outline-none focus:border-accent" />
          <label className="flex items-center gap-1.5 text-[12px] text-ink-dim">
            <input type="checkbox" checked={onlyEnabled} onChange={(e) => setOnlyEnabled(e.target.checked)} /> enabled only
          </label>
          <label className="flex items-center gap-1.5 text-[12px] text-ink-dim">
            <input type="checkbox" checked={onlyRated} onChange={(e) => setOnlyRated(e.target.checked)} /> rated only
          </label>
          <button onClick={() => loadCatalog(tab, true)} disabled={loading}
            className="font-mono text-[11.5px] text-accent hover:opacity-80 disabled:opacity-40">
            {loading ? 'loading…' : 'refresh'}
          </button>
        </div>
        <div className="mb-2 font-mono text-[10.5px] text-ink-faint">
          {cat ? `${rows.length} of ${cat.rows.length} models · ` : ''}
          {tab === 'groq'
            ? `prices and speeds from console.groq.com/docs/models as of ${cat?.prices_as_of ?? '…'} (Groq has no pricing API)`
            : 'live from openrouter.ai · AA scores republished by OpenRouter from artificialanalysis.ai'}
          {' · SWE-bench Verified only where the developer publishes it'}
        </div>
        {(err || cat?.note) && <div className="mb-2 text-[12px] text-warn">{err ?? cat?.note}</div>}

        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[12px]">
            <thead>
              <tr className="border-b border-seam text-left">
                {COLUMNS.map((c) => (
                  <th key={c.key} title={c.tip}
                    onClick={() => setSort((s) => ({ key: c.key, desc: s.key === c.key ? !s.desc : c.key !== 'name' && !c.key.startsWith('price') }))}
                    className={`cursor-pointer select-none px-2 py-1.5 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint hover:text-ink ${c.right ? 'text-right' : ''}`}>
                    {c.label}{sort.key === c.key ? (sort.desc ? ' ↓' : ' ↑') : ''}
                  </th>
                ))}
                <th className="px-2 py-1.5 text-right font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">Use</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.model} className={`border-b border-seam/50 ${r.enabled ? 'bg-accent/[0.04]' : ''}`}>
                  <td className="px-2 py-1.5">
                    <div className="text-ink">{r.name}</div>
                    <div className="font-mono text-[10.5px] text-ink-faint">
                      {r.id}
                      {r.tier ? ` · ${r.tier}` : ''}
                      {r.available === false ? ' · not available to this key' : ''}
                    </div>
                    {rowErr[r.model] && <div className="text-[11px] text-bad">{rowErr[r.model]}</div>}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-ink-dim">{ctxLabel(r.context)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-ink-dim">{r.price_in == null ? '—' : r.price_in.toFixed(2)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-ink-dim">{r.price_out == null ? '—' : r.price_out.toFixed(2)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-ink">{r.price_blended == null ? '—' : r.price_blended.toFixed(2)}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-ink-dim">{r.speed == null ? '—' : `${r.speed} t/s`}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-ink" title={r.rating_label ?? undefined}>{r.swe == null ? '—' : `${r.swe}%`}</td>
                  <td className="px-2 py-1.5 text-right font-mono text-ink" title={r.rating_label ?? undefined}>{r.aa ?? '—'}</td>
                  <td className="px-2 py-1.5 text-right">
                    {r.priced ? (
                      <button onClick={() => toggle(r)} disabled={busy !== null}
                        className={`rounded-lg border px-2.5 py-0.5 font-mono text-[11px] ${r.enabled ? 'border-accent/50 bg-accent/10 text-accent' : 'border-seam text-ink-dim hover:text-ink'}`}>
                        {busy === r.model ? '…' : r.enabled ? '✓ enabled' : 'enable'}
                      </button>
                    ) : (
                      <span className="font-mono text-[10.5px] text-ink-faint" title="No published price, so its use cannot be metered">unpriced</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {doc && doc.spend.by_model.length > 0 && (
        <Panel className="mt-4 p-4">
          <div className="mb-2 text-[13px] font-medium text-ink">Spent today</div>
          {doc.spend.by_model.map((s) => (
            <div key={`${s.provider}/${s.model}`} className="flex justify-between border-b border-seam/50 py-1 font-mono text-[11.5px] text-ink-dim last:border-0">
              <span>{s.model}@{s.provider} · {s.calls} calls · {(s.prompt_tokens + s.completion_tokens).toLocaleString()} tokens</span>
              <span className="text-ink">{usd(s.usd, 4)}</span>
            </div>
          ))}
        </Panel>
      )}
    </div>
  )
}
