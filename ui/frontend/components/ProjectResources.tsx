'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { api, type LoadedModel } from '@/lib/api'
import { external as ext, type ModelRole, type SwarmPlan, type SwarmPlanEntry, usd } from '@/lib/external'
import { RatingChips, useRatings } from '@/lib/ratings'
import { projectResources as res, type DataFile, type Project } from '@/lib/projects'
import { bytesLabel } from '@/lib/format'
import { Panel, Pill } from '@/components/ui'
import QueryBox from '@/components/QueryBox'

/**
 * Which LOADED models this project's swarm may use.
 *
 * Only loaded models are offered. Agents are only ever created for loaded models, so ticking
 * an unloaded one did nothing visible and read as broken. The list follows the Models page
 * live: load a model there and it appears here within a few seconds.
 *
 * "All loaded models" is the default. Unticking narrows the project to an explicit list.
 * Entries for models that are not loaded right now are kept in that list, untouched, rather
 * than silently dropped: unloading a model to free memory must not quietly change what this
 * project may use when it comes back.
 *
 * External (pay-per-token) models enabled on the External page are offered in their own group
 * and are never part of "all loaded models": each one has to be ticked for the project. Below
 * the picker, the swarm plan says which models search and which are kept for new ideas when
 * the search is stuck (app/swarm_policy.py).
 */
export function ProjectModels({ project, onChanged }: { project: Project; onChanged: () => void }) {
  const [llms, setLlms] = useState<string[]>([])
  const [paid, setPaid] = useState<LoadedModel[]>([])
  const [forecasters, setForecasters] = useState<string[]>([])
  const [plan, setPlan] = useState<SwarmPlan | null>(null)
  const ratingFor = useRatings()
  const [ready, setReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const [e, t, p] = await Promise.all([
          api.engines(),
          api.tsInstances().catch(() => ({ instances: [] })),
          ext.plan(project.id).catch(() => null),
        ])
        if (!alive) return
        const ready = e.loaded.filter((m) => m.ready && m.model)
        setLlms([...new Set(ready.filter((m) => !m.external).map((m) => m.model as string))].sort())
        setPaid(ready.filter((m) => m.external).sort((a, b) => (a.model as string).localeCompare(b.model as string)))
        setPlan(p)
        setForecasters(
          [...new Set(t.instances.filter((i) => i.state === 'running').map((i) => i.model_id))].sort(),
        )
        setReady(true)
      } catch {
        if (alive) setReady(true)
      }
    }
    void load()
    const timer = setInterval(load, 5000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [project.id, project.models, JSON.stringify(project.model_roles ?? {})])

  // "All loaded models" covers the free ones only; a paid model is always an explicit choice.
  const loaded = [...llms, ...forecasters]
  const paidNames = paid.map((m) => m.model as string)
  const all = project.models === null || project.models === undefined
  const chosen = new Set(project.models ?? [])
  // Allowed in the saved list but not loaded right now: kept, and mentioned, never lost.
  const dormant = all ? [] : (project.models ?? []).filter((m) => !loaded.includes(m) && !paidNames.includes(m))

  async function save(next: string[] | null) {
    setBusy(true)
    setErr(null)
    try {
      await res.setModels(project.id, next)
      onChanged()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  function toggle(id: string) {
    // From "all": start from every loaded free model. From a list: keep its dormant entries.
    const base = all ? new Set(loaded) : new Set(chosen)
    if (base.has(id)) base.delete(id)
    else base.add(id)
    void save([...base])
  }

  const groups: [string, string[]][] = [
    ['LLMs — become agents', llms],
    ['Time series — offered to agents as the forecast tool', forecasters],
  ]

  return (
    <Panel className="p-5">
      <div className="flex items-center gap-3">
        <span className="text-[15px] font-medium text-ink">Models</span>
        <label className="ml-auto flex cursor-pointer items-center gap-2 text-[12px] text-ink-dim">
          <input
            type="checkbox"
            checked={all}
            disabled={busy || loaded.length === 0}
            onChange={() => save(all ? loaded : null)}
          />
          all loaded models (free)
        </label>
      </div>
      <p className="mt-1 text-[12px] text-ink-faint">
        Only loaded models can be chosen. Each ticked LLM becomes an agent; each ticked
        forecaster is offered to the agents as a tool.
      </p>

      {ready && loaded.length === 0 && (
        <div className="mt-3 text-[12px] text-ink-faint">
          No models are loaded.{' '}
          <Link href="/models" className="text-accent hover:opacity-80">
            Load one on the Models page
          </Link>{' '}
          and it will appear here.
        </div>
      )}

      <div className="mt-3 space-y-3">
        {groups.map(([label, list]) =>
          list.length ? (
            <div key={label}>
              <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">{label}</div>
              <div className="flex flex-wrap gap-2">
                {list.map((id) => {
                  const on = all || chosen.has(id)
                  const remote = id.includes('@')
                  const rating = ratingFor(id)
                  return (
                    <button
                      key={id}
                      onClick={() => toggle(id)}
                      disabled={busy}
                      title={remote ? `served by ${id.slice(id.lastIndexOf('@') + 1)} over the network` : undefined}
                      className={`rounded-lg border px-2.5 py-1 font-mono text-[11.5px] transition-colors ${
                        on
                          ? remote
                            ? 'border-remote/60 bg-remote/10 text-remote'
                            : 'border-accent/50 bg-accent/[0.08] text-ink'
                          : remote
                            ? 'border-remote/30 text-remote/60 hover:text-remote'
                            : 'border-seam text-ink-faint hover:text-ink-dim'
                      }`}
                    >
                      {on ? '✓ ' : ''}
                      {id}
                      {rating && <span className="ml-1.5"><RatingChips rating={rating} /></span>}
                    </button>
                  )
                })}
              </div>
            </div>
          ) : null,
        )}
      </div>

      {paid.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">
            External — pay per token, never included by &quot;all&quot;; tick to allow
          </div>
          <div className="flex flex-wrap gap-2">
            {paid.map((m) => {
              const id = m.model as string
              const on = chosen.has(id)
              const x = m.external!
              return (
                <button key={id} disabled={busy}
                  onClick={() => {
                    const base = all ? new Set(loaded) : new Set(chosen)
                    if (base.has(id)) base.delete(id)
                    else base.add(id)
                    void save([...base])
                  }}
                  title={`${x.provider_label} · $${x.price_in}/$${x.price_out} per Mtok in/out`}
                  className={`rounded-lg border px-2.5 py-1 font-mono text-[11.5px] transition-colors ${
                    on ? 'border-warn/70 bg-warn/20 text-ink' : 'border-warn/30 bg-warn/[0.06] text-ink-dim hover:text-ink'}`}>
                  {on ? '✓ ' : ''}<span className="mr-1 font-bold text-warn">$</span>{id}
                  <span className="ml-1.5 text-ink-faint">{usd(x.price_blended)}/M</span>
                  <span className="ml-1.5"><RatingChips rating={{ key: id, label: id, match: '', swe: m.swe ?? null, swe_source: null, aa: m.aa ?? null, aa_source: null }} /></span>
                </button>
              )
            })}
          </div>
        </div>
      )}
      {ready && paid.length === 0 && (
        <div className="mt-3 text-[11.5px] text-ink-faint">
          External models (Groq, OpenRouter) appear here once enabled on the{' '}
          <Link href="/external" className="text-accent hover:opacity-80">External page</Link>.
        </div>
      )}

      {plan && <SwarmPlanView plan={plan} projectId={project.id} onChanged={onChanged} />}

      {dormant.length > 0 && (
        <div className="mt-3 font-mono text-[10.5px] text-ink-faint">
          also allowed once loaded: {dormant.join(', ')}
        </div>
      )}
      {err && <div className="mt-2 text-[12px] text-bad">{err}</div>}
    </Panel>
  )
}

/**
 * The data folder as the agents see it: every parquet/CSV/JSON file becomes a SQL view, and
 * you can query them here with exactly the confinement the agents get.
 */
export function ProjectDataQuery({ project, refreshKey = 0 }: { project: Project; refreshKey?: number }) {
  const [files, setFiles] = useState<DataFile[] | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    setFiles(null)
    res
      .dataCatalog(project.id)
      .then((r) => alive && setFiles(r.files))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [project.id, project.data_dir, refreshKey])

  const first = files?.[0]

  return (
    <Panel className="p-5">
      <div className="flex items-center gap-3">
        <span className="text-[15px] font-medium text-ink">Data for the swarm</span>
        {files && <Pill tone={files.length ? 'good' : 'neutral'}>{files.length} queryable file{files.length === 1 ? '' : 's'}</Pill>}
      </div>
      <p className="mt-1 text-[12px] leading-relaxed text-ink-faint">
        Point the data directory above at a drive or folder of <strong className="text-ink-dim">parquet, CSV or JSON</strong> and
        agents can query it in SQL — in place, nothing is imported. They cannot open anything outside
        this folder, and cannot write to it.
      </p>
      {err && <div className="mt-2 text-[12px] text-bad">{err}</div>}
      {files && files.length > 0 && (
        <>
          <div className="mt-3 max-h-40 overflow-y-auto rounded-lg border border-seam">
            {files.map((f) => (
              <div key={f.path} className="flex items-center gap-3 border-b border-seam/40 px-3 py-1 font-mono text-[11px] last:border-0">
                <span className="text-accent">{f.view}</span>
                <span className="truncate text-ink-faint">
                  {f.dataset ? `${f.source ?? 'SQL export'} · ${(f.rows ?? 0).toLocaleString()} rows` : f.path}
                </span>
                <span className="ml-auto text-ink-faint">{bytesLabel(f.bytes)}</span>
              </div>
            ))}
          </div>
          <div className="mt-3">
            <QueryBox
              run={(sql) => res.dataQuery(project.id, sql)}
              placeholder={first ? `SELECT * FROM ${first.view} LIMIT 10` : 'SELECT ...'}
              initial={first ? `SELECT * FROM ${first.view} LIMIT 10` : ''}
            />
          </div>
        </>
      )}
      {files && files.length === 0 && (
        <div className="mt-3 text-[12px] text-ink-faint">
          No parquet, CSV or JSON files in {project.data_dir}. Change the data directory above, or upload files below.
        </div>
      )}
    </Panel>
  )
}



/** How the swarm will use this project's models: who searches, who is asked for ideas. */
function SwarmPlanView({ plan, projectId, onChanged }: { plan: SwarmPlan; projectId: string; onChanged: () => void }) {
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  async function setRole(model: string, role: ModelRole) {
    setBusy(model)
    setErr(null)
    try {
      await ext.setRole(projectId, model, role)
      onChanged()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }
  const agents = (m: string) => plan.search.find((x) => x.model === m)?.agents ?? 0
  const chip = (m: SwarmPlanEntry, extra?: string) => (
    <span key={m.model} title={m.why}
      className={`rounded-md border px-2 py-0.5 font-mono text-[11px] ${m.kind === 'external' ? 'border-warn/40 text-ink' : m.kind === 'network' ? 'border-remote/40 text-remote' : 'border-seam text-ink-dim'}`}>
      {extra}{m.model}
      <span className="ml-1 text-ink-faint">
        {m.swe != null ? ` SWE ${m.swe}%` : ''}{m.aa != null ? ` AA ${m.aa}` : ''}
        {m.price_blended != null ? ` ${usd(m.price_blended)}/M` : m.kind !== 'external' ? ' free' : ''}
      </span>
    </span>
  )
  const [lo, hi] = plan.swe_range
  return (
    <div className="mt-4 rounded-xl border border-seam p-3">
      <div className="text-[12.5px] font-medium text-ink">How the swarm uses these</div>
      <div className="mt-1 text-[11.5px] text-ink-faint">
        Give each model a role. <b>Search</b> models write and test strategy code — the signals that end up in the
        library — all at the same time. <b>New ideas</b> models are asked for fresh strategy directions when the search
        stops improving. <b>Auto</b> follows the rules below.
      </div>
      <table className="mt-2 w-full border-collapse text-[11.5px]">
        <thead>
          <tr className="border-b border-seam text-left font-mono text-[10px] uppercase tracking-wide text-ink-faint">
            <th className="py-1 pr-2">model</th><th className="px-2 text-right">SWE</th><th className="px-2 text-right">AA</th>
            <th className="px-2 text-right">cost</th><th className="px-2">role</th><th className="px-2">does</th>
          </tr>
        </thead>
        <tbody>
          {plan.models.map((m) => (
            <tr key={m.model} className={`border-b border-seam/50 ${m.kind === 'external' ? 'bg-warn/[0.07]' : ''}`}>
              <td className={`py-1 pr-2 font-mono ${m.kind === 'external' ? 'text-warn' : m.kind === 'network' ? 'text-remote' : 'text-ink'}`}>
                {m.kind === 'external' && <b className="mr-1">$</b>}{m.model}
              </td>
              <td className="px-2 text-right font-mono text-ink-dim">{m.swe != null ? `${m.swe}%` : '—'}</td>
              <td className="px-2 text-right font-mono text-ink-dim">{m.aa ?? '—'}</td>
              <td className="px-2 text-right font-mono text-ink-dim">{m.price_blended != null ? `${usd(m.price_blended)}/M` : 'free'}</td>
              <td className="px-2">
                <select value={m.role} disabled={busy !== null} onChange={(e) => setRole(m.model, e.target.value as ModelRole)}
                  className="rounded-md border border-seam bg-panel-hi px-1.5 py-0.5 font-mono text-[11px] text-ink">
                  <option value="auto">Auto</option>
                  <option value="search">Search</option>
                  <option value="ideas">New ideas</option>
                  <option value="both">Both</option>
                </select>
              </td>
              <td className="px-2 font-mono text-[10.5px] text-ink-faint">
                {[m.searching ? `search ×${agents(m.model)}` : '', m.ideas ? `ideas #${plan.ladder.findIndex((x) => x.model === m.model) + 1}` : '']
                  .filter(Boolean).join(' · ') || 'idle'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {err && <div className="mt-1 text-[11.5px] text-bad">{err}</div>}
      <div className="mt-2 text-[11.5px] text-ink-dim">
        <b>Search</b> — agents that write and test candidates. Every free model, plus each ticked external model that is
        similar to them: the same model hosted elsewhere, or a SWE-bench score within {plan.swe_margin} points of the free
        models&apos; range{lo != null ? ` (${lo}–${hi}%)` : ''}, or — when SWE can&apos;t decide — an AA score within{' '}
        {plan.aa_margin} points of theirs{plan.aa_range[0] != null ? ` (${plan.aa_range[0]}–${plan.aa_range[1]})` : ''}.
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1.5">{plan.search.length ? plan.search.map((m) => chip(m)) : <span className="text-[11px] text-ink-faint">none loaded</span>}</div>
      <div className="mt-3 text-[11.5px] text-ink-dim">
        <b>When stuck</b> — asked for new directions in this order, climbing one step each time the search stays stuck:
        the free model with the best AA Intelligence score first, then stronger external models, cheapest first.
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        {plan.ladder.length
          ? plan.ladder.map((m, i) => (
              <span key={m.model} className="flex items-center gap-1.5">
                {i > 0 && <span className="text-ink-faint">→</span>}
                {chip(m, `${i + 1}. `)}
              </span>
            ))
          : <span className="text-[11px] text-ink-faint">no rated model — load one, or tick an external model</span>}
      </div>
      {plan.reserved.length > 0 && (
        <div className="mt-2 font-mono text-[10.5px] text-ink-faint">
          not searching: {plan.reserved.map((m) => `${m.model} (${m.why})`).join('; ')}
        </div>
      )}
      {plan.not_in_ladder.length > 0 && (
        <div className="mt-1 font-mono text-[10.5px] text-ink-faint">
          not asked for ideas: {plan.not_in_ladder.map((m) => `${m.model} (${m.why})`).join('; ')}
        </div>
      )}
    </div>
  )
}
