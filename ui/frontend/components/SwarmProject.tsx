'use client'

import Link from 'next/link'
import { Fragment, useEffect, useState } from 'react'
import { projects, projectResources as res, type Project, type SwarmResources } from '@/lib/projects'
import { api, type EngineDetail, type TsInstance } from '@/lib/api'
import { external, usd, type ExternalUsage, type ModelUsage } from '@/lib/external'
import { bytesLabel } from '@/lib/format'
import { usePoll } from '@/lib/usePoll'
import { Panel, Pill } from '@/components/ui'
import AgentInspector from '@/components/AgentInspector'

/**
 * Which project this swarm belongs to, and whether it is running.
 *
 * A swarm IS a project's swarm: its board, its agents and everything they may touch are that
 * project's. Switching here is the same action as the sidebar switcher (it activates the
 * project), so the two can never disagree about which swarm you are looking at.
 */
export function SwarmProjectBar() {
  const [list, setList] = useState<Project[]>([])
  const [active, setActive] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    projects
      .list()
      .then((r) => {
        setList(r.projects)
        setActive(r.active)
      })
      .catch(() => {})
  }, [])

  const current = list.find((p) => p.id === active)

  async function switchTo(id: string) {
    if (!id || id === active) return
    setBusy(true)
    try {
      await projects.activate(id)
      // Same as the sidebar switcher: every panel on this page is scoped to the active
      // project, so reload rather than try to reset each one by hand.
      window.location.reload()
    } finally {
      setBusy(false)
    }
  }

  async function toggle() {
    if (!current) return
    setBusy(true)
    try {
      const updated = await res.setSwarm(current.id, !current.swarm_enabled)
      setList((l) => l.map((p) => (p.id === updated.id ? { ...p, ...updated } : p)))
      window.dispatchEvent(new Event('ft-swarm-changed'))
    } finally {
      setBusy(false)
    }
  }

  if (!current) return null
  const on = current.swarm_enabled !== false

  return (
    <Panel className="mb-4 flex flex-wrap items-center gap-3 p-3">
      <span className="text-[12px] text-ink-faint">Swarm for project</span>
      <select
        value={active ?? ''}
        onChange={(e) => switchTo(e.target.value)}
        disabled={busy}
        className="rounded-lg border border-seam bg-panel-hi px-2.5 py-1.5 text-[13px] font-medium text-ink outline-none"
      >
        {list.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
            {p.swarm_enabled === false ? ' (swarm off)' : ''}
          </option>
        ))}
      </select>
      <button
        onClick={toggle}
        disabled={busy}
        role="switch"
        aria-checked={on}
        title={on ? 'Agents are working this project. Click to stop them.' : 'Start agents for this project.'}
        className="flex items-center gap-2 rounded-lg border border-seam px-2.5 py-1.5 text-[12px] transition-colors hover:border-accent/50"
      >
        <span className={`relative h-4 w-7 rounded-full transition-colors ${on ? 'bg-good' : 'bg-seam'}`}>
          <span
            className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all ${on ? 'left-3.5' : 'left-0.5'}`}
          />
        </span>
        <span className={on ? 'text-ink' : 'text-ink-faint'}>{on ? 'swarm on' : 'swarm off'}</span>
      </button>
      <Link href="/projects" className="ml-auto font-mono text-[11px] text-accent hover:opacity-80">
        edit project resources →
      </Link>
    </Panel>
  )
}

/**
 * What this project's agents can reach -- the same world the swarm runner hands them as
 * tools. If it is not listed here, no agent in this swarm can use it.
 */
export function SwarmResourcesPanel() {
  const [r, setR] = useState<SwarmResources | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [pid, setPid] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const list = await projects.list()
        if (!list.active) return
        const data = await res.swarmResources(list.active)
        if (alive) {
          setR(data)
          setPid(list.active)
          setErr(null)
        }
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e))
      }
    }
    void load()
    // Models load and unload from other pages; keep this current without a reload.
    const t = setInterval(load, 10000)
    const onChange = () => void load()
    window.addEventListener('ft-swarm-changed', onChange)
    return () => {
      alive = false
      clearInterval(t)
      window.removeEventListener('ft-swarm-changed', onChange)
    }
  }, [])

  const live = useLive()
  // The model row clicked open in the agent inspector (what it is asked, its inputs...).
  const [inspect, setInspect] = useState<{ model: string; kind: 'llm' | 'forecaster' } | null>(null)
  // What each hosted model has cost and produced, and what is blocking it. A paid model that
  // spends and produces nothing (rate limits, rejected tool calls, a spent budget) otherwise
  // looks exactly like one that is working. Only fetched once the project has hosted models.
  const hasPaid = !!r?.llms.some((l) => l.loaded && l.external)
  const usage = usePoll<ExternalUsage | null>(
    () => (pid && hasPaid ? external.usage(pid) : Promise.resolve(null)),
    15000,
  )
  const refreshUsage = usage.refresh
  useEffect(() => {
    if (pid && hasPaid) refreshUsage()
  }, [pid, hasPaid, refreshUsage])

  if (err) return <Panel className="p-4 text-[12px] text-bad">{err}</Panel>
  if (!r) return null

  const agents = r.llms.filter((l) => l.loaded)
  const firstPaid = agents.findIndex((l) => l.external)
  const u = usage.data
  const waiting = r.llms.filter((l) => !l.loaded)
  const now = Date.now() / 1000
  const totalTps = agents.reduce((sum, l) => {
    const e = live.engines.find((x) => x.model_id === l.model)
    return sum + (e?.stats?.throughput.decode_tps ?? 0)
  }, 0)

  return (
    <Panel className="p-4">
      <div className="mb-3 flex items-center gap-2">
        <span className="text-[14px] font-medium text-ink">Resources</span>
        {!r.enabled && <Pill tone="warn">swarm off — no agents</Pill>}
        {totalTps > 0.5 && (
          <span className="ml-auto font-mono text-[11px] text-good">{totalTps.toFixed(1)} tok/s total</span>
        )}
      </div>

      <Section label="LLMs → agents" count={agents.length}>
        {agents.length === 0 && (
          <Empty>
            No allowed LLM is loaded, so this swarm has no agents.{' '}
            <Link href="/models" className="text-accent">
              Load one
            </Link>
            .
          </Empty>
        )}
        {agents.map((l, i) => {
          const e = live.engines.find((x) => x.model_id === l.model)
          const st = e?.stats
          // A model on another computer has no local engine: its numbers come with the list.
          // A hosted model has neither: the control plane counts its calls in flight. "Called in
          // the last 20 s" keeps the light on between an agent's back-to-back rounds.
          const h = l.external ? live.hosted?.models[l.model] : undefined
          const hostedActive = h ? h.in_flight + ((live.hosted?.now ?? 0) - h.last_active < 20 ? 1 : 0) : 0
          const active = st?.requests.active ?? l.remote?.active ?? hostedActive
          const tps = st?.throughput.decode_tps ?? (l.remote && active ? l.remote.decode_tps ?? 0 : 0)
          const prefill = st?.throughput.prefill_tps ?? 0
          const state: LedState = !l.ready ? 'loading' : active > 0 ? 'active' : 'ready'
          const rate =
            tps >= 0.5 ? `${tps.toFixed(1)} tok/s` : prefill >= 1 ? `prefill ${Math.round(prefill)} tok/s` : 'thinking'
          const mu = l.external ? u?.models.find((m) => m.model === l.model) : undefined
          return (
            <Fragment key={l.model}>
              {i === firstPaid && u && <SpendLine u={u} />}
              <LiveRow
                name={l.model}
                onOpen={() => setInspect({ model: l.model, kind: 'llm' })}
                remote={!!l.remote}
                paid={!!l.external}
                state={state}
                right={
                  state === 'active'
                    ? `${rate} · ${active} req`
                    : state === 'loading'
                      ? 'loading'
                      : l.external
                        ? `$${l.external.price_blended ?? '?'}/Mtok · ${l.external.provider_label}`
                        : l.remote
                          ? `idle · on ${l.remote.node}`
                          : `idle · GPU ${l.gpu ?? '?'}`
                }
                detail={
                  st?.kv ? (
                    <CtxBar
                      used={st.kv.used_pages * (st.kv.page_size || 1)}
                      total={st.kv.total_pages * (st.kv.page_size || 1)}
                    />
                  ) : mu && u ? (
                    <PaidUsage m={mu} u={u} />
                  ) : undefined
                }
              />
            </Fragment>
          )
        })}
        {waiting.map((l) => (
          <Row key={l.model} name={l.model} right="allowed · not loaded" tone="neutral" />
        ))}
        {r.models_filter && (
          <div className="mt-1 font-mono text-[10px] text-ink-faint">limited to the project&apos;s chosen models</div>
        )}
      </Section>

      <Section label="Forecasters → forecast tool" count={r.forecasters.length}>
        {r.forecasters.length === 0 && <Empty>None loaded.</Empty>}
        {r.forecasters.map((f) => {
          const t = live.ts.find((x) => x.model_id === f.model)
          // A forecast takes well under a second -- far shorter than the poll -- so "active"
          // also lights for a few seconds after a call; otherwise the LED would never be seen.
          const recent = t?.last_call_at != null && now - t.last_call_at < 4
          const running = (t?.state ?? f.state) === 'running'
          const state: LedState = !running ? 'loading' : (t?.in_flight ?? 0) > 0 || recent ? 'active' : 'ready'
          const calls = t?.calls ?? 0
          return (
            <LiveRow
              key={f.model}
              name={f.model}
              onOpen={() => setInspect({ model: f.model, kind: 'forecaster' })}
              state={state}
              right={
                state === 'active'
                  ? `forecasting${t?.last_seconds != null ? ` · ${t.last_seconds}s` : ''}`
                  : `${calls} call${calls === 1 ? '' : 's'} · GPU ${f.gpu}`
              }
            />
          )
        })}
      </Section>

      <Section label="Data → query_data" count={r.data.count}>
        {r.data.count === 0 && <Empty>No parquet, CSV or JSON files in the data folder.</Empty>}
        {r.data.files.slice(0, 8).map((f) => (
          <Row
            key={f.path}
            name={f.view}
            right={f.dataset ? `${(f.rows ?? 0).toLocaleString()} rows` : bytesLabel(f.bytes)}
            tone="good"
          />
        ))}
        {r.data.count > 8 && <Empty>and {r.data.count - 8} more</Empty>}
      </Section>

      <Section label="SQL Server → query_sql" count={r.sql?.tables.length ?? 0}>
        {!r.sql && <Empty>Not set up.</Empty>}
        {r.sql && (
          <>
            <div className="mb-1 font-mono text-[10.5px] text-ink-faint">
              {r.sql.database} · read-only
            </div>
            {r.sql.tables.map((t) => (
              <Row key={t} name={t} right="SELECT" tone="good" />
            ))}
          </>
        )}
      </Section>

      <Section label="Connectors → their tools" count={r.connectors.active.length}>
        {r.connectors.active.length === 0 && <Empty>None active.</Empty>}
        {r.connectors.active.map((c) => (
          <Row key={c} name={c} right="active" tone="good" />
        ))}
        {r.connectors.unavailable.map((c) => (
          <Row key={c} name={c} right="off globally" tone="warn" />
        ))}
      </Section>
      {inspect && (
        <AgentInspector
          model={inspect.model}
          kind={inspect.kind}
          projectId={pid}
          onClose={() => setInspect(null)}
        />
      )}
    </Panel>
  )
}

/** Live activity: engine stats for LLMs, call counters for forecasters. Polled fast and
 *  separately from the resource list, which walks the data folder and must stay slow. */
function useLive() {
  const [engines, setEngines] = useState<EngineDetail[]>([])
  const [ts, setTs] = useState<TsInstance[]>([])
  const [hosted, setHosted] = useState<Awaited<ReturnType<typeof external.activity>> | null>(null)
  useEffect(() => {
    let alive = true
    async function tick() {
      const [c, t, h] = await Promise.all([
        api.console().catch(() => null),
        api.tsInstances().catch(() => null),
        external.activity().catch(() => null),
      ])
      if (!alive) return
      if (c) setEngines(c.engines)
      if (t) setTs(t.instances)
      if (h) setHosted(h)
    }
    void tick()
    const timer = setInterval(tick, 1500)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])
  return { engines, ts, hosted }
}

type LedState = 'active' | 'ready' | 'loading' | 'off'

/** A status light. Active pulses with a glow so a working model is visible at a glance. */
function Led({ state }: { state: LedState }) {
  const color =
    state === 'active'
      ? 'bg-good'
      : state === 'ready'
        ? 'bg-good/35'
        : state === 'loading'
          ? 'bg-warn'
          : 'bg-ink-faint/40'
  return (
    <span className="relative inline-flex h-2.5 w-2.5 shrink-0">
      {state === 'active' && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-good opacity-60" />
      )}
      <span
        className={`relative inline-flex h-2.5 w-2.5 rounded-full ${color} ${
          state === 'active' ? 'shadow-[0_0_8px_2px_rgba(74,222,128,0.55)]' : ''
        } ${state === 'loading' ? 'animate-pulse' : ''}`}
      />
    </span>
  )
}

function LiveRow({
  name,
  state,
  right,
  detail,
  remote,
  paid,
  onOpen,
}: {
  name: string
  state: LedState
  right: React.ReactNode
  detail?: React.ReactNode
  /** Served by another computer: shown in the remote colour. */
  remote?: boolean
  /** A hosted, pay-per-token model: amber background, so spending is visible at a glance. */
  paid?: boolean
  /** Opens the agent inspector for this model (click, Enter or Space). */
  onOpen?: () => void
}) {
  const click = onOpen
    ? {
        role: 'button' as const,
        tabIndex: 0,
        title: `What ${name} is working on: its prompts, inputs and results`,
        onClick: onOpen,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            onOpen()
          }
        },
      }
    : {}
  const hover = onOpen
    ? ` cursor-pointer rounded-md transition-colors ${paid ? 'hover:bg-warn/20' : 'hover:bg-panel-hi'} focus-visible:outline focus-visible:outline-1 focus-visible:outline-accent/60`
    : ''
  return (
    <div
      {...click}
      className={
        (paid
          ? '-mx-1.5 my-0.5 rounded-md border border-warn/35 bg-warn/10 px-1.5 py-0.5'
          : onOpen
            ? '-mx-1 px-1 py-0.5'
            : 'py-0.5') + hover
      }
    >
      <div className="flex items-center gap-2 font-mono text-[11px]">
        <Led state={state} />
        <span className={`truncate ${paid ? 'text-warn' : remote ? 'text-remote' : state === 'active' ? 'text-ink' : 'text-ink-dim'}`}
          title={paid ? 'hosted model — every call costs money' : remote ? 'runs on another computer on the network' : undefined}>
          {paid && <span className="mr-1 font-bold">$</span>}{name}
        </span>
        <span className={`ml-auto shrink-0 ${state === 'active' ? 'text-good' : 'text-ink-faint'}`}>{right}</span>
      </div>
      {detail && <div className="ml-[18px] mt-0.5">{detail}</div>}
    </div>
  )
}

/** How full this engine's context (KV cache) is -- the resource that overflows first. */
function CtxBar({ used, total }: { used: number; total: number }) {
  const pct = total ? Math.min(100, (used / total) * 100) : 0
  const tone = pct > 85 ? 'bg-bad' : pct > 60 ? 'bg-warn' : 'bg-accent'
  return (
    <div className="flex items-center gap-2 font-mono text-[9.5px] text-ink-faint">
      <div className="h-1 flex-1 overflow-hidden rounded bg-seam">
        <div className={`h-full ${tone} transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-[96px] text-right">
        ctx {Math.round(pct)}% of {Math.round(total / 1024)}K
      </span>
    </div>
  )
}

/** HH:MM for today, otherwise the date -- when a hosted model last did something. */
function when(ts: number | null | undefined): string {
  if (!ts) return 'never'
  const d = new Date(ts * 1000)
  return d.toDateString() === new Date().toDateString()
    ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

/** Today's external spend against the limit, once above the hosted models. */
function SpendLine({ u }: { u: ExternalUsage }) {
  const over = u.search_budget_left <= 0
  return (
    <div className="mt-1.5 font-mono text-[10px] text-ink-faint" title={u.search_paused ?? undefined}>
      today <span className={u.today_usd >= u.limit_usd ? 'text-bad' : over ? 'text-warn' : 'text-ink-dim'}>{usd(u.today_usd)}</span> of{' '}
      {usd(u.limit_usd)}
      {u.ideas_reserve_usd > 0 && ` (${usd(u.ideas_reserve_usd)} held for ideas)`}
      {over && u.today_usd < u.limit_usd && ' · search paused until midnight'}
    </div>
  )
}

/** Under a hosted model's row: its role in this project, what it has cost and produced, and
 *  -- in red -- why it is being refused, if it is. */
function PaidUsage({ m, u }: { m: ModelUsage; u: ExternalUsage }) {
  const role = m.role
  const parts: string[] = []
  if (role.search) parts.push(`search ×${role.search.agents}`)
  else if (role.reserved?.budget_paused) parts.push('search paused')
  if (role.ideas) parts.push(`ideas (step ${role.ideas.rung + 1}${role.ideas.of > 1 ? ` of ${role.ideas.of}` : ''})`)
  if (!role.search && !role.ideas && !role.reserved?.budget_paused) parts.push(role.loaded ? 'reserved' : 'no API key')
  const roleText = parts.join(' + ')
  const c = m.candidates
  const cand = !c
    ? null
    : c.total === 0
      ? '0 candidates'
      : `${c.total} candidate${c.total === 1 ? '' : 's'} (${c.by_status.ok ?? 0} ok · ${c.by_status.error ?? 0} err · ${c.lookahead_pass} passed look-ahead${c.champions ? ` · ${c.champions} best` : ''})`
  const line = [roleText, `${m.calls_total.toLocaleString()} call${m.calls_total === 1 ? '' : 's'}`]
  if (m.usd_total > 0) line.push(`${usd(m.usd_total)} total`, `${usd(m.usd_today)} today`)
  // "0 candidates" is the point for a searcher (it spent and produced nothing); for an
  // ideas-only model it is noise.
  if (!role.ideas || role.search || c?.total) line.push(cand ?? '')
  if (role.ideas) line.push(`${m.ideas?.count ?? 0} idea${m.ideas?.count === 1 ? '' : 's'}`)
  line.push(`last${role.ideas && !role.search ? ': ' : ' '}${when(role.ideas && !role.search ? (m.ideas?.last_ts ?? m.last_call_ts) : m.last_call_ts)}`)
  const why = role.search?.why ?? role.reserved?.why ?? role.ideas?.why ?? undefined
  const ref = m.refusals
  const escErr = role.ideas ? u.escalation_errors.find((e) => e.model === m.model) : undefined
  return (
    <div className="space-y-0.5 font-mono text-[10px] leading-snug">
      <div className="text-ink-faint" title={why}>
        {line.filter(Boolean).join(' · ')}
      </div>
      {role.reserved?.budget_paused && role.reserved.why && <div className="text-warn">{role.reserved.why}</div>}
      {(m.throttles?.count ?? 0) > 0 && (
        <div className="text-ink-faint" title="Requests held back to stay inside the provider's per-minute rate limit, then sent -- not refused">
          rate-paced ×{m.throttles!.count} today ({Math.round(m.throttles!.seconds)}s waited), last {when(m.throttles!.ts)}
        </div>
      )}
      {ref.count > 0 && (
        <div className="text-bad" title={ref.reason ?? undefined}>
          {ref.kind === 'budget' ? 'blocked' : 'refused'} ×{ref.count} today, last {when(ref.ts)}: {ref.short}
        </div>
      )}
      {escErr && (
        <div className="text-bad" title={escErr.detail}>
          ideas for &ldquo;{escErr.title}&rdquo; failed {when(escErr.ts)}: {escErr.detail.slice(0, 160)}
        </div>
      )}
    </div>
  )
}

function Section({ label, count, children }: { label: string; count: number; children: React.ReactNode }) {
  return (
    <div className="border-t border-seam py-2.5 first:border-t-0 first:pt-0">
      <div className="mb-1.5 flex items-center justify-between font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">
        <span>{label}</span>
        <span>{count}</span>
      </div>
      <div className="space-y-1">{children}</div>
    </div>
  )
}

function Row({ name, right, tone }: { name: string; right: string; tone: 'good' | 'warn' | 'neutral' }) {
  const dot = tone === 'good' ? 'bg-good' : tone === 'warn' ? 'bg-warn' : 'bg-ink-faint'
  return (
    <div className="flex items-center gap-2 font-mono text-[11px]">
      <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${dot}`} />
      <span className="truncate text-ink">{name}</span>
      <span className="ml-auto shrink-0 text-ink-faint">{right}</span>
    </div>
  )
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="text-[11.5px] text-ink-faint">{children}</div>
}
