'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { projects, projectResources as res, type Project, type SwarmResources } from '@/lib/projects'
import { api, type EngineDetail, type TsInstance } from '@/lib/api'
import { bytesLabel } from '@/lib/format'
import { Panel, Pill } from '@/components/ui'

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

  useEffect(() => {
    let alive = true
    async function load() {
      try {
        const list = await projects.list()
        if (!list.active) return
        const data = await res.swarmResources(list.active)
        if (alive) {
          setR(data)
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

  if (err) return <Panel className="p-4 text-[12px] text-bad">{err}</Panel>
  if (!r) return null

  const agents = r.llms.filter((l) => l.loaded)
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
        {agents.map((l) => {
          const e = live.engines.find((x) => x.model_id === l.model)
          const st = e?.stats
          // A model on another computer has no local engine: its numbers come with the list.
          const active = st?.requests.active ?? l.remote?.active ?? 0
          const tps = st?.throughput.decode_tps ?? (l.remote && active ? l.remote.decode_tps ?? 0 : 0)
          const prefill = st?.throughput.prefill_tps ?? 0
          const state: LedState = !l.ready ? 'loading' : active > 0 ? 'active' : 'ready'
          const rate =
            tps >= 0.5 ? `${tps.toFixed(1)} tok/s` : prefill >= 1 ? `prefill ${Math.round(prefill)} tok/s` : 'thinking'
          return (
            <LiveRow
              key={l.model}
              name={l.model}
              remote={!!l.remote}
              state={state}
              right={
                state === 'active'
                  ? `${rate} · ${active} req`
                  : state === 'loading'
                    ? 'loading'
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
                ) : undefined
              }
            />
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
    </Panel>
  )
}

/** Live activity: engine stats for LLMs, call counters for forecasters. Polled fast and
 *  separately from the resource list, which walks the data folder and must stay slow. */
function useLive() {
  const [engines, setEngines] = useState<EngineDetail[]>([])
  const [ts, setTs] = useState<TsInstance[]>([])
  useEffect(() => {
    let alive = true
    async function tick() {
      const [c, t] = await Promise.all([
        api.console().catch(() => null),
        api.tsInstances().catch(() => null),
      ])
      if (!alive) return
      if (c) setEngines(c.engines)
      if (t) setTs(t.instances)
    }
    void tick()
    const timer = setInterval(tick, 1500)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])
  return { engines, ts }
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
}: {
  name: string
  state: LedState
  right: React.ReactNode
  detail?: React.ReactNode
  /** Served by another computer: shown in the remote colour. */
  remote?: boolean
}) {
  return (
    <div className="py-0.5">
      <div className="flex items-center gap-2 font-mono text-[11px]">
        <Led state={state} />
        <span className={`truncate ${remote ? 'text-remote' : state === 'active' ? 'text-ink' : 'text-ink-dim'}`}
          title={remote ? 'runs on another computer on the network' : undefined}>{name}</span>
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
