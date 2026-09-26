'use client'

import { useRouter } from 'next/navigation'
import { useState } from 'react'
import { api, type ConsoleDoc } from '@/lib/api'
import { compactTokens, duration, gib } from '@/lib/format'
import { usePoll } from '@/lib/usePoll'
import CachePanel from '@/components/CachePanel'
import EngineCard from '@/components/EngineCard'
import TokenUsagePanel from '@/components/TokenUsagePanel'
import { useUnloadAll } from '@/components/UnloadAll'
import { Button, EmptyState, Metric, PageHeader, Panel, Pill, StatCard } from '@/components/ui'

function stateTone(state: string) {
  if (state === 'running') return 'good' as const
  if (state === 'starting' || state === 'stopping') return 'warn' as const
  if (state === 'error') return 'bad' as const
  return 'neutral' as const
}

export default function ConsolePage() {
  const router = useRouter()
  const { data, error, refresh } = usePoll<ConsoleDoc>(api.console, 1000)
  const unload = useUnloadAll(data, refresh)
  const [busy, setBusy] = useState(false)

  const engines = data?.engines ?? []
  const gpus = data?.gpus ?? []
  // Models served by paired computers. They hold none of this machine's VRAM and have no
  // engine card, but they ARE resident capacity the way a local model is -- leaving them off
  // the Console made a connected peer invisible everywhere except the Network page.
  const remote = (data?.loaded ?? []).filter((l) => l.remote)

  // Totals are summed across every engine: with two models resident, "tokens processed"
  // meaning only one of them would be quietly wrong.
  const totals = engines.reduce(
    (acc, e) => {
      const s = e.stats
      if (!s) return acc
      return {
        tokensIn: acc.tokensIn + s.requests.prompt_tokens_total,
        tokensOut: acc.tokensOut + s.requests.completion_tokens_total,
        decode: acc.decode + s.throughput.decode_tps,
        prefill: acc.prefill + s.throughput.prefill_tps,
        active: acc.active + s.requests.active,
        completed: acc.completed + s.requests.completed,
        vram: acc.vram + s.vram_bytes,
      }
    },
    { tokensIn: 0, tokensOut: 0, decode: 0, prefill: 0, active: 0, completed: 0, vram: 0 },
  )

  const running = engines.filter((e) => e.state === 'running')
  // The cache panel is per-engine; show the primary's, since rebuilding is a single-engine
  // operation and picking one arbitrarily would be worse than picking the obvious one.
  const state = data?.engine.state ?? 'stopped'

  return (
    <div className="mx-auto max-w-[1180px] px-8 py-8">
      <PageHeader
        title="Console"
        subtitle={
          engines.length
            ? `${running.length} of ${engines.length} engine${engines.length === 1 ? '' : 's'} running · ${totals.completed} requests total`
            : 'No engine running'
        }
        right={
          <div className="flex items-center gap-3">
            <Pill tone={error ? 'bad' : 'neutral'} pulse={!error}>
              {error ? 'Disconnected' : 'Live · 1s refresh'}
            </Pill>
            {unload.button}
          </div>
        }
      />

      {unload.panel}

      {error && (
        <Panel className="mb-6 border-bad/35 bg-bad/5 p-4 text-[13px] text-bad">
          Control plane unreachable: {error}
          <div className="mt-1 text-ink-faint">
            Start it with <code className="font-mono">scripts\run-ui.bat</code>.
          </div>
        </Panel>
      )}

      {/* ---- Top stat row ---- */}
      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <StatCard
          label="Tokens processed"
          value={compactTokens(totals.tokensIn + totals.tokensOut)}
          sub={`${compactTokens(totals.tokensIn)} in · ${compactTokens(totals.tokensOut)} out · all engines`}
          accent
        />
        <StatCard
          label="Decode throughput"
          value={
            <>
              {totals.decode.toFixed(1)}
              <span className="ml-2 text-[15px] text-ink-faint">tok/s</span>
            </>
          }
          sub={`prefill ${totals.prefill.toFixed(0)} tok/s · combined`}
        />
        <StatCard
          label="Requests"
          value={totals.completed}
          sub={
            <>
              {totals.active} in flight · {gib(totals.vram)} GiB VRAM across{' '}
              {running.length} engine{running.length === 1 ? '' : 's'}
            </>
          }
        />
      </div>

      {/* ---- Engines ---- */}
      {engines.length > 0 ? (
        <div className="mb-6 space-y-4">
          <div className="flex flex-wrap items-baseline gap-3">
            <span className="text-[15px] font-medium text-ink">Engines</span>
            <span className="font-mono text-[11px] text-ink-faint">
              one per GPU — Windows cannot split a model across cards
            </span>
            <Button
              tone="ghost"
              className="ml-auto"
              onClick={() => router.push('/models')}
            >
              Load another model
            </Button>
          </div>
          {engines.map((e) => (
            <EngineCard
              key={e.instance_id}
              engine={e}
              onChanged={refresh}
              // Which cards the OTHER engines hold, so this one's Move menu cannot offer
              // a GPU that is already occupied.
              otherGpus={engines
                .filter((o) => o.instance_id !== e.instance_id && o.state === 'running')
                .map((o) => o.gpus)
                .filter((g): g is string => Boolean(g))}
            />
          ))}
        </div>
      ) : (
        <div className="mb-6">
          <EmptyState
            title="No engine running"
            hint={
              <>
                Pick a checkpoint on the{' '}
                <button
                  className="text-accent underline underline-offset-2"
                  onClick={() => router.push('/models')}
                >
                  Models
                </button>{' '}
                page to start serving. With two GPUs you can run two models at once.
              </>
            }
          />
        </div>
      )}

      {/* ---- Models on other computers ---- */}
      {remote.length > 0 && (
        <Panel className="mb-6 p-5">
          <div className="mb-3 flex flex-wrap items-baseline gap-3">
            <span className="text-[15px] font-medium text-ink">On the network</span>
            <span className="font-mono text-[11px] text-ink-faint">
              served by paired computers — usable in Chat, Projects and the Swarm
            </span>
            <Button tone="ghost" className="ml-auto" onClick={() => router.push('/network')}>
              Network
            </Button>
          </div>
          <div className="space-y-2">
            {remote.map((m) => (
              <div
                key={m.instance_id}
                className="flex flex-wrap items-center gap-3 rounded-xl border border-remote/30 bg-remote/[0.06] px-4 py-3"
              >
                <span className="h-2 w-2 shrink-0 rounded-full bg-remote" />
                <span className="font-mono text-[12.5px] text-remote">{m.model}</span>
                <Pill tone={m.ready ? 'good' : 'warn'}>{m.ready ? 'ready' : m.state}</Pill>
                <span className="ml-auto font-mono text-[11px] text-ink-faint">
                  on {m.remote?.node} · {m.remote?.address}
                  {m.context ? ` · ${Math.round(m.context / 1024)}K ctx` : ''}
                  {m.remote?.active ? ` · ${m.remote.active} active` : ''}
                  {m.remote?.decode_tps ? ` · ${m.remote.decode_tps.toFixed(0)} tok/s` : ''}
                </span>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {/* ---- Tokens processed per model: local, network and external ---- */}
      <TokenUsagePanel />

      {/* ---- Cache config ---- */}
      <CachePanel doc={data?.cache ?? null} running={state === 'running'} onApplied={refresh} />

      {/* ---- GPUs ---- */}
      <Panel className="mt-6 p-5">
        <div className="mb-4 flex flex-wrap items-baseline gap-3">
          <span className="text-[15px] font-medium text-ink">GPU pool</span>
          {(data?.gpus_hidden ?? 0) > 0 && (
            <span className="text-[12px] text-ink-faint">
              {data?.gpus_hidden} card{(data?.gpus_hidden ?? 0) === 1 ? '' : 's'} excluded ·
              change in Settings
            </span>
          )}
        </div>
        <div className="space-y-3">
          {gpus.map((g) => {
            const pctUsed =
              g.memory_total_bytes > 0 ? (g.memory_used_bytes / g.memory_total_bytes) * 100 : 0
            return (
              <div key={g.index} className="rounded-xl border border-seam bg-panel-hi/40 p-4">
                <div className="flex flex-wrap items-baseline justify-between gap-3">
                  <span className="text-[13px] text-ink">
                    <span className="font-mono text-ink-faint">[{g.index}]</span> {g.name}
                    <span className="ml-2 font-mono text-[11px] text-ink-faint">
                      sm_{g.compute_cap.replace('.', '')}
                    </span>
                  </span>
                  <span className="font-mono text-[12px] text-ink-dim">
                    {gib(g.memory_used_bytes)} / {gib(g.memory_total_bytes)} GiB
                    {g.utilization_pct !== null && ` · ${g.utilization_pct.toFixed(0)}%`}
                    {g.temperature_c !== null && ` · ${g.temperature_c.toFixed(0)}°C`}
                    {g.power_draw_w !== null && ` · ${g.power_draw_w.toFixed(0)} W`}
                  </span>
                </div>
                <div className="mt-2.5 h-1 w-full overflow-hidden rounded-full bg-seam">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      pctUsed > 90 ? 'bg-bad' : pctUsed > 66 ? 'bg-warn' : 'bg-accent'
                    }`}
                    style={{ width: `${pctUsed}%` }}
                  />
                </div>
              </div>
            )
          })}
          {gpus.length === 0 && (
            <div className="text-[13px] text-ink-faint">
              {(data?.gpus_hidden ?? 0) > 0
                ? 'Every GPU is excluded from the pool — enable one in Settings.'
                : 'No GPUs reported — is nvidia-smi on PATH?'}
            </div>
          )}
        </div>
      </Panel>
    </div>
  )
}
