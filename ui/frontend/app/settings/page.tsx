'use client'

import { api, type SystemDoc } from '@/lib/api'
import { gib } from '@/lib/format'
import { usePoll } from '@/lib/usePoll'
import GpuSelector from '@/components/GpuSelector'
import ExternalReviewSettings from '@/components/ExternalReviewSettings'
import SwarmIntegritySettings from '@/components/SwarmIntegritySettings'
import { PageHeader, Panel, Pill } from '@/components/ui'

function Row({ label, value, tone }: { label: string; value: React.ReactNode; tone?: 'good' | 'bad' }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-seam/60 py-2.5 last:border-0">
      <span className="text-[13px] text-ink-dim">{label}</span>
      <span
        className={`max-w-[65%] break-all text-right font-mono text-[12px] ${
          tone === 'good' ? 'text-good' : tone === 'bad' ? 'text-bad' : 'text-ink'
        }`}
      >
        {value}
      </span>
    </div>
  )
}

export default function SettingsPage() {
  const { data, error } = usePoll<SystemDoc>(api.system, 5000)
  const tc = data?.toolchain
  const cfg = data?.config

  return (
    <div className="mx-auto max-w-[900px] px-8 py-8">
      <PageHeader
        title="Settings"
        subtitle="Environment this control plane resolved at startup"
        right={<Pill tone={error ? 'bad' : 'good'} pulse={!error}>{error ? 'Offline' : 'Connected'}</Pill>}
      />

      <ExternalReviewSettings />

      <SwarmIntegritySettings />

      <Panel className="mb-6 p-5">
        <div className="mb-2 text-[15px] font-medium text-ink">Build toolchain</div>
        <p className="mb-3 text-[12px] text-ink-faint">
          FreeToken JIT-compiles CUDA kernels on first use, so the engine needs both an MSVC
          host compiler and an nvcc whose CUDA major matches the torch wheel.
        </p>
        <Row
          label="MSVC (vcvars64.bat)"
          value={tc?.vcvars ?? 'not found'}
          tone={tc?.vcvars ? 'good' : 'bad'}
        />
        <Row label="CUDA home" value={tc?.cuda_home ?? 'not found'} tone={tc?.cuda_home ? 'good' : 'bad'} />
        <Row label="nvcc" value={tc?.nvcc ?? 'not found'} tone={tc?.nvcc ? 'good' : 'bad'} />
      </Panel>

      <Panel className="mb-6 p-5">
        <div className="mb-3 text-[15px] font-medium text-ink">Configuration</div>
        <Row label="Engine URL" value={cfg?.engine_url ?? '—'} />
        <Row
          label="CUDA_VISIBLE_DEVICES"
          value={cfg?.visible_devices || '(all GPUs)'}
        />
        <Row label="Engine interpreter" value={cfg?.python ?? '—'} />
        <Row
          label="Model roots"
          value={
            <span className="block space-y-0.5">
              {(cfg?.model_roots ?? []).map((r) => (
                <span key={r} className="block">
                  {r}
                </span>
              ))}
            </span>
          }
        />
      </Panel>

      <div className="mb-6">
        <GpuSelector />
      </div>

      <Panel className="mb-6 p-5">
        {/* GPU details live in the selector above; repeating them here only invited the
            two lists to disagree. */}
        <div className="mb-3 text-[15px] font-medium text-ink">Host</div>
        <Row
          label="GPUs in engine pool"
          value={`${(data?.gpus ?? []).filter((g) => g.enabled).length} of ${(data?.gpus ?? []).length}`}
        />
        <Row
          label="Host memory"
          value={`${gib(data?.host_memory.total_bytes)} GiB total · ${gib(data?.host_memory.available_bytes)} GiB free`}
        />
      </Panel>

      <Panel className="border-warn/30 bg-warn/[0.04] p-5">
        <div className="mb-2 text-[15px] font-medium text-ink">Network exposure</div>
        <p className="text-[12.5px] leading-relaxed text-ink-dim">
          The engine has <strong className="text-ink">no authentication of its own</strong> and
          always binds <code className="font-mono">127.0.0.1</code>. The control plane is the
          only front door, and it refuses to bind a non-loopback address unless a user account
          exists <em>and</em> TLS is configured.
        </p>
        <p className="mt-3 text-[12.5px] leading-relaxed text-ink-dim">
          To reach this box from another machine, prefer an SSH tunnel —{' '}
          <code className="font-mono">ssh -L 8000:127.0.0.1:8000 user@host</code> — which
          exposes nothing on the LAN. Create accounts with{' '}
          <code className="font-mono">python -m app.usercli add &lt;name&gt;</code>.
        </p>
      </Panel>
    </div>
  )
}
