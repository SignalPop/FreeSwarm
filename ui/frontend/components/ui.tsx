'use client'

import type { ReactNode } from 'react'

export function Panel({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`rounded-2xl border border-seam bg-panel ${className}`}>{children}</section>
  )
}

export function StatCard({
  label,
  value,
  sub,
  accent = false,
  right,
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  accent?: boolean
  right?: ReactNode
}) {
  return (
    <Panel
      className={`p-5 ${accent ? 'border-good/35 bg-gradient-to-br from-good/[0.07] to-transparent' : ''}`}
    >
      <div className="flex items-start justify-between gap-3">
        <span className="text-[13px] text-ink-dim">{label}</span>
        {right}
      </div>
      <div
        className={`mt-3 font-mono text-[34px] leading-none tracking-tight ${accent ? 'text-good' : 'text-ink'}`}
      >
        {value}
      </div>
      {sub && <div className="mt-3 text-[12px] text-ink-faint">{sub}</div>}
    </Panel>
  )
}

export function Pill({
  tone = 'neutral',
  children,
  pulse = false,
}: {
  tone?: 'good' | 'warn' | 'bad' | 'accent' | 'neutral'
  children: ReactNode
  pulse?: boolean
}) {
  const tones = {
    good: 'border-good/35 bg-good/10 text-good',
    warn: 'border-warn/35 bg-warn/10 text-warn',
    bad: 'border-bad/35 bg-bad/10 text-bad',
    accent: 'border-accent/35 bg-accent/10 text-accent',
    neutral: 'border-seam bg-panel-hi text-ink-dim',
  }[tone]
  const dot = {
    good: 'bg-good',
    warn: 'bg-warn',
    bad: 'bg-bad',
    accent: 'bg-accent',
    neutral: 'bg-ink-faint',
  }[tone]
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[12px] ${tones}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${dot} ${pulse ? 'animate-dot' : ''}`} />
      {children}
    </span>
  )
}

export function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="text-right">
      <div className="text-[10px] uppercase tracking-[0.14em] text-ink-faint">{label}</div>
      <div className="mt-1 font-mono text-[22px] leading-none text-ink">{value}</div>
    </div>
  )
}

export function Button({
  children,
  onClick,
  tone = 'default',
  disabled = false,
  type = 'button',
  className = '',
}: {
  children: ReactNode
  onClick?: () => void
  tone?: 'default' | 'primary' | 'danger' | 'ghost'
  disabled?: boolean
  type?: 'button' | 'submit'
  className?: string
}) {
  const tones = {
    default: 'bg-ink text-canvas hover:opacity-90',
    primary: 'bg-accent text-white hover:opacity-90',
    danger: 'border border-bad/40 bg-bad/10 text-bad hover:bg-bad/20',
    ghost: 'border border-seam text-ink-dim hover:bg-panel-hi hover:text-ink',
  }[tone]
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-xl px-4 py-2 text-[13px] font-medium transition-all disabled:cursor-not-allowed disabled:opacity-40 ${tones} ${className}`}
    >
      {children}
    </button>
  )
}

export function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string
  subtitle?: ReactNode
  right?: ReactNode
}) {
  return (
    <div className="mb-7 flex items-start justify-between gap-6">
      <div>
        <h1 className="text-[32px] font-semibold tracking-tight text-ink">{title}</h1>
        {subtitle && <p className="mt-1.5 text-[13px] text-ink-dim">{subtitle}</p>}
      </div>
      {right}
    </div>
  )
}

export function EmptyState({ title, hint }: { title: string; hint?: ReactNode }) {
  return (
    <Panel className="grid place-items-center px-6 py-16 text-center">
      <div className="text-[15px] text-ink-dim">{title}</div>
      {hint && <div className="mt-2 max-w-md text-[13px] text-ink-faint">{hint}</div>}
    </Panel>
  )
}
