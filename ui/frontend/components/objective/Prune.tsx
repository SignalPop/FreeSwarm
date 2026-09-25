'use client'

import type { ReactNode } from 'react'

/**
 * Small controls for deleting polluted records from the objective panel's tabs (candidates,
 * library modules, lessons, steering notes, ideas). Shared here rather than in
 * ObjectivePanel so the tabs it imports can use them without a circular import.
 */

/** The row checkbox. Swallows the click so ticking a row never also opens it. */
export function PickBox({
  checked,
  onChange,
  indeterminate = false,
  title,
}: {
  checked: boolean
  onChange: (on: boolean) => void
  indeterminate?: boolean
  title?: string
}) {
  return (
    <input
      type="checkbox"
      title={title}
      checked={checked}
      ref={(el) => {
        if (el) el.indeterminate = indeterminate
      }}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => onChange(e.target.checked)}
      className="h-3 w-3 cursor-pointer align-middle accent-accent"
    />
  )
}

/** The per-item × (lessons, steering notes, ideas). Faint until hovered, so it stays out of the reading. */
export function RowDelete({ onClick, title }: { onClick: () => void; title: string }) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
      title={title}
      className="ml-1.5 font-mono text-[12px] leading-none text-ink-faint/50 hover:text-bad group-hover:text-ink-faint"
    >
      ×
    </button>
  )
}

/** A subtle right-aligned "Delete all…" link heading a list. */
export function DeleteAllRow({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <li className="flex justify-end list-none">
      <DangerLink onClick={onClick}>{label}</DangerLink>
    </li>
  )
}

export function DangerLink({ onClick, children, disabled = false }: { onClick: () => void; children: ReactNode; disabled?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="font-mono text-[10.5px] text-ink-faint hover:text-bad disabled:opacity-40"
    >
      {children}
    </button>
  )
}

/** The compact red action button of a selection toolbar ("Delete 3 selected"). */
export function DangerButton({ onClick, children, disabled = false }: { onClick: () => void; children: ReactNode; disabled?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded-md border border-bad/40 bg-bad/10 px-2 py-0.5 font-mono text-[10.5px] text-bad hover:bg-bad/20 disabled:opacity-40"
    >
      {children}
    </button>
  )
}
