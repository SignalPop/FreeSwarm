'use client'

import { useState } from 'react'

/**
 * Copy-to-clipboard with the familiar double-box glyph, animated into a check on success.
 *
 * The two boxes slide together and fade as the check draws itself in, so the confirmation is
 * visible where the user is already looking -- no toast to hunt for.
 */
export default function CopyButton({
  text,
  label = 'copy',
  className = '',
}: {
  /** The text to copy, or a function producing it (computed on click, not every render). */
  text: string | (() => string)
  label?: string
  className?: string
}) {
  const [state, setState] = useState<'idle' | 'done' | 'failed'>('idle')

  async function copy(e: React.MouseEvent) {
    e.stopPropagation()
    const value = typeof text === 'function' ? text() : text
    try {
      await navigator.clipboard.writeText(value)
      setState('done')
    } catch {
      // Clipboard API is blocked on non-secure origins; fall back to a hidden textarea,
      // which still works on http://localhost-style LAN access.
      try {
        const ta = document.createElement('textarea')
        ta.value = value
        ta.style.position = 'fixed'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.select()
        const ok = document.execCommand('copy')
        document.body.removeChild(ta)
        setState(ok ? 'done' : 'failed')
      } catch {
        setState('failed')
      }
    }
    setTimeout(() => setState('idle'), 1600)
  }

  const done = state === 'done'
  return (
    <button
      onClick={copy}
      title={done ? 'copied' : state === 'failed' ? 'copy failed' : label}
      aria-label={label}
      className={`group/copy inline-flex h-6 w-6 items-center justify-center rounded-md text-ink-faint transition-colors hover:bg-panel-hi hover:text-accent ${
        done ? 'text-good' : ''
      } ${state === 'failed' ? 'text-bad' : ''} ${className}`}
    >
      <svg viewBox="0 0 24 24" className="h-[15px] w-[15px]" fill="none" aria-hidden>
        {/* back box: slides forward onto the front one, then fades */}
        <rect
          x="8.5"
          y="8.5"
          width="11"
          height="11"
          rx="2"
          stroke="currentColor"
          strokeWidth="1.8"
          style={{
            transition: 'transform 220ms ease, opacity 220ms ease',
            transform: done ? 'translate(-2.5px,-2.5px)' : 'none',
            opacity: done ? 0 : 1,
          }}
        />
        {/* front box: the "second copy" -- its open corner reads as a stacked sheet */}
        <path
          d="M15.5 8.5V6.5a2 2 0 0 0-2-2h-7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          style={{
            transition: 'transform 220ms ease, opacity 220ms ease',
            transform: done ? 'translate(2.5px,2.5px)' : 'none',
            opacity: done ? 0 : 1,
          }}
        />
        {/* check: drawn in with a stroke-dash sweep */}
        <path
          d="M5.5 12.5l4 4 9-9"
          stroke="currentColor"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{
            strokeDasharray: 22,
            strokeDashoffset: done ? 0 : 22,
            transition: 'stroke-dashoffset 320ms ease 120ms',
          }}
        />
      </svg>
    </button>
  )
}
