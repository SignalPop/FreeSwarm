'use client'

import { createContext, useContext } from 'react'

/**
 * Lets a sandbox run hand its output back to the conversation.
 *
 * Markdown (and so every code block inside it) is rendered deep inside the message list,
 * far from the composer, and is also used on pages that have no composer at all. A context
 * keeps the plumbing out of every intermediate component and makes "no chat to share with"
 * the natural default: when nothing provides it, the button simply does not appear.
 */
export type ShareToChat = (text: string) => void

export const ShareToChatContext = createContext<ShareToChat | null>(null)

export function useShareToChat(): ShareToChat | null {
  return useContext(ShareToChatContext)
}

/** Format a run for the model: what ran, what it printed, what it produced. */
export function formatRunForChat(opts: {
  ok: boolean
  exitCode: number | null
  timedOut: boolean
  stdout: string
  stderr: string
  artifacts: { name: string; mime: string; size: number }[]
}): string {
  const lines: string[] = []
  lines.push(
    opts.ok
      ? 'I ran that code in the sandbox. Output:'
      : opts.timedOut
        ? 'I ran that code in the sandbox and it hit the time limit. Output:'
        : `I ran that code in the sandbox and it failed (exit ${opts.exitCode}). Output:`,
  )
  if (opts.stdout.trim()) {
    lines.push('', 'stdout:', '```', opts.stdout.trim(), '```')
  }
  if (opts.stderr.trim()) {
    lines.push('', 'stderr:', '```', opts.stderr.trim(), '```')
  }
  if (!opts.stdout.trim() && !opts.stderr.trim()) {
    lines.push('', '(no output)')
  }
  if (opts.artifacts.length) {
    lines.push('', 'Files produced:')
    for (const a of opts.artifacts) {
      lines.push(`- ${a.name} (${a.mime}, ${a.size} bytes)`)
    }
  }
  return lines.join('\n')
}
