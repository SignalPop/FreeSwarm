'use client'

import { api, type SandboxRun } from '@/lib/api'

/**
 * Sandbox runs, held outside React so navigating away does not throw the result away.
 *
 * A code block lives inside the chat transcript, so it unmounts whenever you open Models or
 * Swarm. With the run state in the component, leaving mid-run meant the container kept
 * working, finished, and delivered its chart and files to a component that no longer
 * existed -- coming back showed an idle button, as if nothing had happened.
 *
 * Keyed by the code text itself: the same snippet is the same run, which also means an
 * identical block in two places shares one result rather than executing twice.
 */

export type RunEntry = {
  status: 'running' | 'done' | 'error'
  run?: SandboxRun
  error?: string
  startedAt: number
}

const entries = new Map<string, RunEntry>()
const listeners = new Set<() => void>()

function emit() {
  listeners.forEach((fn) => fn())
}

function set(key: string, entry: RunEntry) {
  entries.set(key, entry)
  emit()
}

export const sandboxRuns = {
  subscribe(fn: () => void) {
    listeners.add(fn)
    return () => listeners.delete(fn)
  },

  /** The entry for one snippet, or undefined. Identity is stable between updates, which is
   *  what useSyncExternalStore needs to avoid re-rendering forever. */
  get(key: string): RunEntry | undefined {
    return entries.get(key)
  },

  /** Start a run, unless this snippet is already running. Safe to call from a click. */
  start(key: string) {
    const existing = entries.get(key)
    if (existing?.status === 'running') return

    set(key, { status: 'running', startedAt: Date.now() })
    api
      .runInSandbox(key)
      .then((run) => set(key, { status: 'done', run, startedAt: Date.now() }))
      .catch((e) =>
        set(key, {
          status: 'error',
          error: e instanceof Error ? e.message : String(e),
          startedAt: Date.now(),
        }),
      )
  },

  /** Drop a result so the block goes back to just showing its Run button. */
  clear(key: string) {
    entries.delete(key)
    emit()
  },
}
