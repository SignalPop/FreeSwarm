'use client'

import { streamChat } from '@/lib/api'

/**
 * The conversation, held OUTSIDE React so navigating away cannot destroy it.
 *
 * The chat page is a route component: opening Models or Console unmounts it. With the
 * messages and the streaming loop living in that component's state, leaving mid-generation
 * threw away the whole conversation and the half-written reply with it -- and the reply was
 * usually the expensive part, several minutes of a 67 GiB model's time.
 *
 * So the loop runs here, in module scope. It keeps writing into this store whether or not
 * anything is rendering; the page is a view over it and can come and go freely. Returning to
 * /chat re-subscribes and shows whatever arrived in the meantime, still streaming if it has
 * not finished.
 *
 * Scope: this survives navigation within the console, not a browser reload -- a reload drops
 * the JS heap, and the engine has no server-side session to reattach to.
 */

export type Msg = {
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  error?: string
  tps?: number
  /** Reasoning-channel token count, so a silent think phase still shows progress. */
  thinkTokens?: number
  /** Seconds spent reasoning before the first content token arrived. */
  thoughtFor?: number
  /** OpenAI finish_reason: 'length' means the token budget ran out mid-generation. */
  finish?: string
}

export type ChatState = {
  messages: Msg[]
  input: string
  streaming: boolean
  temperature: number
  maxTokens: number
  showReasoning: boolean
  /** Which model answers. `name@computer` is served by a paired computer over the network;
   *  null means "whatever the local engine is serving", the behaviour before there was a
   *  choice to make. */
  model: string | null
}

const INITIAL: ChatState = {
  messages: [],
  input: '',
  streaming: false,
  model: null,
  temperature: 0.7,
  // A thinking model spends its budget on reasoning FIRST. At 1024 a code request burns the
  // whole budget mid-thought and returns empty content (finish_reason 'length'), which reads
  // as an empty reply bubble. 4096 leaves room for the answer too.
  maxTokens: 4096,
  showReasoning: false,
}

let state: ChatState = INITIAL
const listeners = new Set<() => void>()
let controller: AbortController | null = null

function notify() {
  listeners.forEach((fn) => fn())
}

function emit(next: Partial<ChatState>) {
  // A new object per change: useSyncExternalStore compares snapshots by identity.
  state = { ...state, ...next }
  cancelFlush()
  notify()
}

// Streaming tokens are COALESCED: the state updates on every token, but subscribers are told
// at most once per FLUSH_MS. An engine delivers ~50 tokens/s, and each notification
// re-renders the whole transcript and re-parses the reply in progress -- which grows, so
// the per-token cost grows with it. Batching to ~16 renders/s is invisible to the eye and
// cuts that work by roughly two-thirds before any other optimisation.
const FLUSH_MS = 60
let flushTimer: ReturnType<typeof setTimeout> | null = null

function cancelFlush() {
  if (flushTimer !== null) {
    clearTimeout(flushTimer)
    flushTimer = null
  }
}

function scheduleFlush() {
  if (flushTimer !== null) return
  flushTimer = setTimeout(() => {
    flushTimer = null
    notify()
  }, FLUSH_MS)
}

/** Replace the last message (the assistant's in-flight reply) via an updater. Batched. */
function patchLast(update: (m: Msg) => Msg) {
  const messages = state.messages.slice()
  if (!messages.length) return
  messages[messages.length - 1] = update(messages[messages.length - 1])
  state = { ...state, messages }
  scheduleFlush()
}

export const chatStore = {
  subscribe(fn: () => void) {
    listeners.add(fn)
    return () => listeners.delete(fn)
  },
  getSnapshot: () => state,
  // Server render has no conversation; the same frozen object every time keeps
  // useSyncExternalStore from looping.
  getServerSnapshot: () => INITIAL,

  setInput: (input: string) => emit({ input }),
  setModel: (model: string | null) => emit({ model }),
  setTemperature: (temperature: number) => emit({ temperature }),
  setMaxTokens: (maxTokens: number) => emit({ maxTokens }),
  setShowReasoning: (showReasoning: boolean) => emit({ showReasoning }),

  appendToInput(text: string) {
    const prev = state.input
    emit({ input: prev.trim() ? `${prev.trimEnd()}\n\n${text}` : text })
  },

  clear() {
    controller?.abort()
    controller = null
    emit({ messages: [], streaming: false })
  },

  stop() {
    controller?.abort()
    controller = null
    emit({ streaming: false })
  },

  async send() {
    const text = state.input.trim()
    if (!text || state.streaming) return

    const history: Msg[] = [...state.messages, { role: 'user', content: text }]
    emit({
      messages: [...history, { role: 'assistant', content: '' }],
      input: '',
      streaming: true,
    })

    controller = new AbortController()
    const started = performance.now()
    let tokenish = 0

    try {
      for await (const chunk of streamChat(
        {
          messages: history.map((m) => ({ role: m.role, content: m.content })),
          model: state.model,
          temperature: state.temperature,
          max_tokens: state.maxTokens,
        },
        controller.signal,
      )) {
        patchLast((last) => {
          const next = { ...last }
          if (chunk.error) next.error = chunk.error
          if (chunk.finish) next.finish = chunk.finish
          if (chunk.delta) {
            next.content += chunk.delta
            tokenish += 1
          }
          if (chunk.reasoning) {
            // gpt-oss and the Qwen thinking models emit their whole chain of thought on
            // this channel BEFORE any content -- an eternity of blank bubble if nothing is
            // shown, which is why it is counted and surfaced.
            next.reasoning = (next.reasoning ?? '') + chunk.reasoning
            next.thinkTokens = (next.thinkTokens ?? 0) + 1
            next.thoughtFor = (performance.now() - started) / 1000
          }
          // Rough: counts SSE deltas, which are ~1 token each for this engine. An
          // at-a-glance rate, not a billing figure -- /v1/stats has the real number.
          const elapsed = (performance.now() - started) / 1000
          if (elapsed > 0.4) next.tps = tokenish / elapsed
          return next
        })
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        patchLast((last) => ({
          ...last,
          error: e instanceof Error ? e.message : String(e),
        }))
      }
    } finally {
      controller = null
      emit({ streaming: false })
    }
  },
}
