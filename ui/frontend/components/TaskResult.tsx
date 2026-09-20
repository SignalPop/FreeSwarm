'use client'

import { useEffect, useState } from 'react'
import { board, type BoardMessage, type Task } from '@/lib/board'
import { duration } from '@/lib/format'
import Markdown from '@/components/Markdown'
import { chatStore } from '@/lib/chatStore'
import { Pill } from '@/components/ui'

/**
 * A finished task's output, in a panel over the Swarm page.
 *
 * The result was previously write-only: an agent did minutes of work, wrote it to the
 * board, and the console showed a "done" chip with no way to read what was produced.
 *
 * It renders through the same Markdown component the chat uses, so a task that returned
 * Python arrives syntax-highlighted with a working Run button -- the result is usually code,
 * and having to copy it into the chat to run it would be the obvious next annoyance.
 *
 * "Continue in chat" hands the whole thing to the chat composer, which is the other thing
 * you want after reading a result: iterate on it with a model.
 */
export default function TaskResult({
  task,
  onClose,
  onContinueInChat,
}: {
  task: Task
  onClose: () => void
  onContinueInChat: () => void
}) {
  // Escape closes: this is a transient overlay, and reaching for the mouse to dismiss a
  // reading pane is a papercut.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const took = task.updated_at - task.created_at
  const [tab, setTab] = useState<'result' | 'board'>('result')
  const [thread, setThread] = useState<BoardMessage[] | null>(null)
  const [threadErr, setThreadErr] = useState<string | null>(null)

  // The task's own message thread. Fetched from ITS session, not the live one, so an older
  // task still shows what happened while it ran. Messages are matched on meta.task_id (the
  // swarm runner tags everything it posts) plus the operator directive that queued it.
  useEffect(() => {
    let alive = true
    const load = task.session_id
      ? board.session(task.session_id).then((d) => d.messages)
      : board.messages(0).then((d) => d.entries)
    load
      .then((msgs) => {
        if (!alive) return
        setThread(
          msgs.filter(
            (m) =>
              (m.meta as { task_id?: string })?.task_id === task.id ||
              (m.kind === 'directive' && m.content.trim().startsWith(task.title.trim())),
          ),
        )
      })
      .catch((e) => alive && setThreadErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [task.id, task.session_id, task.title])

  const [retrying, setRetrying] = useState(false)

  async function retry() {
    setRetrying(true)
    try {
      await board.retryTask(task)
      onClose()
    } finally {
      setRetrying(false)
    }
  }

  function continueInChat() {
    chatStore.appendToInput(
      `Task: ${task.title}\n\nThe swarm produced this result:\n\n${task.result ?? '(no result)'}\n\n`,
    )
    onContinueInChat()
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="flex max-h-[85vh] w-full max-w-[900px] flex-col overflow-hidden rounded-2xl border border-seam bg-panel shadow-2xl"
        // The backdrop closes; a click inside must not bubble up to it.
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3 border-b border-seam px-5 py-4">
          <div className="min-w-0 flex-1">
            <div className="text-[15px] font-medium leading-snug text-ink">{task.title}</div>
            <div className="mt-1.5 flex flex-wrap items-center gap-2 font-mono text-[10.5px] text-ink-faint">
              <Pill tone={task.status === 'done' ? 'good' : 'bad'}>{task.status}</Pill>
              {task.claimed_by && <span>by {task.claimed_by.slice(0, 8)}</span>}
              {took > 0 && <span>took {duration(took)}</span>}
              {task.attempts > 1 && <span>attempt {task.attempts}</span>}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-3">
            {task.status === 'failed' && (
              <button
                onClick={retry}
                disabled={retrying}
                className="rounded-md border border-warn/50 px-2.5 py-1 font-mono text-[11px] text-warn transition-colors hover:bg-warn/10 disabled:opacity-60"
              >
                {retrying ? 'queuing…' : 'retry'}
              </button>
            )}
            <button
              onClick={continueInChat}
              className="rounded-md border border-accent/40 px-2.5 py-1 font-mono text-[11px] text-accent transition-colors hover:bg-accent/10"
            >
              continue in chat
            </button>
            <button
              onClick={onClose}
              className="font-mono text-[11px] text-ink-faint transition-colors hover:text-accent"
            >
              close
            </button>
          </div>
        </div>

        <div className="flex gap-1 border-b border-seam px-5">
          {(['result', 'board'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`-mb-px border-b-2 px-3 py-2 font-mono text-[11px] transition-colors ${
                tab === t
                  ? 'border-accent text-accent'
                  : 'border-transparent text-ink-faint hover:text-ink-dim'
              }`}
            >
              {t === 'result' ? 'result' : `board activity${thread ? ` (${thread.length})` : ''}`}
            </button>
          ))}
        </div>

        {tab === 'board' && (
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
            {threadErr && <div className="font-mono text-[12px] text-bad">{threadErr}</div>}
            {!threadErr && thread === null && (
              <div className="font-mono text-[12px] text-ink-faint">loading…</div>
            )}
            {thread?.length === 0 && (
              <div className="font-mono text-[12px] text-ink-faint">
                No board messages were tagged with this task.
              </div>
            )}
            <ol className="space-y-3">
              {thread?.map((m) => (
                <li key={m.seq} className="rounded-xl border border-seam bg-panel-hi/40 p-3">
                  <div className="mb-1.5 flex flex-wrap items-center gap-2 font-mono text-[10.5px] text-ink-faint">
                    <span className="text-ink">{m.author}</span>
                    <Pill
                      tone={
                        m.kind === 'error' ? 'bad' : m.kind === 'result' ? 'good' : 'neutral'
                      }
                    >
                      {m.kind}
                    </Pill>
                    <span>#{m.channel}</span>
                    <span className="ml-auto">{new Date(m.ts * 1000).toLocaleTimeString()}</span>
                  </div>
                  {m.kind === 'result' || m.kind === 'error' ? (
                    <Markdown source={m.content} />
                  ) : (
                    <pre className="whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-ink-dim">
                      {m.content}
                    </pre>
                  )}
                </li>
              ))}
            </ol>
          </div>
        )}

        <div className={`min-h-0 flex-1 overflow-y-auto px-5 py-4 ${tab === 'result' ? '' : 'hidden'}`}>
          {task.description && (
            <details className="mb-4">
              <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-wider text-ink-faint">
                original request
              </summary>
              <pre className="mt-2 whitespace-pre-wrap font-mono text-[11.5px] leading-relaxed text-ink-dim">
                {task.description}
              </pre>
            </details>
          )}

          {task.result ? (
            <Markdown source={task.result} />
          ) : (
            <div className="font-mono text-[12px] text-ink-faint">
              {task.status === 'done'
                ? 'Finished, but the agent returned no result text.'
                : 'No result yet.'}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
