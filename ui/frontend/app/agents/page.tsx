'use client'

import { useEffect, useRef, useState } from 'react'
import { board, type BoardMessage, type BoardSummary, type Session, type Task } from '@/lib/board'
import { clockTime, duration } from '@/lib/format'
import { usePoll } from '@/lib/usePoll'
import { Button, PageHeader, Panel, Pill } from '@/components/ui'
import { useRouter } from 'next/navigation'
import TaskResult from '@/components/TaskResult'
import CopyButton from '@/components/CopyButton'
import { SwarmProjectBar, SwarmResourcesPanel } from '@/components/SwarmProject'
import TeamPanel from '@/components/TeamPanel'
import ObjectivePanel, { useObjectives } from '@/components/objective/ObjectivePanel'
import NewObjective from '@/components/objective/NewObjective'
import { objectives } from '@/lib/objectives'
import { projects } from '@/lib/projects'
import { isExternal } from '@/lib/external'

/** What the composer does with a message. With an objective selected, a message STEERS it:
 *  every agent reads it at the start of its next iteration. A one-off task is the old
 *  behaviour, still there for questions that are not part of the objective. */
type Mode = 'steer' | 'task' | 'post'

const KIND_TONE: Record<BoardMessage['kind'], 'good' | 'warn' | 'bad' | 'accent' | 'neutral'> = {
  directive: 'accent',
  result: 'good',
  error: 'bad',
  thought: 'neutral',
  system: 'warn',
  chat: 'neutral',
}

const TIER_TONE: Record<Task['tier'], 'good' | 'warn' | 'bad' | 'accent' | 'neutral'> = {
  small: 'good',
  mid: 'accent',
  hard: 'warn',
  auto: 'neutral',
}

function StatusTone(status: Task['status']) {
  if (status === 'done') return 'good' as const
  if (status === 'failed') return 'bad' as const
  if (status === 'claimed') return 'accent' as const
  return 'neutral' as const
}

export default function AgentsPage() {
  const [messages, setMessages] = useState<BoardMessage[]>([])
  const [channel, setChannel] = useState('general')
  const [input, setInput] = useState('')
  const [tier, setTier] = useState<Task['tier']>('auto')
  // null = automatic: steer when there is a live objective, otherwise queue a task.
  const [modeChoice, setModeChoice] = useState<Mode | null>(null)
  const [projectId, setProjectId] = useState<string | null>(null)
  const [newObjective, setNewObjective] = useState<string | null>(null)
  const [posting, setPosting] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  // null = follow the live session; a string = review that historical one (read-only).
  const [viewing, setViewing] = useState<string | null>(null)

  const feedRef = useRef<HTMLDivElement>(null)
  // The read cursor is a ref, not state, and is advanced synchronously from each
  // response. It was previously mirrored from state during render -- but the polling
  // loop iterates immediately after setCursor, before React re-renders, so it re-read
  // the OLD cursor, refetched the same rows, and appended them again. That is what
  // produced duplicate React keys and a hot loop hammering the board.
  const cursorRef = useRef(0)

  const { data: summary, error: summaryErr } = usePoll<BoardSummary>(board.summary, 2000)
  const { data: sessionDoc, refresh: refreshSessions } = usePoll<{ sessions: Session[] }>(
    board.sessions,
    8000,
  )

  useEffect(() => {
    if (sessionDoc) setSessions(sessionDoc.sessions)
  }, [sessionDoc])
  const router = useRouter()

  useEffect(() => {
    projects
      .list()
      .then((r) => setProjectId(r.active))
      .catch(() => setProjectId(null))
  }, [])
  const obj = useObjectives(projectId)
  const liveObjective = obj.detail && obj.detail.status !== 'stopped' ? obj.detail : null
  const mode: Mode =
    modeChoice === 'steer' && !liveObjective ? 'task' : modeChoice ?? (liveObjective ? 'steer' : 'task')
  // Which finished task is being read. Held by id, not by value, so the 2s poll keeps the
  // open panel fresh instead of pinning it to a stale snapshot.
  const [openTaskId, setOpenTaskId] = useState<string | null>(null)

  const { data: taskDoc, refresh: refreshTasks } = usePoll<{ tasks: Task[] }>(
    () => board.tasks(),
    2000,
  )
  const openTask = (taskDoc?.tasks ?? []).find((t) => t.id === openTaskId) ?? null

  // Long-poll the message log. `wait=25` means an idle swarm costs one parked connection
  // rather than a request per second, and a new message appears within ~400 ms.
  useEffect(() => {
    let cancelled = false

    // Reviewing a past session is a one-shot read: it is finished, so long-polling it
    // would park a connection waiting for messages that can never arrive.
    if (viewing) {
      board
        .session(viewing)
        .then((d) => {
          if (!cancelled) {
            setMessages(d.messages.filter((m) => m.channel === channel))
            setErr(null)
          }
        })
        .catch((e) => !cancelled && setErr(e instanceof Error ? e.message : String(e)))
      return () => {
        cancelled = true
      }
    }

    async function loop() {
      while (!cancelled) {
        try {
          const since = cursorRef.current
          const res = await board.messages(since, channel, 25)
          if (cancelled) return

          if (res.entries.length) {
            // Advance BEFORE the state update: the next loop iteration reads this ref
            // and must not see a stale value.
            cursorRef.current = Math.max(
              res.next_cursor,
              ...res.entries.map((e) => e.seq),
            )
            setMessages((prev) => {
              // Defensive de-duplication. The cursor fix removes the systematic cause,
              // but a reconnect or an overlapping fetch can still resend a row, and a
              // duplicate seq would break React's keying.
              const seen = new Set(prev.map((m) => m.seq))
              const fresh = res.entries.filter((e) => !seen.has(e.seq))
              return fresh.length ? [...prev, ...fresh].slice(-500) : prev
            })

            // If the server returned rows but the cursor did not move, looping again
            // would spin at full speed forever. Stop rather than hammer the board.
            if (cursorRef.current <= since) {
              setErr('message cursor did not advance; stopping the live feed')
              return
            }
          }
          setErr(null)
        } catch (e) {
          if (cancelled) return
          setErr(e instanceof Error ? e.message : String(e))
          // Back off so a downed board does not spin the browser.
          await new Promise((r) => setTimeout(r, 3000))
        }
      }
    }

    setMessages([])
    cursorRef.current = 0
    loop()
    return () => {
      cancelled = true
    }
  }, [channel, viewing])

  // Newest first: the feed reads top-down from the latest message, right under the composer.
  // No auto-scroll -- at the top, new messages simply appear; scrolled down reading older
  // ones, the browser's scroll anchoring keeps the view still as messages arrive above.
  const newestFirst = [...messages].reverse()

  async function submit() {
    const text = input.trim()
    if (!text || posting) return
    setPosting(true)
    setErr(null)
    try {
      if (mode === 'steer' && liveObjective) {
        await objectives.steer(liveObjective.id, text)
        obj.refresh()
      } else if (mode === 'task') {
        // A directive becomes a queued task an agent can claim, and is also posted to the
        // channel so the feed shows why work appeared.
        const [title, ...rest] = text.split('\n')
        await board.createTask({ title: title.slice(0, 300), description: rest.join('\n'), tier })
        refreshTasks()
      }
      await board.post({
        channel,
        author: 'operator',
        kind: mode === 'post' ? 'chat' : 'directive',
        content: mode === 'steer' && liveObjective ? `Steering "${liveObjective.title}": ${text}` : text,
        meta: mode === 'steer' && liveObjective ? { objective_id: liveObjective.id } : mode === 'task' ? { tier } : {},
      })
      setInput('')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setPosting(false)
    }
  }

  async function startNewSession() {
    const title = window.prompt('Name this session (blank = timestamp)') ?? ''
    try {
      await board.newSession(title.trim())
      setViewing(null)
      refreshSessions()
      // Force the live feed to re-read from the start of the new session.
      setMessages([])
      cursorRef.current = 0
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const readOnly = viewing !== null
  const agents = summary?.agents ?? []
  const online = agents.filter((a) => a.online)
  const counts = summary?.task_counts ?? {}
  const tasks = taskDoc?.tasks ?? []

  return (
    <div className="mx-auto max-w-[1180px] px-8 py-8">
      <PageHeader
        title="Swarm"
        subtitle={
          summaryErr
            ? 'Message board unreachable'
            : `${online.length} of ${agents.length} agents online · ${summary?.message_total ?? 0} messages · ${counts.open ?? 0} open tasks`
        }
        right={
          <Pill tone={summaryErr ? 'bad' : 'good'} pulse={!summaryErr}>
            {summaryErr ? 'Offline' : 'Live'}
          </Pill>
        }
      />

      <SwarmProjectBar />

      {summaryErr && (
        <Panel className="mb-6 border-bad/35 bg-bad/5 p-4 text-[13px] text-bad">
          {summaryErr}
          <div className="mt-1 text-ink-faint">
            Start it with <code className="font-mono">scripts\run-msgboard.bat</code>.
          </div>
        </Panel>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_330px]">
        {/* ---- Objective + feed + composer ---- */}
        <div className="flex min-h-[560px] min-w-0 flex-col">
          <ObjectivePanel state={obj} onNew={() => setNewObjective('')} />

          {/* Composer first, feed right under it: steer and watch the result in one place. */}
          <Panel className="mb-3 p-3">
            <div className="mb-2 flex flex-wrap items-center gap-2 border-b border-seam pb-2">
              <span className="text-[11.5px] text-ink-dim">Session</span>
              <select
              className="max-w-[300px] rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[11.5px] text-ink outline-none focus:border-accent"
              value={viewing ?? ''}
              onChange={(e) => setViewing(e.target.value || null)}
            >
              <option value="">Live — current session</option>
              {sessions
                .filter((x) => !x.active)
                .map((x) => (
                  <option key={x.id} value={x.id}>
                    {x.title} · {x.message_count} msgs · {x.tasks_done}/{x.task_count} done
                  </option>
                ))}
            </select>
              {readOnly && <Pill tone="warn">history · read-only</Pill>}
              <span className="ml-2 text-[11.5px] text-ink-dim">Channel</span>
              {['general', 'planning', 'team', 'results', 'errors'].map((c) => (
                <button
                  key={c}
                  onClick={() => setChannel(c)}
                  className={`rounded-full px-2.5 py-0.5 font-mono text-[11px] transition-colors ${
                    channel === c
                      ? 'bg-accent/15 text-accent'
                      : 'text-ink-faint hover:bg-panel-hi hover:text-ink-dim'
                  }`}
                >
                  #{c}
                </button>
              ))}
              <Button tone="ghost" className="ml-auto" onClick={startNewSession}>
                New session
              </Button>
            </div>
            <textarea
              className="max-h-40 min-h-[68px] w-full resize-y bg-transparent px-2 py-1.5 font-mono text-[13px] text-ink outline-none placeholder:text-ink-faint"
              placeholder={
                readOnly
                  ? 'Reviewing a past session — switch to Live to post.'
                  : mode === 'steer'
                    ? 'Steer the objective… e.g. "focus on GEX sign flips near the open" (Ctrl+Enter to send)'
                    : mode === 'task'
                      ? 'A one-off task for the swarm…  First line becomes the task title. (Ctrl+Enter to send)'
                      : 'Post to the board… (Ctrl+Enter to send)'
              }
              value={input}
              disabled={readOnly}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault()
                  submit()
                }
              }}
            />
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <div className="flex overflow-hidden rounded-lg border border-seam">
                {(
                  [
                    ['steer', 'Steer objective'],
                    ['task', 'One-off task'],
                    ['post', 'Just post'],
                  ] as const
                ).map(([m, label]) => (
                  <button
                    key={m}
                    disabled={m === 'steer' && !liveObjective}
                    onClick={() => setModeChoice(m)}
                    title={m === 'steer' && !liveObjective ? 'No running or paused objective' : undefined}
                    className={`px-2.5 py-1 font-mono text-[11px] transition-colors disabled:opacity-40 ${
                      mode === m ? 'bg-accent/15 text-accent' : 'text-ink-faint hover:bg-panel-hi hover:text-ink-dim'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {mode === 'task' && input.trim() && projectId && (
                <button
                  onClick={() => setNewObjective(input.trim())}
                  className="font-mono text-[11px] text-accent hover:opacity-80"
                  title="Make this a standing objective the swarm keeps improving on"
                >
                  make it an objective →
                </button>
              )}
              {mode === 'task' && (
                <select
                  className="rounded-lg border border-seam bg-panel-hi px-2 py-1 font-mono text-[12px] text-ink outline-none focus:border-accent"
                  value={tier}
                  onChange={(e) => setTier(e.target.value as Task['tier'])}
                >
                  <option value="auto">auto — router picks</option>
                  <option value="small">small — gpt-oss-20b</option>
                  <option value="mid">mid — Qwen3.6-35B-A3B</option>
                  <option value="hard">hard — DeepSeek-V4 (Ada box)</option>
                </select>
              )}
              <Button
                tone="primary"
                className="ml-auto"
                onClick={submit}
                disabled={!input.trim() || posting || readOnly}
              >
                {posting ? 'Posting…' : mode === 'steer' ? 'Steer' : mode === 'task' ? 'Queue task' : 'Post'}
              </Button>
            </div>
            {err && <div className="mt-2 font-mono text-[11px] text-bad">{err}</div>}
          </Panel>

          <div
            ref={feedRef}
            className="h-[70vh] min-h-[360px] space-y-2 overflow-y-auto rounded-2xl border border-seam bg-panel p-4"
          >
            {messages.length === 0 && (
              <div className="grid h-full place-items-center text-center">
                <div>
                  <div className="text-[14px] text-ink-dim">
                    {readOnly ? `This session has no #${channel} messages` : `No messages on #${channel}`}
                  </div>
                  <div className="mt-2 max-w-sm text-[12px] text-ink-faint">
                    Steer the objective or post a task above. Agents report back here, newest first.
                  </div>
                </div>
              </div>
            )}

            {newestFirst.map((m) => (
              <div key={m.seq} className="rounded-xl border border-seam/60 bg-panel-hi/40 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-[12px] text-ink">{m.author}</span>
                  <Pill tone={KIND_TONE[m.kind]}>{m.kind}</Pill>
                  {typeof m.meta?.to === 'string' && (
                    <span className="font-mono text-[10.5px] text-accent">→ @{String(m.meta.to).split('/').pop()}</span>
                  )}
                  {m.reply_to != null && <span className="font-mono text-[10.5px] text-ink-faint">re #{m.reply_to}</span>}
                  {m.meta?.collab != null && <span className="font-mono text-[10.5px] text-good">collaboration</span>}
                  <span className="font-mono text-[10px] text-ink-faint">#{m.seq}</span>
                  <span className="ml-auto font-mono text-[10px] text-ink-faint">
                    {clockTime(m.ts)}
                  </span>
                  <CopyButton
                    label="copy message"
                    className="-my-1"
                    text={() => `${m.author} · ${m.kind} · #${m.seq} · ${clockTime(m.ts)}\n${m.content}`}
                  />
                </div>
                <pre className="mt-2 overflow-x-auto whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-ink-dim">
                  {m.content}
                </pre>
              </div>
            ))}
          </div>
        </div>

        {/* ---- Resources + agents + tasks ---- */}
        <div className="space-y-4">
          <SwarmResourcesPanel />
          <TeamPanel />
          <Panel className="p-4">
            <div className="mb-3 text-[14px] font-medium text-ink">Agents</div>
            <div className="space-y-2">
              {agents.length === 0 && (
                <div className="text-[12px] text-ink-faint">
                  None registered. Agents call{' '}
                  <code className="font-mono">POST /mb/agents/register</code>.
                </div>
              )}
              {agents.map((a) => (
                <div key={a.id} className={`rounded-xl border p-3 ${isExternal(a.model || a.name) ? 'border-warn/40 bg-warn/10' : 'border-seam bg-panel-hi/40'}`}
                  title={isExternal(a.model || a.name) ? 'hosted model — every call costs money' : undefined}>
                  <div className="flex items-center gap-2">
                    <span
                      className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                        a.online ? 'bg-good animate-dot' : 'bg-ink-faint'
                      }`}
                    />
                    <span
                      className={`min-w-0 flex-1 truncate font-mono text-[12px] ${isExternal(a.model || a.name) ? 'text-warn' : a.name.includes('@') ? 'text-remote' : 'text-ink'}`}
                      title={!isExternal(a.model || a.name) && a.name.includes('@') ? 'runs on another computer on the network' : undefined}
                    >
                      {isExternal(a.model || a.name) && <span className="mr-1 font-bold">$</span>}
                      {a.name}
                    </span>
                    <Pill tone={a.status === 'working' ? 'accent' : 'neutral'}>{a.status}</Pill>
                  </div>
                  {(a.role || a.model) && (
                    <div className="mt-1.5 truncate font-mono text-[10px] text-ink-faint">
                      {[a.role, a.model].filter(Boolean).join(' · ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Panel>

          <Panel className="p-4">
            <div className="mb-3 flex items-center gap-2">
              <span className="text-[14px] font-medium text-ink">Tasks</span>
              <span className="ml-auto font-mono text-[11px] text-ink-faint">
                {counts.open ?? 0} open · {counts.claimed ?? 0} active · {counts.done ?? 0} done
              </span>
            </div>
            <div className="max-h-[420px] space-y-2 overflow-y-auto">
              {tasks.length === 0 && (
                <div className="text-[12px] text-ink-faint">No tasks queued.</div>
              )}
              {tasks.map((t) => {
                // Only a settled task has anything to show; an open one would open an
                // empty panel, which reads as a broken click.
                const readable = t.status === 'done' || t.status === 'failed'
                return (
                <div
                  key={t.id}
                  onClick={readable ? () => setOpenTaskId(t.id) : undefined}
                  role={readable ? 'button' : undefined}
                  tabIndex={readable ? 0 : undefined}
                  onKeyDown={
                    readable
                      ? (e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault()
                            setOpenTaskId(t.id)
                          }
                        }
                      : undefined
                  }
                  className={`rounded-xl border border-seam bg-panel-hi/40 p-3 ${
                    readable
                      ? 'cursor-pointer transition-colors hover:border-accent/50 hover:bg-panel-hi'
                      : ''
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <span className="min-w-0 flex-1 text-[12px] leading-snug text-ink">
                      {t.title}
                    </span>
                    <Pill tone={StatusTone(t.status)}>{t.status}</Pill>
                  </div>
                  <div className="mt-2 flex flex-wrap items-center gap-2 font-mono text-[10px] text-ink-faint">
                    <Pill tone={TIER_TONE[t.tier]}>{t.tier}</Pill>
                    {t.claimed_by && <span>by {t.claimed_by.slice(0, 8)}</span>}
                    {t.attempts > 1 && <span>attempt {t.attempts}</span>}
                    {readable && <span className="text-accent">view result</span>}
                    {t.status === 'failed' && (
                      <button
                        onClick={(e) => {
                          // The card itself opens the result; retry must not also open it.
                          e.stopPropagation()
                          void board.retryTask(t).then(() => refreshTasks())
                        }}
                        className="text-warn hover:opacity-80"
                      >
                        retry
                      </button>
                    )}
                    <span className="ml-auto">{duration((Date.now() / 1000) - t.created_at)} ago</span>
                  </div>
                </div>
                )
              })}
            </div>
          </Panel>
        </div>
      </div>

      {newObjective !== null && projectId && (
        <NewObjective
          projectId={projectId}
          initialText={newObjective}
          onClose={() => setNewObjective(null)}
          onCreated={(o) => {
            setNewObjective(null)
            setInput('')
            setModeChoice(null)
            obj.setSelected(o.id)
            obj.refresh()
          }}
        />
      )}

      {openTask && (
        <TaskResult
          task={openTask}
          onClose={() => setOpenTaskId(null)}
          onContinueInChat={() => {
            setOpenTaskId(null)
            router.push('/chat')
          }}
        />
      )}
    </div>
  )
}
