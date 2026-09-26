'use client'

import { useMemo, useState } from 'react'
import type { AskedPrompt } from '@/lib/agents'
import CopyButton from '@/components/CopyButton'

/**
 * What an agent was actually sent, two ways: as the named sections it is made of (Playbook,
 * Lessons, Leaderboard, Ideas...) -- coloured by group, sized by length, each clickable for its
 * exact text, with the sections an iteration prompt normally carries but this one did NOT
 * marked as missing -- and as the full text, verbatim.
 *
 * The sections are recovered from the text itself, by the conventions the runner writes them
 * with (swarm_runner.iteration_prompt / the system prompt): an UPPERCASE header after a blank
 * line starts an iteration-prompt section; in the system prompt, the rules come first, then the
 * playbook's `# ` headings, then the "Project:" tool briefing. Nothing is inferred beyond that.
 */

type Group = { key: string; label: string }
// Fixed order = fixed colour (--color-series-N); never cycled. Identity is also in the text.
const GROUPS: Group[] = [
  { key: 'rules', label: 'Rules & scoring' },
  { key: 'playbook', label: 'Playbook' },
  { key: 'data', label: 'Data & tools' },
  { key: 'memory', label: 'Team memory' },
  { key: 'direction', label: 'Direction & messages' },
  { key: 'research', label: 'Research findings' },
  { key: 'assignment', label: 'Assignment' },
  { key: 'other', label: 'Other' },
]
const groupColor = (key: string) => `var(--color-series-${GROUPS.findIndex((g) => g.key === key) + 1})`

// Header prefix -> [friendly name, group]. First match wins, so longer prefixes come first.
const KNOWN: [string, string, string][] = [
  ['OBJECTIVE', 'Objective', 'assignment'],
  ['YOUR ASSIGNMENT', 'Assignment', 'assignment'],
  ['HOW "BETTER" IS MEASURED', 'How it is scored', 'rules'],
  ['CANDIDATE CONTRACT', 'Candidate contract', 'rules'],
  ['TIMEFRAMES', 'Timeframes', 'rules'],
  ['REGIMES', 'Regimes', 'rules'],
  ['TIME-SERIES FORECASTING MODELS', 'Forecasting models', 'data'],
  ['FORECASTERS LOADED', 'Forecasters loaded', 'data'],
  ['FIELD GUIDE', 'Field guide', 'data'],
  ['CODE LIBRARY', 'Code library', 'data'],
  ['FORECAST FEATURES ALREADY BUILT', 'Forecast features', 'data'],
  ['TEAM LESSONS', 'Lessons', 'memory'],
  ['TEAM HABITS', 'Team habits', 'memory'],
  ['LEADERBOARD', 'Leaderboard', 'memory'],
  ['RECENT ATTEMPTS', 'Recent attempts', 'memory'],
  ['OPERATOR FAVOURITES', 'Liked runs', 'memory'],
  ['IDEAS SCOREBOARD', 'Ideas scoreboard', 'memory'],
  ['OPERATOR STEERING', 'Steering', 'direction'],
  ['DIRECTIONS FROM THE MENTOR', 'Ideas (mentor / stuck / scheduled)', 'direction'],
  ['MESSAGES TO YOU', 'Messages to it', 'direction'],
  ['TEAMMATES RIGHT NOW', 'Teammates', 'direction'],
  ['LATEST FIELD SCAN', 'Field scan', 'research'],
  ['FIELD SCAN', 'Field scan', 'research'],
  ['REGIME MAP', 'Regime map', 'research'],
  ['DECILE STUDIES', 'Decile studies', 'research'],
  ['FORECAST SCOREBOARD', 'Forecast scoreboard', 'research'],
  ['FORECAST LAB', 'Forecast lab', 'research'],
  ['FORECAST INPUTS', 'Forecast input combos', 'research'],
  // The practices and mentor prompts' own sections.
  ['CURRENT PRACTICES', 'Current practices', 'playbook'],
  ['RECENT CANDIDATES', 'Recent candidates', 'memory'],
  ['RECENT ERRORS', 'Recent errors', 'memory'],
  ['LESSONS', 'Lessons', 'memory'],
  ['HYPOTHESIS', 'Hypothesis', 'memory'],
  ['LIBRARY', 'Library', 'data'],
  ['HOW THE AGENTS COLLABORATED', 'How the agents collaborated', 'direction'],
]

// What an ordinary search iteration is built from. The ones that are data-dependent (no lessons
// yet, no liked runs) can legitimately be absent -- that is exactly what this makes visible.
const EXPECTED = ['Playbook: charter', 'Playbook: practices', 'Playbook: pitfalls', 'Lessons', 'Leaderboard',
  'Recent attempts', 'Ideas (mentor / stuck / scheduled)', 'Steering', 'Liked runs', 'Code library',
  'Forecast scoreboard', 'Field scan', 'Regime map', 'Decile studies', 'Messages to it']

export type Part = { name: string; header: string; group: string; text: string; where: 'system' | 'prompt' }

// An UPPERCASE run of >= 4 letters, ending the line or followed by " (", ":", " --" or a
// lowercase word. A single word must end the line or be followed by ":" / " (" -- so an
// instruction like "IMPROVE candidate 12" inside a section never starts a new one.
const HEADER = /^([A-Z][A-Z0-9"'&/ -]*[A-Z0-9")])(?=$|\s*\(|:|\s--|\s[a-z{])/

function headerOf(line: string): string | null {
  const m = HEADER.exec(line)
  if (!m) return null
  const run = m[1].trim()
  if ((run.match(/[A-Z]/g) ?? []).length < 4) return null
  const rest = line.slice(m[1].length)
  if (!run.includes(' ') && !/^($|\s*\(|:)/.test(rest)) return null
  return run
}

function classify(header: string): [string, string] {
  const hit = KNOWN.find(([p]) => header.startsWith(p))
  if (hit) return [hit[1], hit[2]]
  const pretty = header.charAt(0) + header.slice(1).toLowerCase()
  return [pretty, 'other']
}

/** Iteration / user prompt: split at UPPERCASE headers that follow a blank line. */
export function splitPrompt(text: string, where: Part['where'] = 'prompt'): Part[] {
  const lines = text.replace(/\r\n?/g, '\n').split('\n')
  const parts: Part[] = []
  let cur: { header: string; lines: string[] } = { header: '', lines: [] }
  const flush = () => {
    const body = cur.lines.join('\n').replace(/\n+$/, '')
    if (!body.trim()) return
    const [name, group] = cur.header ? classify(cur.header) : ['Preamble', 'other']
    parts.push({ name, header: cur.header, group, text: body, where })
  }
  lines.forEach((line, i) => {
    const h = (i === 0 || lines[i - 1].trim() === '') && headerOf(line)
    if (h) {
      flush()
      cur = { header: h, lines: [line] }
    } else cur.lines.push(line)
  })
  flush()
  return parts
}

/** System prompt: the rules, then the playbook's `# ` headings, then the "Project:" briefing. */
export function splitSystem(raw: string): Part[] {
  const text = raw.replace(/\r\n?/g, '\n')
  const parts: Part[] = []
  const brief = text.search(/\n\nProject: /)
  const head = brief >= 0 ? text.slice(0, brief) : text
  const tail = brief >= 0 ? text.slice(brief + 2) : ''
  const chunks = head.split(/\n(?=# )/)
  chunks.forEach((chunk, i) => {
    if (!chunk.trim()) return
    if (i === 0 && !chunk.startsWith('# ')) {
      parts.push({ name: 'Role & rules', header: '', group: 'rules', text: chunk.trim(), where: 'system' })
      return
    }
    const title = chunk.split('\n')[0].replace(/^#\s*/, '')
    const name = /^team practices/i.test(title)
      ? 'Playbook: practices'
      : /^pitfalls/i.test(title)
        ? 'Playbook: pitfalls'
        : /charter/i.test(title)
          ? 'Playbook: charter'
          : `Playbook: ${title.slice(0, 40)}`
    parts.push({ name, header: title, group: 'playbook', text: chunk.trim(), where: 'system' })
  })
  if (tail.trim()) parts.push({ name: 'Project & tools briefing', header: 'Project', group: 'data', text: tail.trim(), where: 'system' })
  return parts
}

export default function PromptAnatomy({ a }: { a: AskedPrompt }) {
  const [view, setView] = useState<'blocks' | 'full'>('blocks')
  const [open, setOpen] = useState<number | null>(null)
  const parts = useMemo(() => [...(a.system ? splitSystem(a.system) : []), ...splitPrompt(a.prompt)], [a.system, a.prompt])
  const total = parts.reduce((n, p) => n + p.text.length, 0) || 1
  const names = new Set(parts.map((p) => p.name))
  const isIteration = names.has('Assignment')
  const missing = isIteration ? EXPECTED.filter((n) => !names.has(n)) : []
  const cut = a.system_chars > a.system.length || a.prompt_chars > a.prompt.length
  const full = (a.system ? `### SYSTEM\n${a.system}\n\n### USER\n` : '') + a.prompt
  const used = GROUPS.filter((g) => g.key !== 'other' && parts.some((p) => p.group === g.key))
  // Known sections take their group's colour. Sections this view does not recognise (a
  // prompt shape it has no map for) each get their own, in order -- one shared colour made a
  // whole practices prompt a single red bar.
  const colorOf = useMemo(() => {
    let k = 0
    return parts.map((p) => (p.group === 'other' ? `var(--color-series-${(k++ % 8) + 1})` : groupColor(p.group)))
  }, [parts])
  const others = parts.some((p) => p.group === 'other')

  return (
    <div className="rounded-lg border border-seam">
      <div className="flex flex-wrap items-center gap-2 border-b border-seam px-3 py-1.5">
        <div role="tablist" className="flex rounded-md border border-seam p-0.5 font-mono text-[11px]">
          {(['blocks', 'full'] as const).map((v) => (
            <button
              key={v}
              role="tab"
              aria-selected={view === v}
              onClick={() => setView(v)}
              className={`rounded px-2 py-0.5 ${view === v ? 'bg-panel-hi text-ink' : 'text-ink-faint hover:text-ink'}`}
            >
              {v === 'blocks' ? 'sections' : 'full text'}
            </button>
          ))}
        </div>
        <span className="font-mono text-[10.5px] text-ink-faint">
          {parts.length} sections · {(a.system_chars + a.prompt_chars).toLocaleString()} chars
          {cut ? ' · middle cut by the recorder' : ''}
        </span>
        <span className="flex-1" />
        <CopyButton text={full} label="copy full prompt" />
      </div>

      {view === 'full' ? (
        <pre className="max-h-[560px] overflow-auto whitespace-pre-wrap bg-panel-hi p-3 font-mono text-[11.5px] leading-relaxed text-ink-dim">
          {full}
        </pre>
      ) : (
        <div className="space-y-3 p-3">
          {/* Composition by length: where the prompt's characters go. */}
          {/* Each segment opens its section, like the named blocks below. */}
          <div className="flex h-4 w-full gap-0.5 overflow-hidden rounded">
            {parts.map((p, i) => (
              <button
                key={i}
                onClick={() => setOpen(open === i ? null : i)}
                aria-label={`${p.name}, ${p.text.length.toLocaleString()} characters`}
                aria-pressed={open === i}
                title={`${p.name} · ${p.text.length.toLocaleString()} chars -- click to read`}
                style={{ flexGrow: p.text.length / total, flexBasis: 0, minWidth: 3, background: colorOf[i] }}
                className={`cursor-pointer transition-opacity hover:opacity-100 ${
                  open === null ? 'opacity-85' : open === i ? 'opacity-100 ring-2 ring-inset ring-ink' : 'opacity-40'
                }`}
              />
            ))}
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-1 font-mono text-[10px] text-ink-faint">
            {used.map((g) => (
              <span key={g.key} className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-sm" style={{ background: groupColor(g.key) }} />
                {g.label}
              </span>
            ))}
            {others && <span>other sections: each its own colour, named on its block</span>}
          </div>

          {(['system', 'prompt'] as const).map((where) => {
            const idx = parts.map((p, i) => [p, i] as const).filter(([p]) => p.where === where)
            if (!idx.length) return null
            return (
              <div key={where}>
                <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-ink-faint">
                  {where === 'system' ? 'system prompt' : a.system ? 'iteration prompt' : 'prompt'}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {idx.map(([p, i]) => (
                    <button
                      key={i}
                      onClick={() => setOpen(open === i ? null : i)}
                      aria-expanded={open === i}
                      className={`flex items-center gap-1.5 rounded-md border px-2 py-1 text-left font-mono text-[11px] ${
                        open === i ? 'border-ink-dim bg-panel-hi text-ink' : 'border-seam text-ink-dim hover:text-ink'
                      }`}
                      style={{ borderLeftWidth: 4, borderLeftColor: colorOf[i] }}
                    >
                      {p.name}
                      <span className="text-ink-faint">{fmtChars(p.text.length)}</span>
                    </button>
                  ))}
                </div>
              </div>
            )
          })}

          {missing.length > 0 && (
            <div>
              <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-ink-faint">
                not in this prompt
              </div>
              <div className="flex flex-wrap gap-1.5">
                {missing.map((n) => (
                  <span
                    key={n}
                    title="An iteration prompt carries this when there is something to say (e.g. no lessons yet means no Lessons section)"
                    className="rounded-md border border-dashed border-seam px-2 py-1 font-mono text-[11px] text-ink-faint"
                  >
                    {n}
                  </span>
                ))}
              </div>
            </div>
          )}

          {open !== null && parts[open] && (
            <div className="rounded-lg border border-seam" style={{ borderLeftWidth: 4, borderLeftColor: colorOf[open] }}>
              <div className="flex items-center gap-2 border-b border-seam px-3 py-1.5 font-mono text-[11px] text-ink-dim">
                <span className="text-ink">{parts[open].name}</span>
                <span className="text-ink-faint">
                  {parts[open].where === 'system' ? 'system prompt' : 'prompt'} · {parts[open].text.length.toLocaleString()} chars
                </span>
                <span className="flex-1" />
                <CopyButton text={parts[open].text} label={`copy ${parts[open].name}`} />
              </div>
              <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap bg-panel-hi p-3 font-mono text-[11.5px] leading-relaxed text-ink-dim">
                {parts[open].text}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function fmtChars(n: number): string {
  return n >= 10_000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)
}
