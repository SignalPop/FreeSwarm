'use client'

import {
  createContext,
  isValidElement,
  memo,
  useCallback,
  useMemo,
  useState,
  useContext,
  useSyncExternalStore,
  type ReactNode,
} from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import RunResult from '@/components/RunResult'
import HtmlPreview from '@/components/HtmlPreview'
import { fetchSandboxStatus, useSandboxStatus } from '@/lib/useSandbox'
import { sandboxRuns } from '@/lib/sandboxRuns'

/**
 * Renders a model reply: markdown, LaTeX, tables, code and images.
 *
 * Three things make this harder than dropping ReactMarkdown in:
 *
 * 1. **Models emit LaTeX in delimiters remark-math does not know.** remark-math handles
 *    `$…$` and `$$…$$`; gpt-oss and friends routinely emit `\(…\)` and `\[…\]`, which
 *    would otherwise render as literal backslashes — exactly what the raw view showed.
 *    `normalizeMath` rewrites them, while carefully skipping fenced and inline code so a
 *    shell snippet containing `\[` is not mangled.
 *
 * 2. **It re-renders on every streamed token.** Half-arrived markup is the normal state,
 *    not an edge case: an unclosed `$$`, a fence with no terminator, a half-written table
 *    row. KaTeX must therefore never throw (`throwOnError: false`), and the whole tree is
 *    wrapped so a parser error degrades to plain text instead of blanking the message.
 *
 * 3. **Cost.** Parsing on each delta is why this is memoised on the source string.
 */

/** Rewrite `\(…\)` and `\[…\]` into `$…$` / `$$…$$`, leaving code spans untouched. */
export function normalizeMath(src: string): string {
  // Split on fenced blocks and inline code, keeping the delimiters; only transform the
  // segments between them.
  const parts = src.split(/(```[\s\S]*?```|`[^`\n]*`)/g)
  return parts
    .map((part) => {
      if (part.startsWith('```') || (part.startsWith('`') && part.endsWith('`'))) return part
      return part
        .replace(/\\\[([\s\S]*?)\\\]/g, (_m, body) => `\n$$${body}$$\n`)
        .replace(/\\\(([\s\S]*?)\\\)/g, (_m, body) => `$${body}$`)
    })
    .join('')
}

/**
 * Flatten a rendered node tree back to its source text.
 *
 * Needed because rehype-highlight replaces the code string with a tree of <span> elements,
 * one per token. `String(children)` on that yields "[object Object]" -- which is what the
 * copy button was silently putting on the clipboard whenever highlighting applied, and
 * what would otherwise be sent to the sandbox as the program to run.
 */
function nodeText(node: ReactNode): string {
  if (node == null || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(nodeText).join('')
  if (isValidElement(node)) return nodeText((node.props as { children?: ReactNode }).children)
  return ''
}

/** Languages the sandbox can execute. */
const RUNNABLE = new Set(['python', 'py', 'python3'])

/** False where the generic sandbox is the wrong place to run the code shown -- a swarm
 *  candidate needs the scoring harness (``import ft``, the data), which the viewer offers. */
const SandboxRunContext = createContext(true)

function CodeBlock({ children, className }: { children: React.ReactNode; className?: string }) {
  const [copied, setCopied] = useState(false)
  const lang = /language-(\w+)/.exec(className ?? '')?.[1]
  const runnable = useContext(SandboxRunContext) && lang !== undefined && RUNNABLE.has(lang.toLowerCase())
  // HTML needs no sandbox container -- it renders in an isolated frame in the browser.
  const previewable = lang !== undefined && ['html', 'htm', 'svg'].includes(lang.toLowerCase())
  const [previewing, setPreviewing] = useState(false)
  const code = useMemo(() => nodeText(children), [children])

  // Shared + self-healing: one probe for the whole page, re-checked periodically, and an
  // unknown result counts as available. Only ask at all for code that could be run.
  const sandbox = useSandboxStatus(runnable)
  const unavailable = sandbox?.available === false

  // The run lives in a module-level store, so leaving the page mid-run no longer discards
  // the result -- come back and it is waiting, or still going.
  const entry = useSyncExternalStore(
    sandboxRuns.subscribe,
    useCallback(() => sandboxRuns.get(code), [code]),
    () => undefined,
  )
  const running = entry?.status === 'running'
  const run = entry?.status === 'done' ? entry.run : undefined
  const runError = entry?.status === 'error' ? entry.error : undefined

  async function copy() {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard blocked (insecure origin / permissions) -- not worth surfacing */
    }
  }

  function execute() {
    sandboxRuns.start(code)
    // Whatever the outcome, re-probe: a success proves a stale "unavailable" wrong, and a
    // failure is worth reflecting in the other blocks' tooltips.
    if (unavailable) void fetchSandboxStatus(true)
  }

  return (
    <div className="group relative my-3">
      <div className="flex items-center justify-between rounded-t-lg border border-b-0 border-seam bg-panel-hi px-3 py-1.5">
        <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
          {lang ?? 'code'}
        </span>
        <button
          onClick={copy}
          className="font-mono text-[10px] text-ink-faint transition-colors hover:text-accent"
        >
          {copied ? 'copied' : 'copy'}
        </button>
      </div>
      <pre
        className={`overflow-x-auto border border-seam bg-canvas p-3 font-mono text-[12px] leading-relaxed ${
          runnable || previewable ? '' : 'rounded-b-lg'
        }`}
      >
        {children}
      </pre>
      {/* Run sits BELOW the code: you decide to run it after reading it, and this is where
          the output then appears -- so the button, its result and your eye are in one place
          instead of the button being back up at the top of a long listing. */}
      {runnable && (
        <div className="flex items-center gap-3 rounded-b-lg border border-t-0 border-seam bg-panel-hi px-3 py-2">
          <button
            onClick={execute}
            disabled={running}
            title={
              unavailable
                ? `${sandbox?.reason ?? 'Sandbox unavailable'} — click to try anyway`
                : 'Run this in an isolated container (no network access)'
            }
            className={`rounded-md border px-2.5 py-1 font-mono text-[11px] transition-colors disabled:cursor-wait disabled:opacity-60 ${
              unavailable
                ? 'border-seam text-ink-faint hover:border-accent/40 hover:text-ink-dim'
                : 'border-accent/40 text-accent hover:bg-accent/10'
            }`}
          >
            {running ? 'running…' : '▶ run'}
          </button>
          <span className="font-mono text-[10px] text-ink-faint">
            {running
              ? 'in a sandbox container'
              : unavailable
                ? (sandbox?.reason ?? 'sandbox unavailable')
                : 'runs in an isolated container — no network'}
          </span>
        </div>
      )}
      {previewable && (
        <div className="flex items-center gap-3 rounded-b-lg border border-t-0 border-seam bg-panel-hi px-3 py-2">
          <button
            onClick={() => setPreviewing((p) => !p)}
            className="rounded-md border border-accent/40 px-2.5 py-1 font-mono text-[11px] text-accent transition-colors hover:bg-accent/10"
          >
            {previewing ? 'hide preview' : '▶ preview'}
          </button>
          <span className="font-mono text-[10px] text-ink-faint">
            renders in an isolated frame — scripts run, but cannot reach the console
          </span>
        </div>
      )}
      {previewable && previewing && (
        <div className="mt-2">
          <HtmlPreview html={code} title={lang === 'svg' ? 'svg preview' : 'html preview'} />
        </div>
      )}
      {runError && (
        <div className="mt-2 rounded-lg border border-bad/40 bg-bad/[0.07] px-3 py-2 font-mono text-[11.5px] text-ink-dim">
          {runError}
        </div>
      )}
      {run && <RunResult run={run} code={code} onClose={() => sandboxRuns.clear(code)} />}
    </div>
  )
}

// Measured with the production pipeline: highlighting a 100-line UNTAGGED code block with
// `detect: true` costs 126 ms per render (every grammar is tried against it) versus 1.7 ms
// without. A streaming reply re-renders on every token, so at ~50 tok/s that was ~6 s of
// CPU per second of output -- the page fell further behind the longer the reply ran, until
// it stopped responding. Models tag their fences (```python) almost always; an untagged
// block now renders plain rather than guessing.
const KATEX = [rehypeKatex, { throwOnError: false, strict: false, output: 'htmlAndMathml' }] as const
const HIGHLIGHT = [rehypeHighlight, { detect: false, ignoreMissing: true }] as const

function MarkdownInner({ source, streaming }: { source: string; streaming: boolean }) {
  const normalized = useMemo(() => normalizeMath(source), [source])

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      // While a reply is still arriving it is re-rendered many times a second, so the
      // costliest step -- syntax highlighting -- waits until it is complete. The finished
      // message gets one full render with colours. KaTeX stays: a half-streamed formula is
      // normal, and throwOnError:false keeps it from taking the message down.
      rehypePlugins={streaming ? [KATEX as never] : [KATEX as never, HIGHLIGHT as never]}
      components={{
        p: ({ children }) => (
          <p className="my-2 text-[13.5px] leading-relaxed text-ink first:mt-0 last:mb-0">
            {children}
          </p>
        ),
        h1: ({ children }) => (
          <h1 className="mt-4 mb-2 text-[18px] font-semibold text-ink first:mt-0">{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="mt-4 mb-2 text-[16px] font-semibold text-ink first:mt-0">{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="mt-3 mb-1.5 text-[14px] font-semibold text-ink first:mt-0">{children}</h3>
        ),
        ul: ({ children }) => (
          <ul className="my-2 ml-5 list-disc space-y-1 text-[13.5px] leading-relaxed text-ink marker:text-ink-faint">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="my-2 ml-5 list-decimal space-y-1 text-[13.5px] leading-relaxed text-ink marker:text-ink-faint">
            {children}
          </ol>
        ),
        a: ({ children, href }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent underline underline-offset-2 hover:opacity-80"
          >
            {children}
          </a>
        ),
        strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
        blockquote: ({ children }) => (
          <blockquote className="my-3 border-l-2 border-accent/40 pl-3 text-[13px] italic text-ink-dim">
            {children}
          </blockquote>
        ),
        hr: () => <hr className="my-4 border-seam" />,
        table: ({ children }) => (
          <div className="my-3 overflow-x-auto rounded-lg border border-seam">
            <table className="w-full border-collapse text-[12.5px]">{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead className="bg-panel-hi">{children}</thead>,
        th: ({ children }) => (
          <th className="border-b border-seam px-3 py-2 text-left font-medium text-ink">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border-b border-seam/50 px-3 py-2 align-top text-ink-dim">{children}</td>
        ),
        img: ({ src, alt }) => (
          // Plain <img>: sources are arbitrary model output, which next/image cannot be
          // configured for, and a broken URL must not break the message.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={typeof src === 'string' ? src : ''}
            alt={alt ?? ''}
            loading="lazy"
            className="my-3 max-h-[480px] max-w-full rounded-lg border border-seam"
          />
        ),
        pre: ({ children }) => <>{children}</>,
        code: ({ className, children, ...props }) => {
          const isBlock = (className ?? '').includes('language-')
          if (isBlock) {
            return (
              <CodeBlock className={className}>
                <code className={className} {...props}>
                  {children}
                </code>
              </CodeBlock>
            )
          }
          return (
            <code className="rounded border border-seam bg-panel-hi px-1 py-0.5 font-mono text-[12px] text-accent">
              {children}
            </code>
          )
        },
      }}
    >
      {normalized}
    </ReactMarkdown>
  )
}

const Memoised = memo(
  MarkdownInner,
  (a, b) => a.source === b.source && a.streaming === b.streaming,
)

export default function Markdown({
  source,
  streaming = false,
  sandboxRun = true,
}: {
  source: string
  /** True while this text is still arriving: skips syntax highlighting until it is done. */
  streaming?: boolean
  /** False hides the sandbox Run button on Python blocks. */
  sandboxRun?: boolean
}) {
  try {
    return (
      <div className="ft-markdown">
        <SandboxRunContext.Provider value={sandboxRun}>
          <Memoised source={source} streaming={streaming} />
        </SandboxRunContext.Provider>
      </div>
    )
  } catch {
    // Last resort: show the text rather than an empty bubble.
    return (
      <pre className="whitespace-pre-wrap font-mono text-[13px] leading-relaxed text-ink">
        {source}
      </pre>
    )
  }
}
