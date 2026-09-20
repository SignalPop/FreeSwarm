'use client'

import { useRouter } from 'next/navigation'
import { artifactUrl, type SandboxRun } from '@/lib/api'
import { chatStore } from '@/lib/chatStore'
import { formatRunForChat, useShareToChat } from '@/components/ShareToChat'
import HtmlPreview from '@/components/HtmlPreview'

// What the sandbox can import. Told to the model when it is asked to fix a failure: the
// container has NO network, so "pip install rdkit" -- the usual reflex -- can never work,
// and without this the model keeps suggesting it instead of rewriting.
const SANDBOX_PACKAGES =
  'numpy, pandas, scipy, matplotlib, pillow, rdkit, imageio, plotly, openpyxl, XlsxWriter, ' +
  'odfpy, python-docx, python-pptx, reportlab, fpdf2, pyarrow, tabulate, python-dateutil'

function fixPrompt(code: string, run: SandboxRun): string {
  const why = run.timed_out
    ? 'It hit the time limit and was killed.'
    : `It exited with code ${run.exit_code}.`
  const parts = [
    `This code failed when I ran it in the sandbox. ${why}`,
    '',
    '```python',
    code.trimEnd(),
    '```',
    '',
    'Error output:',
    '```',
    (run.stderr || '(none)').trim(),
    '```',
  ]
  if (run.stdout.trim()) parts.push('', 'Standard output before it failed:', '```', run.stdout.trim(), '```')
  parts.push(
    '',
    `Fix it and return the COMPLETE corrected script in one \`\`\`python block. The sandbox ` +
      `has no network access, so nothing can be pip-installed; only these packages are ` +
      `available: ${SANDBOX_PACKAGES}. It has no display either -- save figures to files ` +
      `(plt.show() is saved automatically).`,
  )
  return parts.join('\n')
}

/** 1.2 MB / 340 kB / 812 B */
function humanSize(bytes: number): string {
  if (bytes >= 1 << 20) return `${(bytes / (1 << 20)).toFixed(1)} MB`
  if (bytes >= 1 << 10) return `${Math.round(bytes / (1 << 10))} kB`
  return `${bytes} B`
}

/**
 * What a sandbox run produced: output streams, then the files themselves.
 *
 * Images are shown, not linked. A chart the user has to download to look at is the same
 * as no chart -- and a plot is the single most common thing this sandbox is asked for.
 * Everything else (xlsx, docx, csv, pdf) becomes a download, because the browser cannot
 * usefully render those inline anyway.
 */
export default function RunResult({
  run,
  code,
  onClose,
}: {
  run: SandboxRun
  /** The code that ran -- needed to ask a model to fix it. */
  code?: string
  onClose?: () => void
}) {
  const images = run.artifacts.filter((a) => a.mime.startsWith('image/'))
  const pages = run.artifacts.filter((a) => a.mime === 'text/html')
  const files = run.artifacts.filter(
    (a) => !a.mime.startsWith('image/') && a.mime !== 'text/html',
  )
  const share = useShareToChat()
  const router = useRouter()

  /** Send the failure straight to the model and go watch it answer. The reply comes back
   *  with its own Run button, so fix -> run -> fix is a loop you can keep turning. */
  function fixWithChat() {
    if (!code) return
    chatStore.setInput(fixPrompt(code, run))
    void chatStore.send()
    router.push('/chat')
  }

  return (
    <div className="mt-2 overflow-hidden rounded-lg border border-seam bg-canvas">
      <div className="flex items-center justify-between border-b border-seam bg-panel-hi px-3 py-1.5">
        <div className="flex items-center gap-2">
          <span
            className={`h-1.5 w-1.5 rounded-full ${run.ok ? 'bg-good' : 'bg-bad'}`}
            aria-hidden
          />
          <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            {run.ok ? 'ran' : run.timed_out ? 'timed out' : `exit ${run.exit_code}`} ·{' '}
            {run.duration_s}s
          </span>
        </div>
        <div className="flex items-center gap-3">
          {!run.ok && code && (
            <button
              onClick={fixWithChat}
              title="Send the code and its error to the model, and get a fixed version back"
              className="rounded border border-accent/50 px-2 py-0.5 font-mono text-[10px] text-accent transition-colors hover:bg-accent/10"
            >
              fix with chat
            </button>
          )}
          {share && (
            <button
              onClick={() =>
                share(
                  formatRunForChat({
                    ok: run.ok,
                    exitCode: run.exit_code,
                    timedOut: run.timed_out,
                    stdout: run.stdout,
                    stderr: run.stderr,
                    artifacts: run.artifacts,
                  }),
                )
              }
              title="Put this output in the composer so the model can see it"
              className="font-mono text-[10px] text-accent transition-colors hover:opacity-80"
            >
              send to chat
            </button>
          )}
          {onClose && (
            <button
              onClick={onClose}
              className="font-mono text-[10px] text-ink-faint transition-colors hover:text-accent"
            >
              close
            </button>
          )}
        </div>
      </div>

      <div className="space-y-3 p-3">
        {run.stdout && (
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap font-mono text-[11.5px] leading-relaxed text-ink">
            {run.stdout}
          </pre>
        )}
        {run.stderr && (
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap font-mono text-[11.5px] leading-relaxed text-bad">
            {run.stderr}
          </pre>
        )}
        {!run.stdout && !run.stderr && run.artifacts.length === 0 && (
          <div className="font-mono text-[11.5px] text-ink-faint">
            No output and no files produced.
          </div>
        )}

        {images.map((a) => (
          <figure key={a.name} className="space-y-1">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={artifactUrl(run.run_id, a.name)}
              alt={a.name}
              className="max-h-[520px] max-w-full rounded border border-seam bg-white"
            />
            <figcaption className="flex items-center gap-2 font-mono text-[10px] text-ink-faint">
              <span>{a.name}</span>
              <a
                href={artifactUrl(run.run_id, a.name)}
                download={a.name.split('/').pop()}
                className="text-accent hover:opacity-80"
              >
                download
              </a>
            </figcaption>
          </figure>
        ))}

        {pages.map((a) => (
          <HtmlPreview key={a.name} url={artifactUrl(run.run_id, a.name)} title={a.name} />
        ))}

        {files.length > 0 && (
          <div className="space-y-1">
            <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
              files
            </div>
            <ul className="space-y-1">
              {files.map((a) => (
                <li key={a.name} className="flex items-center gap-2 text-[12px]">
                  <a
                    href={artifactUrl(run.run_id, a.name)}
                    download={a.name.split('/').pop()}
                    className="font-mono text-accent underline underline-offset-2 hover:opacity-80"
                  >
                    {a.name}
                  </a>
                  <span className="font-mono text-[10px] text-ink-faint">
                    {humanSize(a.size)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {run.truncated && (
          <div className="font-mono text-[10px] text-ink-faint">
            Output was truncated in the middle.
          </div>
        )}
      </div>
    </div>
  )
}
