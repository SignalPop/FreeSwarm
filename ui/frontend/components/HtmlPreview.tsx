'use client'

import { useEffect, useState } from 'react'

/**
 * Renders model-written HTML in an isolated frame.
 *
 * `sandbox="allow-scripts"` WITHOUT `allow-same-origin` is the important part: the page gets
 * an opaque origin, so its scripts run (a chart library, an animation) but cannot read the
 * console's storage, call its API with the user's session, or navigate the top window. This
 * is arbitrary code from a model -- it gets a canvas, not the app.
 *
 * Content goes in via `srcDoc` rather than `src`, even for sandbox artifacts: the artifact
 * route is behind the control plane's auth, and a frame's own request cannot carry the
 * bearer token, so the HTML is fetched with the app's credentials and handed over as text.
 */
export default function HtmlPreview({
  html,
  url,
  title,
}: {
  /** Inline HTML (a ```html code block). */
  html?: string
  /** Or: an artifact to fetch (a .html file a sandbox run produced). */
  url?: string
  title?: string
}) {
  const [doc, setDoc] = useState<string | null>(html ?? null)
  const [err, setErr] = useState<string | null>(null)
  const [tall, setTall] = useState(false)

  useEffect(() => {
    if (html !== undefined) {
      setDoc(html)
      return
    }
    if (!url) return
    let alive = true
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
        return r.text()
      })
      .then((t) => alive && setDoc(t))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : String(e)))
    return () => {
      alive = false
    }
  }, [html, url])

  if (err) {
    return <div className="font-mono text-[11.5px] text-bad">could not load preview: {err}</div>
  }
  if (doc === null) {
    return <div className="font-mono text-[11.5px] text-ink-faint">loading preview…</div>
  }

  return (
    <div className="overflow-hidden rounded-lg border border-seam">
      <div className="flex items-center justify-between border-b border-seam bg-panel-hi px-3 py-1.5">
        <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
          {title ?? 'html preview'} · sandboxed
        </span>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setTall((t) => !t)}
            className="font-mono text-[10px] text-ink-faint transition-colors hover:text-accent"
          >
            {tall ? 'shorter' : 'taller'}
          </button>
          <button
            onClick={() => {
              // NOT a blob: URL. A blob URL inherits the ORIGIN OF THE PAGE THAT CREATED IT,
              // so opening one would give model-written HTML the console's origin -- its
              // storage, and its API with the user's session. Instead the new tab holds only
              // our own wrapper, and the content lives in the same opaque-origin sandboxed
              // frame as the inline preview.
              const w = window.open('', '_blank')
              if (!w) return
              w.opener = null
              w.document.title = title ?? 'html preview'
              w.document.body.style.margin = '0'
              const frame = w.document.createElement('iframe')
              frame.setAttribute('sandbox', 'allow-scripts')
              frame.setAttribute('referrerpolicy', 'no-referrer')
              frame.srcdoc = doc
              frame.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;border:0'
              w.document.body.appendChild(frame)
            }}
            className="font-mono text-[10px] text-ink-faint transition-colors hover:text-accent"
          >
            open in tab
          </button>
        </div>
      </div>
      <iframe
        title={title ?? 'html preview'}
        srcDoc={doc}
        sandbox="allow-scripts"
        referrerPolicy="no-referrer"
        className={`block w-full bg-white ${tall ? 'h-[760px]' : 'h-[420px]'}`}
      />
    </div>
  )
}
