/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Next gzips proxied responses when the client sends Accept-Encoding: gzip -- which
  // every browser does. For an SSE stream that is fatal: the compressor buffers, so the
  // whole reply lands in one frame and chat renders in a single lump instead of token by
  // token. It is invisible to a curl/urllib test, because those do not ask for gzip.
  //
  // There is no per-route opt-out for Next's built-in compression, and a browser cannot
  // set Accept-Encoding itself (it is a forbidden header), so the switch has to be global.
  // The cost is nil here: this console is served over loopback, and its largest asset is
  // ~280 kB.
  compress: false,
  // Next's rewrite proxy gives up on an upstream response after 30s and answers the
  // browser with a bare 500. An external review is one blocking POST that thinks for a
  // minute or more before it sends a byte, so every review failed in the console while
  // the same request succeeded against :8000 directly. The ceiling here has to clear
  // review.py's own TIMEOUT_S (600s), so that a slow review ends in the reviewer's error
  // message rather than a proxy timeout with nothing to read.
  experimental: {
    proxyTimeout: 900_000,
  },
  // Both backends are proxied through this origin, so the browser never issues a
  // cross-origin request and neither service has to be reachable from anywhere but
  // loopback (or an SSH tunnel to it).
  //   /api/* -> control plane   (engine lifecycle, telemetry, chat)
  //   /mb/*  -> message board   (agent coordination)
  async rewrites() {
    const api = process.env.FREESWARM_API_URL || 'http://127.0.0.1:8000'
    const board = process.env.FREESWARM_BOARD_URL || 'http://127.0.0.1:8100'
    return [
      { source: '/api/:path*', destination: `${api}/api/:path*` },
      { source: '/mb/:path*', destination: `${board}/mb/:path*` },
    ]
  },
}
export default nextConfig
