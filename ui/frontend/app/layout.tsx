import type { Metadata } from 'next'
import './globals.css'
import AuthGate from '@/components/AuthGate'
import Shell from '@/components/Shell'

export const metadata: Metadata = {
  title: 'FreeSwarm Console',
  description: 'Local MoE inference control plane',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning: the inline script below sets data-theme before React
    // hydrates, so the server-rendered html tag necessarily differs from the client one.
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Applied before first paint so a dark-theme user never sees a white flash. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('ft-theme');if(t)document.documentElement.setAttribute('data-theme',t);}catch(e){}})()`,
          }}
        />
      </head>
      <body>
        <AuthGate>
          <Shell>{children}</Shell>
        </AuthGate>
      </body>
    </html>
  )
}
