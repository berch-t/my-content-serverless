import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'My Content - Recommandations Intelligentes',
  description: 'Système de recommandation hybride utilisant l\'intelligence artificielle pour vous proposer les meilleurs articles',
  keywords: 'recommandation, intelligence artificielle, articles, contenu personnalisé',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="fr">
      <body className={inter.className}>
        <div className="min-h-full bg-gradient-to-br from-black via-purple-980 to-black bg-fixed bg-no-repeat bg-cover" style={{minHeight: '100vh'}}>
          {children}
        </div>
      </body>
    </html>
  )
}