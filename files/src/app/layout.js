import './globals.css'

export const metadata = {
  title: 'RAG Video Analyzer',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}