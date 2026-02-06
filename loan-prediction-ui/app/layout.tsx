import './globals.css'

export const metadata = {
  title: 'Loan Approval Prediction',
  description: 'Predict loan approval using machine learning',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
