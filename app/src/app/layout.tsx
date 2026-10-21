import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Craving Engine • Автономное исследование нейросети',
  description: 'Локальная операционная панель автономного поиска Craving Engine',
};

export const viewport: Viewport = {
  colorScheme: 'light dark',
  initialScale: 1,
  width: 'device-width',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang='ru' suppressHydrationWarning>
      <body className='min-h-screen bg-background text-foreground antialiased'>
        {children}
      </body>
    </html>
  );
}
