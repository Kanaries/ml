import './global.css';
import { RootProvider } from 'fumadocs-ui/provider';
import { Inter } from 'next/font/google';
import type { ReactNode } from 'react';
import type { Metadata } from 'next';

const inter = Inter({
  subsets: ['latin'],
});

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: '@kanaries/ml Documentation',
    template: '%s | @kanaries/ml Docs',
  },
  description:
    'Learn machine learning in JavaScript and TypeScript with @kanaries/ml. API guides, practical algorithm tutorials, and deployment patterns for browser and Node.js.',
  applicationName: '@kanaries/ml Docs',
  keywords: [
    'JavaScript machine learning',
    'TypeScript machine learning library',
    'scikit-learn for JavaScript',
    'browser machine learning',
    'Node.js machine learning',
    '@kanaries/ml documentation',
  ],
  openGraph: {
    type: 'website',
    siteName: '@kanaries/ml',
    title: '@kanaries/ml Documentation',
    description:
      'Machine learning documentation for JavaScript and TypeScript: algorithms, examples, and sklearn-style APIs.',
    url: `${siteUrl}/docs`,
  },
  twitter: {
    card: 'summary_large_image',
    title: '@kanaries/ml Documentation',
    description:
      'JavaScript and TypeScript machine learning docs with practical algorithm guides for browser and Node.js.',
  },
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
