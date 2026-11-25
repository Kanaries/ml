import './global.css';
import { RootProvider } from 'fumadocs-ui/provider';
import { Inter, Fraunces, Newsreader, JetBrains_Mono } from 'next/font/google';
import type { ReactNode } from 'react';
import type { Metadata } from 'next';
import { DEFAULT_OG_IMAGE, SITE_URL } from '@/lib/seo';

// UI / chrome sans
const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

// Editorial display serif — characterful, optical
const fraunces = Fraunces({
  subsets: ['latin'],
  variable: '--font-fraunces',
  display: 'optional',
  preload: false,
  style: ['normal', 'italic'],
});

// Reading serif for long-form / scholarly prose
const newsreader = Newsreader({
  subsets: ['latin'],
  variable: '--font-newsreader',
  display: 'optional',
  preload: false,
  style: ['normal', 'italic'],
});

// Technical monospace for code, labels and metadata
const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jbmono',
  display: 'optional',
  preload: false,
});

const fontVars = [
  inter.variable,
  fraunces.variable,
  newsreader.variable,
  jetbrainsMono.variable,
].join(' ');

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: '@kanaries/ml Documentation',
    template: '%s | @kanaries/ml',
  },
  description:
    'Train and run scikit-learn-style ML models in JavaScript and TypeScript with @kanaries/ml. API docs, tutorials, and interactive tools for browser and Node.js.',
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
    images: [DEFAULT_OG_IMAGE],
  },
  twitter: {
    card: 'summary_large_image',
    title: '@kanaries/ml Documentation',
    description:
      'JavaScript and TypeScript machine learning docs with practical algorithm guides for browser and Node.js.',
    images: [DEFAULT_OG_IMAGE.url],
  },
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={fontVars} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
