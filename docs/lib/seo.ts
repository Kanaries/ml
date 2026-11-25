import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

export const SITE_URL = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export const DEFAULT_OG_IMAGE = {
  url: `${SITE_URL}/opengraph-image`,
  width: 1200,
  height: 630,
  alt: '@kanaries/ml — Machine Learning in JavaScript & TypeScript',
};

export const TOOLS_OG_IMAGE = {
  url: `${SITE_URL}/tools/opengraph-image`,
  width: 1200,
  height: 630,
  alt: 'Free Browser-Based Machine Learning Calculators — @kanaries/ml',
};

export const PLAYGROUND_OG_IMAGE = {
  url: `${SITE_URL}/playground/opengraph-image`,
  width: 1200,
  height: 630,
  alt: 'Interactive Machine Learning Playgrounds in JavaScript — @kanaries/ml',
};

export function safeJsonLd(value: unknown): string {
  return JSON.stringify(value).replace(/</g, '\\u003c');
}

export type BreadcrumbItem = {
  name: string;
  path: string;
};

export function buildBreadcrumbLd(items: BreadcrumbItem[]) {
  return {
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: item.name,
      item: item.path === '/' ? `${SITE_URL}/` : `${SITE_URL}${item.path}`,
    })),
  };
}

export type CollectionItem = {
  name: string;
  description: string;
  path: string;
};

export function buildCollectionLd(params: {
  name: string;
  description: string;
  pathname: string;
  items: CollectionItem[];
  breadcrumbs: BreadcrumbItem[];
}) {
  const url = `${SITE_URL}${params.pathname}`;
  return {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'CollectionPage',
        '@id': `${url}#collection`,
        name: params.name,
        description: params.description,
        url,
        inLanguage: 'en',
        isPartOf: {
          '@type': 'WebSite',
          name: '@kanaries/ml',
          url: `${SITE_URL}/`,
        },
        mainEntity: {
          '@type': 'ItemList',
          numberOfItems: params.items.length,
          itemListElement: params.items.map((item, index) => ({
            '@type': 'ListItem',
            position: index + 1,
            name: item.name,
            description: item.description,
            url: `${SITE_URL}${item.path}`,
          })),
        },
      },
      buildBreadcrumbLd(params.breadcrumbs),
    ],
  };
}

const lastModifiedCache = new Map<string, Date>();

export function getFileLastModified(filePath: string): Date {
  const resolved = path.resolve(filePath);
  const cached = lastModifiedCache.get(resolved);
  if (cached) return cached;

  let result: Date | undefined;
  try {
    const out = execFileSync('git', ['log', '-1', '--format=%cI', '--', resolved], {
      cwd: path.dirname(resolved),
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    if (out) {
      const parsed = new Date(out);
      if (!Number.isNaN(parsed.getTime())) result = parsed;
    }
  } catch {
    // Fallback to filesystem mtime when git metadata is unavailable
  }

  if (!result) {
    try {
      result = fs.statSync(resolved).mtime;
    } catch {
      result = new Date();
    }
  }

  lastModifiedCache.set(resolved, result);
  return result;
}
