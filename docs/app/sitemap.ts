import path from 'node:path';
import type { MetadataRoute } from 'next';
import { source } from '@/lib/source';
import { SITE_URL, getFileLastModified } from '@/lib/seo';

const siteUrl = SITE_URL;
const docsRoot = process.cwd();

function appPageLastModified(relativeAppPath: string): Date {
  return getFileLastModified(path.join(docsRoot, 'app', relativeAppPath));
}

export default function sitemap(): MetadataRoute.Sitemap {
  const docPages = source.getPages().map((page) => {
    const isGuide = page.url.startsWith('/docs/guides/') || page.url === '/docs/sklearn-equivalents';
    const isHub = page.url === '/docs' || page.url === '/docs/apis';
    return {
      url: new URL(page.url, siteUrl).toString(),
      lastModified: getFileLastModified(path.join(docsRoot, 'content/docs', page.file.path)),
      changeFrequency: isHub ? ('weekly' as const) : ('monthly' as const),
      priority: isHub ? 0.9 : isGuide ? 0.8 : 0.7,
    };
  });

  const landingPages: MetadataRoute.Sitemap = [
    {
      url: `${siteUrl}/tools`,
      lastModified: appPageLastModified('tools/page.tsx'),
      changeFrequency: 'weekly',
      priority: 0.9,
    },
    {
      url: `${siteUrl}/tools/outlier-calculator`,
      lastModified: appPageLastModified('tools/outlier-calculator/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.85,
    },
    {
      url: `${siteUrl}/tools/polynomial-regression-calculator`,
      lastModified: appPageLastModified('tools/polynomial-regression-calculator/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.85,
    },
    {
      url: `${siteUrl}/tools/customer-segmentation`,
      lastModified: appPageLastModified('tools/customer-segmentation/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.85,
    },
    {
      url: `${siteUrl}/tools/confusion-matrix-calculator`,
      lastModified: appPageLastModified('tools/confusion-matrix-calculator/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.85,
    },
    {
      url: `${siteUrl}/tools/roc-auc-calculator`,
      lastModified: appPageLastModified('tools/roc-auc-calculator/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.85,
    },
    {
      url: `${siteUrl}/tools/linear-regression-calculator`,
      lastModified: appPageLastModified('tools/linear-regression-calculator/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.85,
    },
    {
      url: `${siteUrl}/tools/logistic-regression-calculator`,
      lastModified: appPageLastModified('tools/logistic-regression-calculator/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.85,
    },
    {
      url: `${siteUrl}/playground`,
      lastModified: appPageLastModified('playground/page.tsx'),
      changeFrequency: 'weekly',
      priority: 0.9,
    },
    {
      url: `${siteUrl}/playground/pca`,
      lastModified: appPageLastModified('playground/pca/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.8,
    },
    {
      url: `${siteUrl}/playground/knn`,
      lastModified: appPageLastModified('playground/knn/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.8,
    },
    {
      url: `${siteUrl}/playground/gradient-descent`,
      lastModified: appPageLastModified('playground/gradient-descent/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.8,
    },
    {
      url: `${siteUrl}/playground/kmeans`,
      lastModified: appPageLastModified('playground/kmeans/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.8,
    },
    {
      url: `${siteUrl}/playground/decision-tree`,
      lastModified: appPageLastModified('playground/decision-tree/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.8,
    },
    {
      url: `${siteUrl}/playground/random-forest`,
      lastModified: appPageLastModified('playground/random-forest/page.tsx'),
      changeFrequency: 'monthly',
      priority: 0.8,
    },
  ];

  return [
    {
      url: siteUrl,
      lastModified: appPageLastModified('(home)/page.tsx'),
      changeFrequency: 'weekly',
      priority: 1,
    },
    ...landingPages,
    ...docPages,
  ];
}
