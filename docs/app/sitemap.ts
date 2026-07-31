import type { MetadataRoute } from 'next';
import { source } from '@/lib/source';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export default function sitemap(): MetadataRoute.Sitemap {
  const docPages = source.getPages().map((page) => ({
    url: new URL(page.url, siteUrl).toString(),
    changeFrequency: 'monthly' as const,
    priority: 0.7,
  }));

  const landingPages: MetadataRoute.Sitemap = [
    { url: `${siteUrl}/tools`, changeFrequency: 'weekly', priority: 0.9 },
    { url: `${siteUrl}/tools/confusion-matrix-calculator`, changeFrequency: 'monthly', priority: 1 },
    { url: `${siteUrl}/tools/logistic-regression-calculator`, changeFrequency: 'monthly', priority: 1 },
    { url: `${siteUrl}/playground`, changeFrequency: 'weekly', priority: 0.9 },
    { url: `${siteUrl}/playground/pca`, changeFrequency: 'monthly', priority: 1 },
    { url: `${siteUrl}/playground/knn`, changeFrequency: 'monthly', priority: 1 },
    { url: `${siteUrl}/playground/gradient-descent`, changeFrequency: 'monthly', priority: 1 },
    { url: `${siteUrl}/playground/kmeans`, changeFrequency: 'monthly', priority: 1 },
    { url: `${siteUrl}/playground/decision-tree`, changeFrequency: 'monthly', priority: 0.9 },
    { url: `${siteUrl}/playground/random-forest`, changeFrequency: 'monthly', priority: 0.8 },
  ];

  return [{ url: `${siteUrl}/`, changeFrequency: 'weekly', priority: 1 }, ...landingPages, ...docPages];
}
