import type { MetadataRoute } from 'next';
import { source } from '@/lib/source';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export default function sitemap(): MetadataRoute.Sitemap {
  const docPages = source.getPages().map((page) => ({
    url: new URL(page.url, siteUrl).toString(),
  }));

  return [{ url: `${siteUrl}/` }, ...docPages];
}
