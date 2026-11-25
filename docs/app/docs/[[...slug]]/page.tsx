import path from 'node:path';
import { source } from '@/lib/source';
import {
  DocsPage,
  DocsBody,
  DocsDescription,
  DocsTitle,
} from 'fumadocs-ui/page';
import { notFound } from 'next/navigation';
import { createRelativeLink } from 'fumadocs-ui/mdx';
import defaultMdxComponents from 'fumadocs-ui/mdx';
import { getMDXComponents } from '@/mdx-components';
import type { Metadata } from 'next';
import { getDisplayTitle, getDisplayTitleNode, getMetadataTitle } from '@/lib/title';
import { getMarkdownPath } from '@/lib/agent-docs';
import {
  DEFAULT_OG_IMAGE,
  SITE_URL,
  buildBreadcrumbLd,
  getFileLastModified,
  safeJsonLd,
  type BreadcrumbItem,
} from '@/lib/seo';

const siteUrl = SITE_URL;

function buildDocsBreadcrumbs(slug: string[], pageTitle: string): BreadcrumbItem[] {
  const items: BreadcrumbItem[] = [
    { name: '@kanaries/ml', path: '/' },
    { name: 'Docs', path: '/docs' },
  ];

  if (slug.length === 0) return items;

  for (let i = 0; i < slug.length - 1; i++) {
    const prefix = slug.slice(0, i + 1);
    const parentPage = source.getPage(prefix);
    const segmentLabel = parentPage
      ? getDisplayTitle(parentPage.data.title)
      : slug[i].replace(/[-_]/g, ' ');
    items.push({
      name: segmentLabel,
      path: `/docs/${prefix.join('/')}`,
    });
  }

  items.push({
    name: pageTitle,
    path: `/docs/${slug.join('/')}`,
  });

  return items;
}

export default async function Page(props: {
  params: Promise<{ slug?: string[] }>;
}) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const MDXContent = page.data.body;
  const slug = params.slug ?? [];
  const path = slug.length ? `/docs/${slug.join('/')}` : '/docs';
  const markdownPath = getMarkdownPath(page);
  const canonicalUrl = new URL(path, siteUrl).toString();
  const displayTitle = getDisplayTitle(page.data.title);
  const displayToc = page.data.toc
    .map((item) => {
      const title = getDisplayTitleNode(item.title);

      return {
        ...item,
        title,
      };
    })
    .filter((item) => item.title !== displayTitle);
  const sourceFilePath = path.join(process.cwd(), 'content/docs', page.file.path);
  const lastModified = getFileLastModified(sourceFilePath).toISOString();
  const isAlgorithmPage = slug[0] === 'apis' && slug.length >= 3;
  const headline = getMetadataTitle(page.data.title, isAlgorithmPage);
  const structuredData = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'TechArticle',
        '@id': `${canonicalUrl}#article`,
        headline,
        description: page.data.description,
        url: canonicalUrl,
        mainEntityOfPage: canonicalUrl,
        image: DEFAULT_OG_IMAGE.url,
        dateModified: lastModified,
        inLanguage: 'en',
        author: {
          '@type': 'Organization',
          name: 'Kanaries',
          url: 'https://kanaries.net/',
        },
        publisher: {
          '@type': 'Organization',
          name: 'Kanaries',
          url: 'https://kanaries.net/',
        },
      },
      buildBreadcrumbLd(buildDocsBreadcrumbs(slug, displayTitle || headline)),
    ],
  };

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: safeJsonLd(structuredData) }} />
      <DocsPage toc={displayToc} full={page.data.full}>
        <DocsTitle>{displayTitle}</DocsTitle>
        <DocsDescription>{page.data.description}</DocsDescription>
        <a
          href={markdownPath}
          className="mb-6 inline-flex text-sm font-medium text-fd-muted-foreground underline decoration-fd-border underline-offset-4 transition-colors hover:text-fd-foreground"
        >
          View as Markdown
        </a>
        <DocsBody>
          <MDXContent
            components={getMDXComponents({
              // this allows you to link to other pages with relative file paths
              a: createRelativeLink(source, page),
              h1: (props) => {
                const children = getDisplayTitleNode(props.children);

                if (children === displayTitle) return null;

                return defaultMdxComponents.h1({
                  ...props,
                  children,
                });
              },
            })}
          />
        </DocsBody>
      </DocsPage>
    </>
  );
}

export async function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: {
  params: Promise<{ slug?: string[] }>;
}): Promise<Metadata> {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const slug = params.slug ?? [];
  const path = slug.length ? `/docs/${slug.join('/')}` : '/docs';
  const markdownPath = getMarkdownPath(page);
  const canonicalUrl = new URL(path, siteUrl).toString();
  const title = page.data.title;
  const description = page.data.description;
  const isAlgorithmPage = slug[0] === 'apis' && slug.length >= 3;
  const metadataTitle = getMetadataTitle(title, isAlgorithmPage);

  return {
    title: metadataTitle,
    description,
    alternates: {
      canonical: canonicalUrl,
      types: {
        'text/markdown': new URL(markdownPath, siteUrl).toString(),
      },
    },
    openGraph: {
      type: 'article',
      title: metadataTitle,
      description,
      url: canonicalUrl,
      siteName: '@kanaries/ml',
      images: [DEFAULT_OG_IMAGE],
    },
    twitter: {
      card: 'summary_large_image',
      title: metadataTitle,
      description,
      images: [DEFAULT_OG_IMAGE.url],
    },
  };
}
