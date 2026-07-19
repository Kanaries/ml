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
import { getDisplayTitle, getDisplayTitleNode } from '@/lib/title';
import { getMarkdownPath } from '@/lib/agent-docs';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

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
  const articleLd = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: page.data.title,
    description: page.data.description,
    url: canonicalUrl,
    inLanguage: 'en',
    author: {
      '@type': 'Organization',
      name: 'Kanaries',
    },
    publisher: {
      '@type': 'Organization',
      name: 'Kanaries',
      url: 'https://kanaries.net/',
    },
  };

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(articleLd) }} />
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
  const metadataTitle =
    isAlgorithmPage && !/javascript|typescript/i.test(title)
      ? `${title} in JavaScript and TypeScript`
      : title;

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
    },
    twitter: {
      card: 'summary_large_image',
      title: metadataTitle,
      description,
    },
  };
}
