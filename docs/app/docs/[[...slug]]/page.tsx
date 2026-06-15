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
import { buildSeoProfile } from '@/lib/seo';
import { getDisplayTitle, getDisplayTitleNode } from '@/lib/title';

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
  const seoProfile = buildSeoProfile({
    title: page.data.title,
    description: page.data.description,
    slug,
  });
  const articleLd = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: page.data.title,
    description: page.data.description,
    url: canonicalUrl,
    inLanguage: 'en',
    articleSection: seoProfile.moduleName,
    keywords: seoProfile.keywords,
    author: {
      '@type': 'Organization',
      name: 'Kanaries',
    },
    publisher: {
      '@type': 'Organization',
      name: 'Kanaries',
      url: 'https://kanaries.net/',
    },
    about: seoProfile.keywords,
  };
  const faqLd = {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: seoProfile.faq.map((item) => ({
      '@type': 'Question',
      name: item.question,
      acceptedAnswer: {
        '@type': 'Answer',
        text: item.answer,
      },
    })),
  };

  return (
    <>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(articleLd) }} />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(faqLd) }} />
      <DocsPage toc={displayToc} full={page.data.full}>
        <DocsTitle>{displayTitle}</DocsTitle>
        <DocsDescription>{page.data.description}</DocsDescription>
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
          <section className="mt-12 rounded-lg border border-fd-border bg-fd-muted/40 p-6">
            <h2 className="text-lg font-semibold">FAQ</h2>
            <div className="mt-4 space-y-3">
              {seoProfile.faq.map((item) => (
                <details key={item.question} className="rounded-md border border-fd-border bg-fd-background p-4">
                  <summary className="cursor-pointer font-medium">{item.question}</summary>
                  <p className="mt-2 text-sm text-fd-muted-foreground">{item.answer}</p>
                </details>
              ))}
            </div>
          </section>
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
  const canonicalUrl = new URL(path, siteUrl).toString();
  const title = page.data.title;
  const description = page.data.description;
  const seoProfile = buildSeoProfile({ title, description, slug });
  const isAlgorithmPage = slug[0] === 'apis' && slug.length >= 3;
  const metadataTitle =
    isAlgorithmPage && !/javascript|typescript/i.test(title)
      ? `${title} in JavaScript and TypeScript`
      : title;

  return {
    title: metadataTitle,
    description,
    keywords: seoProfile.keywords,
    alternates: {
      canonical: canonicalUrl,
    },
    openGraph: {
      type: 'article',
      title: metadataTitle,
      description,
      url: canonicalUrl,
      siteName: '@kanaries/ml',
      tags: seoProfile.keywords,
    },
    twitter: {
      card: 'summary',
      title: metadataTitle,
      description,
    },
    other: {
      'seo:primary_keyword': seoProfile.primaryKeyword,
      'seo:secondary_keywords': seoProfile.secondaryKeywords.join(', '),
      'seo:long_tail_keywords': seoProfile.longTailKeywords.join(', '),
    },
  };
}
