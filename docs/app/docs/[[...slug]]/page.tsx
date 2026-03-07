import { source } from '@/lib/source';
import {
  DocsPage,
  DocsBody,
  DocsDescription,
  DocsTitle,
} from 'fumadocs-ui/page';
import { notFound } from 'next/navigation';
import { createRelativeLink } from 'fumadocs-ui/mdx';
import { getMDXComponents } from '@/mdx-components';
import type { Metadata } from 'next';
import { buildSeoProfile } from '@/lib/seo';

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
      <DocsPage toc={page.data.toc} full={page.data.full}>
        <DocsTitle>{page.data.title}</DocsTitle>
        <DocsDescription>{page.data.description}</DocsDescription>
        <DocsBody>
          <MDXContent
            components={getMDXComponents({
              // this allows you to link to other pages with relative file paths
              a: createRelativeLink(source, page),
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

  return {
    title,
    description,
    keywords: seoProfile.keywords,
    alternates: {
      canonical: canonicalUrl,
    },
    openGraph: {
      type: 'article',
      title,
      description,
      url: canonicalUrl,
      siteName: '@kanaries/ml',
      tags: seoProfile.keywords,
    },
    twitter: {
      card: 'summary',
      title,
      description,
    },
    other: {
      'seo:primary_keyword': seoProfile.primaryKeyword,
      'seo:secondary_keywords': seoProfile.secondaryKeywords.join(', '),
      'seo:long_tail_keywords': seoProfile.longTailKeywords.join(', '),
    },
  };
}
