import type { ReactNode } from 'react';
import Link from 'next/link';
import styles from './toolPage.module.css';

export type ToolFaq = {
  question: string;
  answer: string;
};

export type RelatedTool = {
  href: string;
  title: string;
  description: string;
};

type ToolPageLayoutProps = {
  name: string;
  description: string;
  pathname: string;
  tool: ReactNode;
  children: ReactNode;
  faq: ToolFaq[];
  related: RelatedTool[];
  sectionName?: string;
  sectionPath?: string;
  eyebrow?: string;
  activityLabel?: string;
};

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

function safeJsonLd(value: unknown) {
  return JSON.stringify(value).replace(/</g, '\\u003c');
}

export function ToolPageLayout({
  name,
  description,
  pathname,
  tool,
  children,
  faq,
  related,
  sectionName = 'Tools',
  sectionPath = '/tools',
  eyebrow = 'Free browser-based ML tool',
  activityLabel = 'Live calculation',
}: ToolPageLayoutProps) {
  const url = `${siteUrl}${pathname}`;
  const jsonLd = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'WebApplication',
        '@id': `${url}#application`,
        name,
        description,
        url,
        applicationCategory: ['EducationalApplication', 'UtilitiesApplication'],
        operatingSystem: 'Any',
        browserRequirements: 'Requires JavaScript. Runs locally in a modern browser.',
        isAccessibleForFree: true,
        offers: {
          '@type': 'Offer',
          price: '0',
          priceCurrency: 'USD',
        },
        featureList: ['Instant client-side calculation', 'CSV data input', 'Interactive visualization', 'JavaScript and Python code examples'],
        softwareHelp: `${siteUrl}/docs`,
        provider: {
          '@type': 'Organization',
          name: 'Kanaries',
          url: 'https://kanaries.net/',
        },
      },
      {
        '@type': 'FAQPage',
        '@id': `${url}#faq`,
        mainEntity: faq.map((item) => ({
          '@type': 'Question',
          name: item.question,
          acceptedAnswer: {
            '@type': 'Answer',
            text: item.answer,
          },
        })),
      },
    ],
  };

  return (
    <main className={styles.page}>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: safeJsonLd(jsonLd) }} />
      <div className={styles.shell}>
        <nav className={styles.breadcrumb} aria-label="Breadcrumb">
          <Link href="/">@kanaries/ml</Link>
          <span aria-hidden="true">/</span>
          <Link href={sectionPath}>{sectionName}</Link>
          <span aria-hidden="true">/</span>
          <span aria-current="page">{name}</span>
        </nav>

        <header className={styles.hero}>
          <div className={styles.eyebrow}>{eyebrow}</div>
          <h1 className={styles.title}>{name}</h1>
          <p className={styles.description}>{description}</p>
          <div className={styles.trustRow} aria-label="Tool benefits">
            <span>Free to use</span>
            <span>No upload or sign-in</span>
            <span>Runs entirely in your browser</span>
          </div>
        </header>

        <section className={styles.toolFrame} aria-label={`${name} interactive tool`}>
          <div className={styles.toolHeader}>
            <span className={styles.live}>{activityLabel}</span>
            <span>Powered by @kanaries/ml</span>
          </div>
          {tool}
        </section>

        <article className={styles.article}>{children}</article>

        <section className={styles.faq} aria-labelledby="faq-heading">
          <div className={styles.kicker}>Frequently asked questions</div>
          <h2 id="faq-heading">Questions about {name.toLowerCase()}</h2>
          {faq.map((item) => (
            <details key={item.question}>
              <summary>{item.question}</summary>
              <p>{item.answer}</p>
            </details>
          ))}
        </section>

        <section className={styles.related} aria-labelledby="related-heading">
          <div className={styles.kicker}>Continue exploring</div>
          <h2 id="related-heading">Related tools and documentation</h2>
          <div className={styles.relatedGrid}>
            {related.map((item) => (
              <Link className={styles.relatedCard} href={item.href} key={item.href}>
                <strong>{item.title}</strong>
                <span>{item.description}</span>
              </Link>
            ))}
          </div>
        </section>

        <footer className={styles.footer}>
          <span>Powered by @kanaries/ml — the scikit-learn-style ML library for JavaScript</span>
          <span className={styles.footerLinks}>
            <a href="https://www.npmjs.com/package/@kanaries/ml">npm</a>
            <a href="https://github.com/Kanaries/ml">GitHub</a>
          </span>
        </footer>
      </div>
    </main>
  );
}
