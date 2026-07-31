import type { Metadata } from 'next';
import Link from 'next/link';
import styles from '@/components/tools/toolPage.module.css';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export const metadata: Metadata = {
  title: { absolute: 'Free Machine Learning Calculators & Visualizations — @kanaries/ml' },
  description:
    'Use free, private machine learning calculators in your browser. Calculate confusion-matrix metrics, fit logistic regression, visualize models, and copy JavaScript or Python code.',
  alternates: { canonical: `${siteUrl}/tools` },
  openGraph: {
    title: 'Free Machine Learning Calculators & Visualizations',
    description: 'Interactive, browser-native ML tools powered by @kanaries/ml.',
    url: `${siteUrl}/tools`,
  },
};

const tools = [
  {
    href: '/tools/confusion-matrix-calculator',
    title: 'Confusion Matrix & F1 Calculator',
    description: 'Calculate accuracy, precision, recall, specificity, macro/micro/weighted F1, MCC, and Cohen’s kappa for binary or multiclass predictions.',
  },
  {
    href: '/tools/logistic-regression-calculator',
    title: 'Logistic Regression Calculator',
    description: 'Fit pasted CSV data, inspect coefficients and odds ratios, visualize a sigmoid or decision boundary, and predict new samples.',
  },
];

export default function ToolsIndexPage() {
  return (
    <main className={styles.page}>
      <div className={styles.shell}>
        <nav className={styles.breadcrumb} aria-label="Breadcrumb">
          <Link href="/">@kanaries/ml</Link><span aria-hidden="true">/</span><span aria-current="page">Tools</span>
        </nav>
        <header className={styles.hero}>
          <div className={styles.eyebrow}>Browser-native machine learning</div>
          <h1 className={styles.title}>Free ML calculators and interactive tools</h1>
          <p className={styles.description}>
            Paste data, calculate results instantly, and see how the model works. Everything runs locally in your browser,
            with reproducible JavaScript and Python examples for the next step.
          </p>
          <div className={styles.trustRow}>
            <span>No account</span><span>No server upload</span><span>Open-source calculation engine</span>
          </div>
        </header>
        <section className={styles.related} aria-label="Available machine learning tools">
          <div className={styles.relatedGrid}>
            {tools.map((tool) => (
              <Link className={styles.relatedCard} href={tool.href} key={tool.href}>
                <strong>{tool.title}</strong>
                <span>{tool.description}</span>
              </Link>
            ))}
          </div>
        </section>
        <article className={styles.article}>
          <h2>Machine learning tools that work where JavaScript works</h2>
          <p>
            These calculators are interactive examples of <code>@kanaries/ml</code>, a scikit-learn-style machine learning
            library for JavaScript and TypeScript. Unlike a form that sends your dataset to a calculation server, each tool
            parses data, fits models, evaluates metrics, and draws its visualization in the current browser tab.
          </p>
          <p>
            Start with the confusion matrix calculator when you already have model predictions. Use the logistic regression
            calculator when you have labeled rows and want to fit an interpretable binary classifier. Each page includes a
            practical guide, metric explanations, downloadable results, and code that maps the browser calculation to
            <Link href="/docs"> @kanaries/ml documentation</Link> and familiar scikit-learn calls.
          </p>
        </article>
      </div>
    </main>
  );
}
