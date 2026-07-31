import type { Metadata } from 'next';
import Link from 'next/link';
import styles from '@/components/tools/toolPage.module.css';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export const metadata: Metadata = {
  title: { absolute: 'Interactive Machine Learning Playgrounds in JavaScript — @kanaries/ml' },
  description: 'Explore PCA, KNN, gradient descent, and K-Means with interactive browser visualizations powered by @kanaries/ml. No install or backend required.',
  alternates: { canonical: `${siteUrl}/playground` },
  openGraph: {
    title: 'Interactive Machine Learning Playgrounds',
    description: 'Learn machine learning by changing data and watching algorithms respond in your browser.',
    url: `${siteUrl}/playground`,
  },
};

const playgrounds = [
  { href: '/playground/pca', title: 'PCA visualization', description: 'Paste multidimensional data, inspect explained variance, and rotate a valid 2D loading basis.' },
  { href: '/playground/knn', title: 'KNN visualization', description: 'Draw labeled points, move a query, and compare k, distance metrics, and weighted voting.' },
  { href: '/playground/gradient-descent', title: 'Gradient descent visualization', description: 'Compare SGD, Momentum, and Adam trajectories on convex and non-convex objectives.' },
  { href: '/playground/kmeans', title: 'K-Means visualization', description: 'Step through assignment and centroid updates on blobs, moons, and uneven clusters.' },
  { href: '/playground/decision-tree', title: 'Decision tree visualization', description: 'Tune a classifier and connect its decision regions to the fitted rule hierarchy.' },
  { href: '/playground/random-forest', title: 'Random Forest visualization', description: 'Compare a single tree with a bootstrap-aggregated forest on noisy data.' },
];

export default function PlaygroundIndexPage() {
  return (
    <main className={styles.page}>
      <div className={styles.shell}>
        <nav className={styles.breadcrumb} aria-label="Breadcrumb">
          <Link href="/">@kanaries/ml</Link><span aria-hidden="true">/</span><span aria-current="page">Playground</span>
        </nav>
        <header className={styles.hero}>
          <div className={styles.eyebrow}>Learn by changing the model</div>
          <h1 className={styles.title}>Interactive machine learning playgrounds</h1>
          <p className={styles.description}>Move points, change parameters, and see the algorithm update immediately. Every playground runs locally with JavaScript and links the visualization back to a reusable API.</p>
          <div className={styles.trustRow}><span>No setup</span><span>Runs in your browser</span><span>Built for teaching and exploration</span></div>
        </header>
        <section className={styles.related} aria-label="Available algorithm playgrounds">
          <div className={styles.relatedGrid}>
            {playgrounds.map((item) => (
              <Link className={styles.relatedCard} href={item.href} key={item.href}>
                <strong>{item.title}</strong><span>{item.description}</span>
              </Link>
            ))}
          </div>
        </section>
        <article className={styles.article}>
          <h2>Browser-native algorithm visualizations</h2>
          <p>These playgrounds turn the JavaScript implementations in <code>@kanaries/ml</code> into inspectable learning tools. They are designed for students, teachers, frontend developers, and data practitioners who want to test an intuition without opening a notebook or sending data to a server.</p>
          <p>Start with PCA to understand dimensionality reduction, KNN to see local voting, gradient descent to compare optimizers, or K-Means to follow assignment and centroid updates. Continue to the <Link href="/tools">ML calculators</Link> when you need a numerical result rather than an algorithm demonstration.</p>
        </article>
      </div>
    </main>
  );
}
