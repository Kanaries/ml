import Link from 'next/link';
import Script from 'next/script';
import type { Metadata } from 'next';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export const metadata: Metadata = {
  title: 'Machine Learning in JavaScript (Browser & Node) — @kanaries/ml',
  description:
    'Train and deploy ML models in JavaScript with a scikit-learn-style API. Works in browser and Node.js. Fast examples, tiny bundles, docs for every algorithm.',
  alternates: {
    canonical: `${siteUrl}/`,
  },
  openGraph: {
    title: 'Machine Learning in JavaScript — @kanaries/ml',
    description: 'Train and deploy ML models in JavaScript (browser & Node).',
    url: `${siteUrl}/`,
  },
};

const WHY_JS = [
  {
    title: 'Run models in the browser',
    description: 'Ship zero-backend ML experiences with full client-side inference and privacy by default.',
  },
  {
    title: 'One language front to back',
    description: 'Build training pipelines and production inference with Node.js and modern frameworks.',
  },
  {
    title: 'Interactive ML UIs',
    description: 'Create exploratory dashboards that react instantly without server round-trips.',
  },
  {
    title: 'Tiny bundles & WASM-ready',
    description: 'Tree-shakable ESM builds, typed arrays and Web Worker friendly execution.',
  },
];

const USE_CASES = [
  {
    title: 'Tabular',
    description: 'Client-side churn prediction, lead scoring, and data capture scoring widgets.',
  },
  {
    title: 'Text',
    description: 'Intent detection, TF–IDF pipelines, and simple classifiers entirely in JS.',
  },
  {
    title: 'Clustering & DR',
    description: 'k-means, PCA, and dimensionality reduction for visualization dashboards.',
  },
  {
    title: 'Time series',
    description: 'Prototype forecasts in Node.js with classical regression and smoothing.',
  },
];

const FEATURES = [
  {
    title: 'Familiar API',
    description: 'Just like scikit-learn: fit, predict, pipelines, transformers, and metrics.',
  },
  {
    title: 'Browser & Node support',
    description: 'ESM-first distribution runs anywhere JavaScript does.',
  },
  {
    title: 'Fast numerics',
    description: 'Typed arrays and WASM-friendly architecture keep memory low and performance high.',
  },
  {
    title: 'Model persistence',
    description: 'Serialize trained pipelines to JSON and reload them in any runtime.',
  },
  {
    title: 'Docs for every algorithm',
    description: 'Runnable guides for each estimator shorten onboarding time.',
  },
  {
    title: 'Tiny footprint',
    description: 'Import only what you need with tree-shakable modules.',
  },
];

const ALGORITHMS: Record<string, { label: string; href: string }[]> = {
  Classification: [
    { label: 'Logistic Regression', href: '/docs/algorithms/logistic-regression/' },
    { label: 'Linear SVM', href: '/docs/algorithms/linear-svm/' },
    { label: 'Random Forest', href: '/docs/algorithms/random-forest/' },
    { label: 'Naive Bayes', href: '/docs/algorithms/naive-bayes/' },
    { label: 'k-NN', href: '/docs/algorithms/knn/' },
  ],
  Regression: [
    { label: 'Linear Regression', href: '/docs/algorithms/linear-regression/' },
    { label: 'Ridge & Lasso', href: '/docs/algorithms/ridge-lasso/' },
    { label: 'Random Forest Regressor', href: '/docs/algorithms/random-forest-regressor/' },
    { label: 'k-NN Regressor', href: '/docs/algorithms/knn-regressor/' },
  ],
  Clustering: [
    { label: 'k-Means', href: '/docs/algorithms/kmeans/' },
    { label: 'DBSCAN', href: '/docs/algorithms/dbscan/' },
  ],
  'Dimensionality Reduction': [
    { label: 'PCA', href: '/docs/algorithms/pca/' },
    { label: 'Truncated SVD', href: '/docs/algorithms/truncated-svd/' },
  ],
  Preprocessing: [
    { label: 'StandardScaler', href: '/docs/data/preprocessing/standard-scaler/' },
    { label: 'MinMaxScaler', href: '/docs/data/preprocessing/minmax-scaler/' },
    { label: 'OneHotEncoder', href: '/docs/data/preprocessing/onehot-encoder/' },
    { label: 'Train/Test Split', href: '/docs/data/preprocessing/train-test-split/' },
  ],
};

const HOW_IT_WORKS = [
  {
    title: 'Prepare data',
    description: 'Clean and transform with scalers, encoders, and pipeline helpers.',
  },
  {
    title: 'Fit a model',
    description: 'Use the sklearn-style API to train estimators on typed arrays or plain arrays.',
  },
  {
    title: 'Deploy anywhere',
    description: 'Persist to JSON, load in browsers or Node.js, and call predict in milliseconds.',
  },
];

const FAQ_ITEMS = [
  {
    question: 'Is JavaScript fast enough for ML?',
    answer:
      'Yes—for small to medium datasets and interactive experiences, @kanaries/ml delivers responsive inference. Use Web Workers to keep UIs snappy and review our benchmarks for throughput guidance.',
  },
  {
    question: 'Does it run fully in the browser?',
    answer:
      'Absolutely. All supported algorithms can execute in modern browsers and in Node.js environments.',
  },
  {
    question: 'Can I save and load models?',
    answer:
      'Every estimator can serialize parameters to JSON. Reload them in any JS runtime and resume predictions instantly.',
  },
  {
    question: 'How close is the API to scikit-learn?',
    answer:
      'We follow scikit-learn naming and options wherever practical. The migration guide highlights the rare differences.',
  },
  {
    question: 'How big is the bundle?',
    answer:
      'The core stays well below typical UI bundle budgets and each algorithm tree-shakes to just the code you import. See the benchmarks section for the latest size numbers.',
  },
  {
    question: 'Does it support Web Workers or WASM?',
    answer:
      'Yes. The runtime is designed for off-main-thread execution and plays nicely with WASM-backed kernels.',
  },
];

const COMMUNITY_LINKS = [
  { label: 'Star on GitHub', href: 'https://github.com/Kanaries/ml' },
  { label: 'Join Discord', href: 'https://discord.gg/kanaries' },
  { label: 'Subscribe for updates', href: 'https://kanaries.net/#newsletter' },
];

const JSON_LD = {
  '@context': 'https://schema.org',
  '@type': ['SoftwareSourceCode', 'WebSite'],
  name: '@kanaries/ml',
  description:
    'A scikit-learn-style JavaScript machine learning library for browser and Node.js. Familiar API, tiny bundles, runnable examples.',
  url: `${siteUrl}/`,
  programmingLanguage: 'JavaScript',
  codeRepository: 'https://github.com/Kanaries/ml',
  license: 'https://opensource.org/licenses/MIT',
  isAccessibleForFree: true,
  publisher: {
    '@type': 'Organization',
    name: 'Kanaries',
    url: 'https://kanaries.net/',
  },
  inLanguage: 'en',
  keywords: [
    'machine learning in JavaScript',
    'JavaScript machine learning library',
    'scikit-learn for JavaScript',
    'browser machine learning',
    'Node.js machine learning',
  ],
  sameAs: [
    'https://www.npmjs.com/package/@kanaries/ml',
    'https://twitter.com/KanariesData',
    'https://github.com/Kanaries/ml',
  ],
};

const FAQ_LD = {
  '@context': 'https://schema.org',
  '@type': 'FAQPage',
  mainEntity: FAQ_ITEMS.slice(0, 5).map((item) => ({
    '@type': 'Question',
    name: item.question,
    acceptedAnswer: {
      '@type': 'Answer',
      text: item.answer,
    },
  })),
};

export default function HomePage() {
  return (
    <>
      <Script
        id="json-ld-software"
        type="application/ld+json"
        strategy="afterInteractive"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(JSON_LD) }}
      />
      <Script
        id="json-ld-faq"
        type="application/ld+json"
        strategy="afterInteractive"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(FAQ_LD) }}
      />
      <main className="flex flex-1 flex-col gap-20 pb-24 pt-16">
        <section className="mx-auto max-w-5xl px-6 text-center">
          <div className="space-y-6">
            <h1 className="text-balance bg-gradient-to-r from-purple-500 via-pink-500 to-orange-400 bg-clip-text text-4xl font-black text-transparent sm:text-6xl">
              Machine Learning in JavaScript — scikit-learn style
            </h1>
            <p className="mx-auto max-w-2xl text-lg text-fd-muted-foreground sm:text-xl">
              A lightweight JS machine learning library for <strong>browser</strong> and <strong>Node.js</strong>. Familiar
              scikit-learn API, zero-install demos, production-ready builds.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Link className="rounded-md bg-fd-foreground px-6 py-3 font-semibold text-white shadow-lg transition hover:opacity-90" href="/playground">
                Try in Browser
              </Link>
              <Link className="rounded-md border border-fd-border px-6 py-3 font-semibold text-fd-foreground transition hover:bg-fd-foreground/5" href="/docs/get-started">
                Get Started (npm)
              </Link>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-fd-muted-foreground">
              <Link href="https://www.npmjs.com/package/@kanaries/ml" className="flex items-center gap-2" target="_blank" rel="noreferrer">
                <span aria-hidden>📦</span> npm install @kanaries/ml
              </Link>
              <Link href="https://github.com/Kanaries/ml" className="flex items-center gap-2" target="_blank" rel="noreferrer">
                <span aria-hidden>⭐</span> GitHub stars & contributors
              </Link>
              <span className="flex items-center gap-2">
                <span aria-hidden>📉</span> Core &lt; 50 kB gzipped
              </span>
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-6xl px-6">
          <h2 className="text-2xl font-semibold">Why JavaScript for ML?</h2>
          <div className="mt-6 grid gap-6 md:grid-cols-2">
            {WHY_JS.map((item) => (
              <div key={item.title} className="rounded-lg border border-fd-border bg-fd-background p-6 shadow-sm">
                <h3 className="text-lg font-semibold">{item.title}</h3>
                <p className="mt-2 text-sm text-fd-muted-foreground">{item.description}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-6xl px-6">
          <h2 className="text-2xl font-semibold">What you can build</h2>
          <div className="mt-6 grid gap-6 md:grid-cols-2">
            {USE_CASES.map((useCase) => (
              <div key={useCase.title} className="rounded-lg border border-dashed border-fd-border bg-fd-background p-6">
                <h3 className="text-lg font-semibold">{useCase.title}</h3>
                <p className="mt-2 text-sm text-fd-muted-foreground">{useCase.description}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-6xl px-6">
          <h2 className="text-2xl font-semibold">Key features</h2>
          <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((feature) => (
              <div key={feature.title} className="rounded-lg border border-fd-border bg-fd-background p-6">
                <h3 className="text-lg font-semibold">{feature.title}</h3>
                <p className="mt-2 text-sm text-fd-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-5xl px-6">
          <h2 className="text-2xl font-semibold">Quick start</h2>
          <div className="mt-6 space-y-6">
            <div>
              <p className="text-sm font-semibold text-fd-muted-foreground">Install</p>
              <pre className="mt-2 overflow-x-auto rounded-lg bg-fd-muted p-4 text-left text-sm text-fd-foreground">
                <code>yarn add @kanaries/ml</code>
              </pre>
            </div>
            <div>
              <p className="text-sm font-semibold text-fd-muted-foreground">Node.js</p>
              <pre className="mt-2 overflow-x-auto rounded-lg bg-fd-muted p-4 text-left text-sm text-fd-foreground">
                <code>{`import { StandardScaler, LogisticRegression } from '@kanaries/ml';

const X = [
  [5.1, 3.5],
  [4.9, 3.0],
  [7.0, 3.2],
];
const y = [0, 0, 1];

const scaler = new StandardScaler();
const Xs = scaler.fitTransform(X);
const clf = new LogisticRegression({ maxIter: 200 }).fit(Xs, y);

console.log(clf.predict([[6.1, 3.1]]));
`}</code>
              </pre>
            </div>
            <div>
              <details className="rounded-lg border border-fd-border bg-fd-background p-4">
                <summary className="cursor-pointer font-semibold">Use from a CDN (browser)</summary>
                <pre className="mt-4 overflow-x-auto rounded-lg bg-fd-muted p-4 text-left text-sm text-fd-foreground">
                  <code>{`<script type="module">
  import { KMeans } from 'https://cdn.skypack.dev/@kanaries/ml';

  const km = new KMeans({ k: 3 }).fit([
    [1, 1],
    [1.2, 1.1],
    [5, 5],
  ]);

  console.log(km.predict([[1.1, 1]]));
</script>`}</code>
                </pre>
              </details>
            </div>
            <div className="flex flex-wrap gap-4">
              <Link className="rounded-md border border-fd-border px-4 py-2 text-sm font-semibold transition hover:bg-fd-foreground/5" href="https://stackblitz.com/github/Kanaries/ml" target="_blank" rel="noreferrer">
                Open in StackBlitz
              </Link>
              <Link className="rounded-md border border-fd-border px-4 py-2 text-sm font-semibold transition hover:bg-fd-foreground/5" href="https://codesandbox.io/p/github/Kanaries/ml" target="_blank" rel="noreferrer">
                Open in CodeSandbox
              </Link>
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-6xl px-6">
          <h2 className="text-2xl font-semibold">Supported algorithms</h2>
          <div className="mt-6 grid gap-8 md:grid-cols-2">
            {Object.entries(ALGORITHMS).map(([group, items]) => (
              <div key={group}>
                <h3 className="text-lg font-semibold">{group}</h3>
                <ul className="mt-2 space-y-2 text-sm">
                  {items.map((item) => (
                    <li key={item.label}>
                      <Link className="text-fd-foreground underline-offset-4 hover:underline" href={item.href}>
                        {item.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-5xl px-6">
          <h2 className="text-2xl font-semibold">Benchmarks & footprint</h2>
          <div className="mt-4 space-y-4 text-sm text-fd-muted-foreground">
            <p>
              Core bundle <strong>&lt; 50 kB gzipped</strong>; individual algorithms typically stay below <strong>15 kB</strong>.
              Use Web Workers for CPU-heavy tasks and opt into the WASM build for additional throughput.
            </p>
            <p>
              Explore performance charts for training time vs. samples and inference vs. feature counts in the dedicated
              benchmarks section.
            </p>
            <Link className="inline-flex items-center gap-2 text-fd-foreground underline-offset-4 hover:underline" href="/benchmarks/">
              See all benchmarks →
            </Link>
          </div>
        </section>

        <section className="mx-auto max-w-5xl px-6">
          <h2 className="text-2xl font-semibold">How it works</h2>
          <div className="mt-6 grid gap-6 sm:grid-cols-3">
            {HOW_IT_WORKS.map((step, index) => (
              <div key={step.title} className="rounded-lg border border-fd-border bg-fd-background p-6 text-center">
                <span className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-fd-foreground text-sm font-bold text-white">
                  {index + 1}
                </span>
                <h3 className="mt-4 text-lg font-semibold">{step.title}</h3>
                <p className="mt-2 text-sm text-fd-muted-foreground">{step.description}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-5xl px-6">
          <div className="rounded-xl border border-fd-border bg-gradient-to-br from-fd-background via-fd-background to-fd-muted p-8">
            <h2 className="text-2xl font-semibold">Compare with scikit-learn</h2>
            <p className="mt-2 text-sm text-fd-muted-foreground">
              Understand parameter parity, migration tips, and when it makes sense to stay in Python versus delivering ML
              experiences directly in the browser.
            </p>
            <Link className="mt-4 inline-flex items-center gap-2 text-fd-foreground underline-offset-4 hover:underline" href="/compare/sklearn/">
              Migration guide →
            </Link>
          </div>
        </section>

        <section className="mx-auto max-w-5xl px-6">
          <h2 className="text-2xl font-semibold">FAQ</h2>
          <div className="mt-6 space-y-4">
            {FAQ_ITEMS.map((item) => (
              <details key={item.question} className="group rounded-lg border border-fd-border bg-fd-background p-4">
                <summary className="cursor-pointer text-lg font-semibold">{item.question}</summary>
                <p className="mt-2 text-sm text-fd-muted-foreground">{item.answer}</p>
              </details>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-5xl px-6">
          <h2 className="text-2xl font-semibold">Join the community</h2>
          <ul className="mt-6 grid gap-4 sm:grid-cols-3">
            {COMMUNITY_LINKS.map((link) => (
              <li key={link.label}>
                <Link className="flex h-full items-center justify-center rounded-lg border border-fd-border bg-fd-background px-4 py-3 text-center text-sm font-semibold transition hover:bg-fd-foreground/5" href={link.href} target="_blank" rel="noreferrer">
                  {link.label}
                </Link>
              </li>
            ))}
          </ul>
        </section>

        <section className="mx-auto max-w-5xl px-6 text-center">
          <div className="rounded-3xl border border-fd-border bg-fd-background p-10 shadow-lg">
            <h2 className="text-3xl font-bold">Build ML in JavaScript today</h2>
            <p className="mt-3 text-lg text-fd-muted-foreground">
              Deploy machine learning models anywhere JavaScript runs with @kanaries/ml.
            </p>
            <div className="mt-6 flex flex-wrap items-center justify-center gap-4">
              <Link className="rounded-md bg-fd-foreground px-6 py-3 font-semibold text-white transition hover:opacity-90" href="/playground">
                Try in Browser
              </Link>
              <Link className="rounded-md border border-fd-border px-6 py-3 font-semibold text-fd-foreground transition hover:bg-fd-foreground/5" href="/docs/get-started">
                Read the Docs
              </Link>
              <Link className="rounded-md border border-fd-border px-6 py-3 font-semibold text-fd-foreground transition hover:bg-fd-foreground/5" href="https://www.npmjs.com/package/@kanaries/ml" target="_blank" rel="noreferrer">
                npm install @kanaries/ml
              </Link>
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
