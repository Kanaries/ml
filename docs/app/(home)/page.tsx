import './home.css';
import Link from 'next/link';
import Script from 'next/script';
import type { Metadata } from 'next';
import { CopyButton } from '@/components/copyButton';

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
    title: 'Tiny, tree-shakable bundles',
    description: 'Tree-shakable ESM builds, typed arrays, and Web Worker friendly execution via asyncMode.',
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
    description: 'Typed arrays and tight inner loops keep memory low and performance high.',
  },
  {
    title: 'Built-in metrics',
    description: 'Accuracy, precision/recall, F1, ROC-AUC, R², and confusion matrices ship in the metrics module.',
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
  'Linear & SVM': [
    { label: 'Linear module', href: '/docs/apis/linear' },
    { label: 'LinearRegression', href: '/docs/apis/linear/linearRegression' },
    { label: 'LogisticRegression', href: '/docs/apis/linear/logisticRegression' },
    { label: 'SVM module', href: '/docs/apis/svm' },
    { label: 'SVC / NuSVC / LinearSVC', href: '/docs/apis/svm/SVC' },
  ],
  'Tree & Ensemble': [
    { label: 'Tree module', href: '/docs/apis/tree' },
    { label: 'DecisionTreeClassifier', href: '/docs/apis/tree/decisionTreeClassifier' },
    { label: 'ExtraTreeClassifier', href: '/docs/apis/tree/extraTreeClassifier' },
    { label: 'Ensemble module', href: '/docs/apis/ensemble' },
    { label: 'IsolationForest / AdaBoost', href: '/docs/apis/ensemble/iforest' },
  ],
  'Neighbors & Clustering': [
    { label: 'Neighbors module', href: '/docs/apis/neighbors' },
    { label: 'KNearestNeighbors', href: '/docs/apis/neighbors/knn' },
    { label: 'BallTree / KDTree', href: '/docs/apis/neighbors/ballTree' },
    { label: 'Clusters module', href: '/docs/apis/clusters' },
    { label: 'KMeans', href: '/docs/apis/clusters/kmeans' },
    { label: 'DBSCAN', href: '/docs/apis/clusters/dbscan' },
    { label: 'OPTICS', href: '/docs/apis/clusters/optics' },
    { label: 'HDBSCAN', href: '/docs/apis/clusters/hdbscan' },
    { label: 'MeanShift', href: '/docs/apis/clusters/meanShift' },
  ],
  'Decomposition & Manifold': [
    { label: 'Decomposition module', href: '/docs/apis/decomposition' },
    { label: 'PCA / SparsePCA / TruncatedSVD', href: '/docs/apis/decomposition/pca' },
    { label: 'Manifold module', href: '/docs/apis/manifold' },
    { label: 'MDS / LLE / TSNE / SpectralEmbedding', href: '/docs/apis/manifold/MDS' },
  ],
  'Bayes, Semi-Supervised & Utils': [
    { label: 'Bayes module', href: '/docs/apis/bayes' },
    { label: 'BernoulliNB / CategoricalNB', href: '/docs/apis/bayes/bernoulliNB' },
    { label: 'SemiSupervised module', href: '/docs/apis/semi_supervised' },
    { label: 'LabelPropagation / LabelSpreading', href: '/docs/apis/semi_supervised/labelPropagation' },
    { label: 'Utils module (asyncMode, trainTestSplit, preprocessing, model selection)', href: '/docs/apis/utils' },
  ],
};

const HOW_IT_WORKS = [
  {
    title: 'Prepare data',
    description: 'Clean and transform with scalers, encoders, and pipeline helpers.',
    code: 'scaler.fitTransform(X)',
  },
  {
    title: 'Fit a model',
    description: 'Use the sklearn-style API to train estimators on typed arrays or plain arrays.',
    code: 'model.fit(X, y)',
  },
  {
    title: 'Deploy anywhere',
    description: 'Run the same fit/predict code in browsers, Node.js, and edge functions.',
    code: 'model.predict(Xnew)',
  },
];

const FAQ_ITEMS = [
  {
    question: 'Is JavaScript fast enough for ML?',
    answer:
      'Yes—for small to medium datasets and interactive experiences, @kanaries/ml delivers responsive inference. Wrap heavy training or inference in utils.asyncMode to run it off the main thread and keep UIs snappy.',
  },
  {
    question: 'Does it run fully in the browser?',
    answer:
      'Absolutely. All supported algorithms can execute in modern browsers and in Node.js environments.',
  },
  {
    question: 'Can I save and load models?',
    answer:
      'There is no built-in serialization yet. For now you can re-fit from stored training data, or read a fitted estimator’s parameters and reconstruct it manually.',
  },
  {
    question: 'How close is the API to scikit-learn?',
    answer:
      'We follow scikit-learn naming and options wherever practical. Note that some estimators take positional constructor arguments rather than keyword options, so check each algorithm page for the exact signature.',
  },
  {
    question: 'How big is the bundle?',
    answer:
      'The full library is about 32 kB gzipped, and each algorithm tree-shakes to just the code you import.',
  },
  {
    question: 'Does it support Web Workers?',
    answer:
      'Yes. utils.asyncMode runs a synchronous function in a Web Worker (browser) or worker thread (Node.js) and returns a Promise, so training and inference can stay off the main thread.',
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

// Deterministic two-class scatter for the hero figure (seeded — stable SSR markup).
function makeScatter() {
  let seed = 7;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const g = () => (rnd() + rnd() + rnd() - 1.5) / 1.5; // ~N(0,1) in roughly [-1,1]
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  const a: { x: number; y: number }[] = [];
  const b: { x: number; y: number }[] = [];
  for (let i = 0; i < 22; i++) {
    a.push({ x: clamp(132 + g() * 52, 46, 360), y: clamp(196 + g() * 40, 28, 252) });
    b.push({ x: clamp(252 + g() * 52, 46, 360), y: clamp(98 + g() * 40, 28, 252) });
  }
  return { a, b };
}

const SCATTER = makeScatter();

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

      <main className="acad flex flex-1 flex-col pb-16">
        <div className="acad-wrap">
          {/* Masthead */}
          <div className="acad-masthead acad-rise">
            <span>
              <b>@kanaries/ml</b> — Machine Learning Library
            </span>
            <span>JavaScript · TypeScript</span>
            <span>scikit-learn-style · MIT</span>
          </div>
          <hr className="acad-rule" />

          {/* Hero */}
          <section className="acad-hero">
            <div>
              <span className="acad-eyebrow acad-rise" style={{ animationDelay: '60ms' }}>
                Open source · Browser + Node.js
              </span>
              <h1 className="acad-title acad-rise" style={{ animationDelay: '120ms' }}>
                Machine Learning in JavaScript,
                <br />
                <em>the scikit-learn way.</em>
              </h1>
              <p className="acad-lead acad-rise" style={{ animationDelay: '200ms' }}>
                A lightweight ML library for the <strong>browser</strong> and{' '}
                <strong>Node.js</strong>. Familiar <code>fit</code> / <code>predict</code> API, zero-install
                demos, production-ready builds.
              </p>

              <div className="acad-cmd acad-rise" style={{ animationDelay: '280ms' }}>
                <span className="prompt">$</span>
                <code>npm install @kanaries/ml</code>
                <CopyButton value="npm install @kanaries/ml" />
              </div>

              <div className="acad-actions acad-rise" style={{ animationDelay: '340ms' }}>
                <Link className="acad-btn acad-btn--solid" href="/docs/apis">
                  Browse API
                </Link>
                <Link className="acad-btn acad-btn--ghost" href="/docs">
                  Read the docs
                </Link>
              </div>

              <div className="acad-meta acad-rise" style={{ animationDelay: '420ms' }}>
                <span>TypeScript-native</span>
                <span>
                  <b>~32 kB</b> gzipped
                </span>
                <span>Tree-shakable ESM</span>
                <span>
                  <b>11</b> algorithm families
                </span>
              </div>
            </div>

            {/* Figure 1 */}
            <figure className="acad-figure acad-rise" style={{ animationDelay: '320ms' }}>
              <svg viewBox="0 0 400 290" role="img" aria-label="Two-class scatter plot separated by a decision boundary">
                {/* grid */}
                <g className="fig-grid">
                  {[68, 124, 180, 236, 292, 348].map((x) => (
                    <line key={`vx${x}`} x1={x} y1={24} x2={x} y2={256} />
                  ))}
                  {[56, 102, 148, 194, 240].map((y) => (
                    <line key={`hz${y}`} x1={42} y1={y} x2={372} y2={y} />
                  ))}
                </g>
                {/* axes */}
                <line className="fig-axis" x1={42} y1={24} x2={42} y2={256} />
                <line className="fig-axis" x1={42} y1={256} x2={372} y2={256} />
                {/* decision boundary */}
                <line className="fig-boundary" x1={58} y1={256} x2={360} y2={48} />
                {/* points */}
                {SCATTER.a.map((p, i) => (
                  <circle
                    key={`a${i}`}
                    className="fig-pt fig-pt--a"
                    cx={p.x}
                    cy={p.y}
                    r={4.2}
                    style={{ animationDelay: `${600 + i * 22}ms` }}
                  />
                ))}
                {SCATTER.b.map((p, i) => (
                  <rect
                    key={`b${i}`}
                    className="fig-pt fig-pt--b"
                    x={p.x - 3.6}
                    y={p.y - 3.6}
                    width={7.2}
                    height={7.2}
                    style={{ animationDelay: `${640 + i * 22}ms` }}
                  />
                ))}
                {/* axis labels */}
                <text className="fig-label" x={368} y={272}>
                  x₁
                </text>
                <text className="fig-label" x={20} y={32}>
                  x₂
                </text>
              </svg>
              <figcaption className="acad-figcaption">
                <b>Fig. 1</b> — Binary classification over a two-dimensional feature space. A linear
                decision boundary (dashed) separates two classes, fit with <code>Linear.LogisticRegression</code>.
              </figcaption>
            </figure>
          </section>

          {/* Abstract */}
          <section className="acad-abstract">
            <div className="acad-abstract-label">Abstract</div>
            <div>
              <p>
                Bringing the ergonomics of scikit-learn to JavaScript and TypeScript, @kanaries/ml pairs
                a small, predictable estimator API with tree-shakable ESM builds — so classical machine
                learning runs natively in the browser, on Node.js servers, and at the edge.
              </p>
              <p>
                The library favours practical coverage and runnable documentation over breadth for its own
                sake. Every estimator is typed, every algorithm has a guide, and the same{' '}
                <code>fit</code>/<code>predict</code> code ships from prototype to production.
              </p>
            </div>
          </section>

          {/* §01 Why JavaScript */}
          <section className="acad-section" id="why-javascript">
            <header className="acad-sec-head">
              <span className="acad-sec-num">§ 01</span>
              <span className="acad-sec-kicker">Motivation</span>
              <h2 className="acad-sec-title">Why JavaScript for machine learning?</h2>
              <hr className="acad-rule acad-rule--thin" />
            </header>
            <div className="acad-grid acad-grid--2">
              {WHY_JS.map((item, i) => (
                <article className="acad-cell" key={item.title}>
                  <span className="acad-cell-idx">{String(i + 1).padStart(2, '0')}</span>
                  <h3>{item.title}</h3>
                  <p>{item.description}</p>
                </article>
              ))}
            </div>
          </section>

          {/* §02 Quick start / listings */}
          <section className="acad-section" id="quick-start">
            <header className="acad-sec-head">
              <span className="acad-sec-num">§ 02</span>
              <span className="acad-sec-kicker">Method</span>
              <h2 className="acad-sec-title">Quick start</h2>
              <hr className="acad-rule acad-rule--thin" />
            </header>
            <div className="acad-listings">
              <div className="acad-listing">
                <div className="acad-listing-bar">
                  <span className="tag">Listing 1</span>
                  <span className="file">node.ts</span>
                </div>
                <pre>
                  <code>
                    <span className="kw">import</span> {'{ Linear, utils }'} <span className="kw">from</span>{' '}
                    <span className="st">'@kanaries/ml'</span>;{'\n\n'}
                    <span className="kw">const</span> X = [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2]];{'\n'}
                    <span className="kw">const</span> y = [0, 0, 1];{'\n\n'}
                    <span className="cm">{'// standardize features, then fit'}</span>
                    {'\n'}
                    <span className="kw">const</span> scaler = <span className="kw">new</span>{' '}
                    utils.Preprocessing.StandardScaler();{'\n'}
                    <span className="kw">const</span> clf = <span className="kw">new</span>{' '}
                    Linear.LogisticRegression({'{ maxIter: 200 }'});{'\n'}
                    clf.fit(scaler.fitTransform(X), y);{'\n\n'}
                    clf.predict([[6.1, 3.1]]); <span className="cm">{'// → [1]'}</span>
                  </code>
                </pre>
              </div>
              <div className="acad-listing">
                <div className="acad-listing-bar">
                  <span className="tag">Listing 2</span>
                  <span className="file">index.html — browser via CDN</span>
                </div>
                <pre>
                  <code>
                    <span className="cm">{'<script type="module">'}</span>
                    {'\n  '}
                    <span className="kw">import</span> {'{ Clusters }'} <span className="kw">from</span>{'\n    '}
                    <span className="st">'https://cdn.skypack.dev/@kanaries/ml'</span>;{'\n\n  '}
                    <span className="kw">const</span> km = <span className="kw">new</span> Clusters.KMeans(2);
                    {'\n  '}km.fitPredict([[1, 1], [1.2, 1.1], [5, 5]]);{'\n  '}
                    <span className="cm">{'// → [0, 0, 1]'}</span>
                    {'\n'}
                    <span className="cm">{'</script>'}</span>
                  </code>
                </pre>
              </div>
            </div>
          </section>

          {/* §03 What you can build */}
          <section className="acad-section" id="use-cases">
            <header className="acad-sec-head">
              <span className="acad-sec-num">§ 03</span>
              <span className="acad-sec-kicker">Applications</span>
              <h2 className="acad-sec-title">What you can build</h2>
              <hr className="acad-rule acad-rule--thin" />
            </header>
            <div className="acad-grid acad-grid--2">
              {USE_CASES.map((useCase, i) => (
                <article className="acad-cell" key={useCase.title}>
                  <span className="acad-cell-idx">{String.fromCharCode(97 + i)}.</span>
                  <h3>{useCase.title}</h3>
                  <p>{useCase.description}</p>
                </article>
              ))}
            </div>
          </section>

          {/* §04 Capabilities */}
          <section className="acad-section" id="features">
            <header className="acad-sec-head">
              <span className="acad-sec-num">§ 04</span>
              <span className="acad-sec-kicker">Capabilities</span>
              <h2 className="acad-sec-title">Key features</h2>
              <hr className="acad-rule acad-rule--thin" />
            </header>
            <div className="acad-grid acad-grid--3">
              {FEATURES.map((feature, i) => (
                <article className="acad-cell" key={feature.title}>
                  <span className="acad-cell-idx">{String(i + 1).padStart(2, '0')}</span>
                  <h3>{feature.title}</h3>
                  <p>{feature.description}</p>
                </article>
              ))}
            </div>
          </section>

          {/* §05 Module index */}
          <section className="acad-section" id="modules">
            <header className="acad-sec-head">
              <span className="acad-sec-num">§ 05</span>
              <span className="acad-sec-kicker">Reference</span>
              <h2 className="acad-sec-title">Index of modules</h2>
              <hr className="acad-rule acad-rule--thin" />
            </header>
            <table className="acad-table">
              <thead>
                <tr>
                  <th>Family</th>
                  <th>Estimators &amp; algorithms</th>
                  <th>Reference</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(ALGORITHMS).map(([group, items]) => (
                  <tr key={group}>
                    <td className="grp">{group}</td>
                    <td className="est">
                      {items.map((item, idx) => (
                        <span key={item.label}>
                          {idx > 0 && <span className="chip"> · </span>}
                          <Link className="acad-inline-link" href={item.href}>
                            {item.label}
                          </Link>
                        </span>
                      ))}
                    </td>
                    <td className="ref">
                      <Link className="acad-ref-link" href={items[0].href}>
                        View →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {/* §06 How it works */}
          <section className="acad-section" id="how-it-works">
            <header className="acad-sec-head">
              <span className="acad-sec-num">§ 06</span>
              <span className="acad-sec-kicker">Workflow</span>
              <h2 className="acad-sec-title">From data to deployment</h2>
              <hr className="acad-rule acad-rule--thin" />
            </header>
            <div className="acad-flow">
              {HOW_IT_WORKS.map((step, index) => (
                <article className="acad-step" key={step.title}>
                  <span className="acad-step-num">STEP {String(index + 1).padStart(2, '0')}</span>
                  <h3>{step.title}</h3>
                  <p>{step.description}</p>
                  <p>
                    <code>{step.code}</code>
                  </p>
                </article>
              ))}
            </div>
          </section>

          {/* §07 FAQ */}
          <section className="acad-section" id="faq">
            <header className="acad-sec-head">
              <span className="acad-sec-num">§ 07</span>
              <span className="acad-sec-kicker">Notes</span>
              <h2 className="acad-sec-title">Frequently asked</h2>
              <hr className="acad-rule acad-rule--thin" />
            </header>
            <div className="acad-faq">
              {FAQ_ITEMS.map((item, i) => (
                <details className="acad-q" key={item.question} open={i === 0}>
                  <summary>
                    <span className="acad-q-num">Q{i + 1}</span>
                    <span>{item.question}</span>
                    <span className="acad-q-toggle">+</span>
                  </summary>
                  <p className="acad-q-body">{item.answer}</p>
                </details>
              ))}
            </div>
          </section>

          {/* Community */}
          <section className="acad-section" id="community" style={{ paddingBottom: '1rem' }}>
            <header className="acad-sec-head">
              <span className="acad-sec-num">§ 08</span>
              <span className="acad-sec-kicker">Community</span>
              <h2 className="acad-sec-title">Join the project</h2>
              <hr className="acad-rule acad-rule--thin" />
            </header>
            <nav className="acad-community">
              {COMMUNITY_LINKS.map((link) => (
                <Link key={link.label} href={link.href} target="_blank" rel="noreferrer">
                  {link.label}
                  <span className="arr">↗</span>
                </Link>
              ))}
            </nav>
          </section>

          {/* Coda / CTA */}
          <section className="acad-section" style={{ paddingTop: '1rem' }}>
            <div className="acad-coda">
              <span className="acad-eyebrow" style={{ justifyContent: 'center' }}>
                Get started
              </span>
              <h2>Build ML in JavaScript today</h2>
              <p>Deploy machine learning models anywhere JavaScript runs with @kanaries/ml.</p>
              <div className="acad-actions">
                <Link className="acad-btn acad-btn--solid" href="/docs/apis">
                  Browse API docs
                </Link>
                <Link
                  className="acad-btn acad-btn--ghost"
                  href="https://www.npmjs.com/package/@kanaries/ml"
                  target="_blank"
                  rel="noreferrer"
                >
                  npm install @kanaries/ml
                </Link>
              </div>
            </div>

            <div className="acad-colophon">
              <span>© {`${siteUrl}`.replace(/^https?:\/\//, '')} — released under MIT</span>
              <span>
                Set in Fraunces &amp; Newsreader · Built with{' '}
                <Link href="https://github.com/Kanaries/ml">@kanaries/ml</Link>
              </span>
            </div>
          </section>
        </div>
      </main>
    </>
  );
}
