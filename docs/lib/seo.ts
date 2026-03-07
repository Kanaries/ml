type SeoInput = {
  title: string;
  description?: string;
  slug: string[];
};

export type FaqItem = {
  question: string;
  answer: string;
};

export type SeoProfile = {
  topic: string;
  moduleName: string;
  primaryKeyword: string;
  secondaryKeywords: string[];
  longTailKeywords: string[];
  keywords: string[];
  faq: FaqItem[];
};

const moduleLabels: Record<string, string> = {
  bayes: 'Bayes',
  clusters: 'Clusters',
  decomposition: 'Decomposition',
  ensemble: 'Ensemble',
  linear: 'Linear',
  manifold: 'Manifold',
  neighbors: 'Neighbors',
  neural_network: 'NeuralNetwork',
  semi_supervised: 'SemiSupervised',
  svm: 'SVM',
  tree: 'Tree',
  utils: 'Utils',
};

const moduleKeywordPools: Record<string, string[]> = {
  bayes: ['naive bayes javascript', 'categorical naive bayes typescript', 'probabilistic classification nodejs'],
  clusters: ['javascript clustering library', 'kmeans javascript tutorial', 'density clustering typescript'],
  decomposition: ['pca javascript', 'truncated svd typescript', 'dimensionality reduction browser ml'],
  ensemble: ['isolation forest javascript', 'adaboost classifier typescript', 'ensemble learning nodejs'],
  linear: ['linear regression javascript', 'logistic regression typescript', 'supervised machine learning browser'],
  manifold: ['tsne javascript', 'manifold learning typescript', 'spectral embedding browser ml'],
  neighbors: ['knn javascript', 'kdtree typescript', 'nearest neighbor search nodejs'],
  neural_network: ['bernoulli rbm javascript', 'feature learning typescript', 'lightweight neural models browser'],
  semi_supervised: ['semi supervised learning javascript', 'label propagation typescript', 'low label learning nodejs'],
  svm: ['svm javascript', 'linearsvc typescript', 'support vector machine nodejs'],
  tree: ['decision tree javascript', 'tree classifier typescript', 'interpretable machine learning nodejs'],
  utils: ['ml preprocessing javascript', 'async machine learning browser', 'model selection typescript'],
  docs: ['javascript machine learning docs', 'typescript machine learning api', 'scikit-learn style api javascript'],
};

function unique(values: string[]): string[] {
  const seen = new Set<string>();
  return values.filter((value) => {
    if (!value || seen.has(value)) return false;
    seen.add(value);
    return true;
  });
}

function titleToTopic(title: string): string {
  return title.replace(/\s+[-–]\s+@kanaries\/ml.*$/i, '').replace(/\s{2,}/g, ' ').trim();
}

function slugToWords(value: string): string {
  return value
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

function fallbackTopic(slug: string[]): string {
  if (!slug.length) return 'JavaScript machine learning';
  return slugToWords(slug[slug.length - 1]);
}

function detectModule(slug: string[]): string {
  if (slug[0] === 'apis' && slug.length >= 2) return slug[1];
  return 'docs';
}

function buildFaq(topic: string, moduleName: string): FaqItem[] {
  return [
    {
      question: `What problem does ${topic} solve in JavaScript machine learning projects?`,
      answer: `${topic} helps teams implement production-ready ML workflows in browser and Node.js environments with a familiar scikit-learn-style API.`,
    },
    {
      question: `When should I choose ${topic} instead of other ${moduleName} algorithms?`,
      answer: `Use ${topic} when it best matches your data shape, labeling strategy, and runtime constraints. Benchmark against at least one alternative in the same module before finalizing defaults.`,
    },
    {
      question: `Can I run ${topic} in both browser and Node.js with @kanaries/ml?`,
      answer: `Yes. @kanaries/ml is designed for JavaScript and TypeScript runtimes across browser applications, server-side Node.js services, and edge-friendly workflows.`,
    },
  ];
}

export function buildSeoProfile({ title, description, slug }: SeoInput): SeoProfile {
  const rawTopic = titleToTopic(title) || fallbackTopic(slug);
  const topic = rawTopic.length > 72 ? fallbackTopic(slug) : rawTopic;
  const moduleKey = detectModule(slug);
  const moduleName = moduleLabels[moduleKey] ?? 'Documentation';
  const pool = moduleKeywordPools[moduleKey] ?? moduleKeywordPools.docs;

  const primaryKeyword = `${topic} JavaScript`;
  const secondaryKeywords = [
    `${topic} TypeScript`,
    `${topic} tutorial`,
    `${moduleName} algorithms in JavaScript`,
  ];
  const longTailKeywords = [
    ...pool,
    `@kanaries/ml ${topic}`,
    description ? `${topic} in browser and Node.js` : '',
  ];
  const keywords = unique([
    primaryKeyword,
    ...secondaryKeywords,
    ...longTailKeywords,
    '@kanaries/ml',
    'browser machine learning',
    'Node.js machine learning',
    'scikit-learn style API',
  ]);

  return {
    topic,
    moduleName,
    primaryKeyword,
    secondaryKeywords,
    longTailKeywords: unique(longTailKeywords),
    keywords,
    faq: buildFaq(topic, moduleName),
  };
}
