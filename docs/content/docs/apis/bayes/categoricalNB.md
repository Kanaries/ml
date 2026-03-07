---
title: CategoricalNB
description: API and practical guide for CategoricalNB in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Bayes.CategoricalNB

Naive Bayes classifier for features with a finite number of discrete
categories. It counts how often each feature value appears in each class and
uses additive smoothing to estimate the conditional probabilities.

```ts
interface CategoricalNBProps {
    alpha?: number;
    forceAlpha?: boolean;
    fitPrior?: boolean;
    classPrior?: number[] | null;
    minCategories?: number | number[] | null;
}
constructor(props: CategoricalNBProps = {})
```

### Parameters
- `alpha` &mdash; Smoothing parameter used when computing category
  probabilities. Larger values make the model less sensitive to missing
  observations.
- `forceAlpha` &mdash; If `true`, ensures that `alpha` is strictly positive even
  when a small value is provided.
- `fitPrior` &mdash; Whether to learn class prior probabilities from data. When
  `false`, class priors are assumed to be uniform.
- `classPrior` &mdash; Optional array of prior probabilities for each class.
  Overrides the data-derived priors when provided.
- `minCategories` &mdash; Minimum number of categories assumed for each feature.
  Can be a single number or an array specifying the value per feature.

### Example
```ts
const clf = new CategoricalNB();
clf.fit(trainX, trainY);
const result = clf.predict(testX);
```

## Practical guide: CategoricalNB in JavaScript and TypeScript

CategoricalNB is designed for discrete categorical features encoded as integer category IDs.

### When to use CategoricalNB
- Inputs are naturally categorical and not meaningful on a numeric distance scale.
- You need fast classification with probabilistic interpretation.
- You want a robust baseline for tabular categoricals in pure JavaScript.

### Implementation workflow
1. Encode every categorical column to stable integer category indices.
2. Fit on labeled rows and validate per-class calibration quality.
3. Monitor unseen-category behavior and keep category mapping versioned.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `CategoricalNB JavaScript`
- `CategoricalNB TypeScript`
- `CategoricalNB browser machine learning`
- `@kanaries/ml CategoricalNB`

