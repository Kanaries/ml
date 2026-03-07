---
title: BernoulliNB
description: API and practical guide for BernoulliNB in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Bayes.BernoulliNB

Bernoulli Naive Bayes classifier for binary or boolean features. Continuous
features can be converted to binary values using the `binarize` threshold.
Class and feature probabilities are estimated with additive smoothing.

```ts
interface BernoulliNBProps {
    alpha?: number;
    binarize?: number | null;
    fitPrior?: boolean;
    classPrior?: number[] | null;
}
constructor(props: BernoulliNBProps = {})
```

### Parameters
- `alpha` &mdash; Additive smoothing parameter applied when estimating
  probabilities. Defaults to `1.0`.
- `binarize` &mdash; Threshold for binarizing input features. If `null`, the
  input is assumed to already be binary.
- `fitPrior` &mdash; Whether to learn class prior probabilities from the
  training data. When `false`, a uniform prior is used.
- `classPrior` &mdash; Optional array of prior probabilities for each class.
  If provided, these values override the learned priors.

### Example
```ts
const clf = new BernoulliNB();
clf.fit(trainX, trainY);
const result = clf.predict(testX);
```

## Practical guide: BernoulliNB in JavaScript and TypeScript

BernoulliNB models binary feature presence and is effective for fast text or event-flag classification baselines.

### When to use BernoulliNB
- Your features represent yes/no states such as token presence, clicks, or product flags.
- You need probabilistic outputs for ranking, threshold tuning, or alert prioritization.
- You want a lightweight baseline before trying heavier linear or tree models.

### Implementation workflow
1. Convert inputs to binary indicators or thresholded boolean-like features.
2. Train with `fit(X, y)` and inspect class probabilities with prediction outputs.
3. Tune priors and evaluate precision/recall for your target class tradeoff.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `BernoulliNB JavaScript`
- `BernoulliNB TypeScript`
- `BernoulliNB browser machine learning`
- `@kanaries/ml BernoulliNB`

