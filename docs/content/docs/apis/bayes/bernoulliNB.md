---
title: BernoulliNB
description: API reference for BernoulliNB
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
