---
title: CategoricalNB
description: API reference for CategoricalNB
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
