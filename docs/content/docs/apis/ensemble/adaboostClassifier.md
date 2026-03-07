---
title: AdaBoostClassifier
description: API and practical guide for AdaBoostClassifier in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Ensemble.AdaBoostClassifier

```ts
interface AdaBoostClassifierProps {
    nEstimators?: number;
    learningRate?: number;
    randomState?: number;
}
constructor(props: AdaBoostClassifierProps = {})
```

### Parameters
| name | type | default | description |
|-|-|-|-|
| nEstimators | number | 50 | Number of boosting iterations |
| learningRate | number | 1.0 | Weight applied to each stump |
| randomState | number | undefined | Seed for reproducibility |

### Algorithm
AdaBoostClassifier trains decision stumps sequentially and reweights samples so that misclassified points receive more focus in subsequent rounds.

### Methods

+ `fit(trainX: number[][], trainY: number[]): void`
+ `predict(testX: number[][]): number[]`
+ `predictProba(testX: number[][]): number[][]`
+ `getFeatureImportances(): number[]`

### Example
```ts
const clf = new AdaBoostClassifier();
clf.fit(trainX, trainY);
const result = clf.predict(testX);
```

## Practical guide: AdaBoostClassifier in JavaScript and TypeScript

AdaBoostClassifier focuses on hard examples across rounds to improve classification performance over weak learners.

### When to use AdaBoostClassifier
- Baseline classifiers miss difficult boundary regions.
- You need improved recall/precision without switching to heavy models.
- You can invest in hyperparameter tuning for learning rate and rounds.

### Implementation workflow
1. Start with balanced preprocessing and clear label quality checks.
2. Fit with several estimator counts and compare classification metrics.
3. Tune threshold and class-weight strategy for product-specific tradeoffs.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `AdaBoostClassifier JavaScript`
- `AdaBoostClassifier TypeScript`
- `AdaBoostClassifier browser machine learning`
- `@kanaries/ml AdaBoostClassifier`

