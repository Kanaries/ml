---
title: LabelPropagation
description: API and practical guide for LabelPropagation in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# SemiSupervised.LabelPropagation

```ts
interface LabelPropagationOptions {
    kernel?: 'rbf' | 'knn' | ((X: number[][], Y: number[][]) => number[][]);
    gamma?: number;
    nNeighbors?: number;
    maxIter?: number;
    tol?: number;
}
constructor(options: LabelPropagationOptions = {})
```

Label propagation assigns labels to unlabeled data by propagating
information from labeled points across a graph defined by a kernel.

### Methods
* `fit(trainX: number[][], trainY: number[]): void`
* `predict(testX: number[][]): number[]`
* `predictProba(testX: number[][]): number[][]`

#### Example
```ts
const lp = new LabelPropagation();
lp.fit(trainX, trainY);
const preds = lp.predict(testX);
```

## Practical guide: LabelPropagation in JavaScript and TypeScript

LabelPropagation spreads labels through similarity graphs to leverage unlabeled data in transductive settings.

### When to use LabelPropagation
- Only a small subset of samples is labeled.
- Similarity graph structure reflects class continuity.
- You need to bootstrap labels before training a final supervised model.

### Implementation workflow
1. Build feature representations and seed reliable initial labels.
2. Fit LabelPropagation and inspect propagated label confidence.
3. Validate on known labels and tune graph-related hyperparameters.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `LabelPropagation JavaScript`
- `LabelPropagation TypeScript`
- `LabelPropagation browser machine learning`
- `@kanaries/ml LabelPropagation`

