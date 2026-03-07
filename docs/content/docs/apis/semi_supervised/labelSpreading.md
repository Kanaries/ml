---
title: LabelSpreading
description: API and practical guide for LabelSpreading in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# SemiSupervised.LabelSpreading

```ts
interface LabelSpreadingOptions {
    kernel?: 'rbf' | 'knn' | ((X: number[][], Y: number[][]) => number[][]);
    gamma?: number;
    nNeighbors?: number;
    alpha?: number;
    maxIter?: number;
    tol?: number;
}
constructor(options: LabelSpreadingOptions = {})
```

Label spreading assigns labels to unlabeled data using a normalized graph
and soft clamping controlled by `alpha`.

### Methods
* `fit(trainX: number[][], trainY: number[]): void`
* `predict(testX: number[][]): number[]`
* `predictProba(testX: number[][]): number[][]`

#### Example
```ts
const ls = new LabelSpreading();
ls.fit(trainX, trainY);
const preds = ls.predict(testX);
```

## Practical guide: LabelSpreading in JavaScript and TypeScript

LabelSpreading performs smoother semi-supervised label diffusion with regularization to reduce over-confident propagation.

### When to use LabelSpreading
- You want semi-supervised learning with stronger stability than pure propagation.
- Graph neighborhoods are useful but somewhat noisy.
- You need better robustness for low-label datasets.

### Implementation workflow
1. Create labeled/unlabeled split and construct feature graph inputs.
2. Fit LabelSpreading and monitor convergence behavior.
3. Tune kernel and regularization parameters with validation labels.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `LabelSpreading JavaScript`
- `LabelSpreading TypeScript`
- `LabelSpreading browser machine learning`
- `@kanaries/ml LabelSpreading`

