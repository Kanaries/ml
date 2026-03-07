---
title: ExtraTreeRegressor
description: API and practical guide for ExtraTreeRegressor in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Tree.ExtraTreeRegressor

```ts
interface ExtraTreeRegressorProps {
    max_depth?: number;
    min_samples_split?: number;
    splitter?: 'random';
    max_features?: number | 'sqrt' | 'log2';
}

constructor(props: ExtraTreeRegressorProps = {})
```

### Example
```ts
const extra = new ExtraTreeRegressor();
extra.fit(X, Y);
const preds = extra.predict(T);
```

## Practical guide: ExtraTreeRegressor in JavaScript and TypeScript

ExtraTreeRegressor uses randomized splits for regression to reduce variance and capture non-linear structure efficiently.

### When to use ExtraTreeRegressor
- DecisionTreeRegressor is too sensitive to small data perturbations.
- You need robust tree-style regression with limited tuning overhead.
- Non-linear relationships dominate your numeric prediction task.

### Implementation workflow
1. Fit with baseline constraints and inspect holdout error metrics.
2. Benchmark against linear and decision tree baselines.
3. Tune depth/min-sample controls for stable generalization.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `ExtraTreeRegressor JavaScript`
- `ExtraTreeRegressor TypeScript`
- `ExtraTreeRegressor browser machine learning`
- `@kanaries/ml ExtraTreeRegressor`

