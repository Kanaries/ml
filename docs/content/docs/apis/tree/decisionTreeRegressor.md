---
title: DecisionTreeRegressor
description: API and practical guide for DecisionTreeRegressor in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Tree.DecisionTreeRegressor

```ts
interface RegressionTreeProps {
    max_depth?: number;
    min_samples_split?: number;
}

constructor(props: RegressionTreeProps = {})
```

### Example
```ts
const reg = new DecisionTreeRegressor({ max_depth: 5 });
reg.fit(X, Y);
const preds = reg.predict(T);
```

## Practical guide: DecisionTreeRegressor in JavaScript and TypeScript

DecisionTreeRegressor models non-linear numeric targets with interpretable split rules.

### When to use DecisionTreeRegressor
- Linear regression misses stepwise or interaction-driven behavior.
- You need explainable regression predictions for stakeholders.
- Fast inference with simple control over model complexity is required.

### Implementation workflow
1. Train with baseline depth constraints and evaluate residuals.
2. Inspect important splits for domain plausibility.
3. Tune split constraints to balance bias and variance.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `DecisionTreeRegressor JavaScript`
- `DecisionTreeRegressor TypeScript`
- `DecisionTreeRegressor browser machine learning`
- `@kanaries/ml DecisionTreeRegressor`

