---
title: DecisionTreeClassifier
description: API and practical guide for DecisionTreeClassifier in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Tree.DecisionTreeClassifier

```ts
interface DecisionTreeProps {
    max_depth?: number;
    min_samples_split?: number;
    criterion?: 'entropy' | 'gini';
}

constructor(props: DecisionTreeProps = {})
```

### Example
```ts
const dt = new DecisionTreeClassifier({ criterion: 'gini' });
dt.fit(X, Y);
const result = dt.predict(T);
```

## Practical guide: DecisionTreeClassifier in JavaScript and TypeScript

DecisionTreeClassifier learns human-readable if/else rules for classification tasks on tabular data.

### When to use DecisionTreeClassifier
- Interpretability and decision-path transparency are important.
- Feature interactions are non-linear and heterogeneous.
- You need a baseline that is easy to inspect and debug.

### Implementation workflow
1. Prepare cleaned tabular features and split train/validation data.
2. Fit the classifier and inspect depth, splits, and leaf purity.
3. Tune depth/min-sample settings to reduce overfitting.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `DecisionTreeClassifier JavaScript`
- `DecisionTreeClassifier TypeScript`
- `DecisionTreeClassifier browser machine learning`
- `@kanaries/ml DecisionTreeClassifier`

