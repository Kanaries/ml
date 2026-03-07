---
title: ExtraTreeClassifier
description: API and practical guide for ExtraTreeClassifier in @kanaries/ml, including when to use it in JavaScript and TypeScript ML workflows.
---

# Tree.ExtraTreeClassifier

```ts
interface ExtraTreeProps {
    max_depth?: number;
    min_samples_split?: number;
    criterion?: 'entropy' | 'gini';
    max_features?: number;
}

constructor(props: ExtraTreeProps = {})
```

### Example
```ts
const clf = new ExtraTreeClassifier();
clf.fit(X, Y);
const result = clf.predict(T);
```

## Practical guide: ExtraTreeClassifier in JavaScript and TypeScript

ExtraTreeClassifier injects additional split randomness to reduce variance and improve generalization in noisy settings.

### When to use ExtraTreeClassifier
- Standard decision trees overfit your training set.
- You need fast tree-based classification with stronger randomization.
- You plan to combine trees in ensemble-style workflows.

### Implementation workflow
1. Train with randomized split behavior on cleaned tabular inputs.
2. Compare validation stability against DecisionTreeClassifier.
3. Tune depth and split constraints based on overfit indicators.

### JavaScript deployment notes
- Prefer feature scaling for distance-based and gradient-based algorithms to improve stability.
- In browser apps, run heavy training in Web Workers to keep UI interactions smooth.
- Keep a simple baseline from the same module as a fallback model for comparison.

### Search intents this page targets
- `ExtraTreeClassifier JavaScript`
- `ExtraTreeClassifier TypeScript`
- `ExtraTreeClassifier browser machine learning`
- `@kanaries/ml ExtraTreeClassifier`

