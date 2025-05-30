---
title: DecisionTreeClassifier
description: API reference for DecisionTreeClassifier
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
