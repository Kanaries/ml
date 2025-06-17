---
title: ExtraTreeClassifier
description: API reference for ExtraTreeClassifier
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
