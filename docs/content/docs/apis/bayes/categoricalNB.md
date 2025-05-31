---
title: CategoricalNB
description: API reference for CategoricalNB
---

# Bayes.CategoricalNB

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

### Example
```ts
const clf = new CategoricalNB();
clf.fit(trainX, trainY);
const result = clf.predict(testX);
```
