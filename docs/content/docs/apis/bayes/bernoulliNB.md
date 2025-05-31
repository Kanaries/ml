---
title: BernoulliNB
description: API reference for BernoulliNB
---

# Bayes.BernoulliNB

```ts
interface BernoulliNBProps {
    alpha?: number;
    binarize?: number | null;
    fitPrior?: boolean;
    classPrior?: number[] | null;
}
constructor(props: BernoulliNBProps = {})
```

### Example
```ts
const clf = new BernoulliNB();
clf.fit(trainX, trainY);
const result = clf.predict(testX);
```
