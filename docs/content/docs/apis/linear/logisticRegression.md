---
title: LogisticRegression
description: API reference for LogisticRegression
---

# Linear.LogisticRegression

```ts
interface LogisticRegressionProps {
    learningRate?: number;
    maxIter?: number;
}
constructor(props: LogisticRegressionProps = {})
```

### Example
```ts
const clf = new LogisticRegression();
clf.fit(trainX, trainY);
const result = clf.predict(testX);
```
