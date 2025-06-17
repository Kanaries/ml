---
title: ExtraTreeRegressor
description: API reference for ExtraTreeRegressor
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
