# 起步

## 安装
![](https://img.shields.io/github/license/kanaries/ml?)
![](https://img.shields.io/npm/v/@kanaries/ml?)
```bash
npm i --save @kanaries/ml
```

## 使用

```typescript
import { Ensemble } from '@kanaries';

const iForest = new Ensemble.IsolationForest();

const X = [[-1.1], [0.3], [0.5], [100]];
iForest.fit(X);
const result = iForest.predict([[0.1], [0], [90]]);
// [0, 0, 1]
```