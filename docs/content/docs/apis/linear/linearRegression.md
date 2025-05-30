---
title: LinearRegression
---

# Linear.LinearRegression

```ts
constructor()
```

### Methods
+ `fit(X: number[][], Y: number[]): void`
+ `predict(X: number[][]): number[]`

### Example
```ts
const lr = new LinearRegression();
lr.fit([[0], [1]], [1, 3]);
const pred = lr.predict([[2]]); // about 5
```
