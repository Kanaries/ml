## LinearRegression

```ts
constructor()
```

### 方法
+ `fit(X: number[][], Y: number[]): void` 根据训练数据计算回归系数。
+ `predict(X: number[][]): number[]` 对给定样本进行预测。

### 例子
```ts
const lr = new LinearRegression();
const X = [[0], [1], [2]];
const Y = [3, 5, 7];
lr.fit(X, Y);
const pred = lr.predict([[3]]); // 预期约为 9
```
