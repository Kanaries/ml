## IsolationForest (孤立森林)

`Ensemble.IsolationForest` 孤立森林是一种异常检测的算法。

```ts
constructor (subsampling_size: number = 256, tree_num: number = 100, contamination: 'auto' | number = 'auto')
```

### 参数
+ subsampling_size `number`, 每一个孤立树使用的样本数量的大小，默认为256.
+ tree_num `number`, 森林里含有的的数的数量，默认为100
+ contamination `'auto' | number`，对评分的调整，如果为auto则会默认以0.5为阈值(论文的建议)，大于则认为是异常，否则是正常值。否则的话指定一个异常的比例。

### 方法
+ public fit (samplesX: number[][]): void
+ public predict (samples: number[][]): (0|1)[]


### 例子
```ts
const iForest = new IsolationForest(256,10,0.25);
const X = [
    [-2, -1],
    [-1, -1],
    [-1, -2],
    [1, 1],
    [1, 2],
    [2, 1],
    [6, 3],
    [-4, 7],
];
iForest.fit(X);
const result = iForest.predict(X)
// expect(result).toEqual([0, 0, 0, 0, 0, 0, 1, 1]);
```