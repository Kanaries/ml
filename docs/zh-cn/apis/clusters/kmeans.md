# Clusters.KMeans

KMeans聚类算法

```ts
constructor (n_clusters: number = 2, opt_ratio: number = 0.05, initCenters?: number[][], max_iter: number = 30)
```

## 参数
+ n_clusters: `number`, default = 2. 群簇的数量/或者中心的数量。
+ opt_ratio: `number`, default = 0.05。 优化率阈值，当一次迭代后，损失函数的下降小于该比率时不进行下次迭代。
+ initCenters `number[][]`。初始化中心，如果不提供，则使用完全随机生成的方式。
+ max_iter `number`, default = 30。最大迭代数。

## 方法
+ fitPredict (sampleX: number[][], sampleWeights?: number[]) 进行聚类计算，并给出样本本身的类别。

### fitPredict

#### 参数
+ sampleX `number[][]`。输入样本。
+ sampleWeights `number[]` 样本的权重。如果不传，则默认所有样本的权重相同。

#### 返回值
`number[]`。一个数组，每一个值是样本对应的类别的序号。

## 用例
```js
const X = [
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [1, 1],
];
const sampleWeights = [3, 1, 1, 3];
const initCenters = [[0, 0], [1, 1]];

const kmeans = new KMeans(2, 0.05, initCenters);
const result = kmeans.fitPredict(X, sampleWeights);
```