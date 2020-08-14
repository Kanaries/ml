# Clusters.KMeans

KMeans聚类算法，KMeans是一个根据距离和中心点进行聚类的算法。

```ts
constructor (n_clusters: number = 2, opt_ratio: number = 0.05, initCenters?: number[][], max_iter: number = 30)
```

## 参数
+ n_clusters: `number`, default = 2. 群簇的数量/或者中心的数量。
+ opt_ratio: `number`, default = 0.05。 优化率阈值，当一次迭代后，损失函数的下降小于该比率时不进行下次迭代。
+ initCenters `number[][]`。初始化中心，如果不提供，则使用完全随机生成的方式。
+ max_iter `number`, default = 30。最大迭代数。

## 方法
+ `fitPredict (sampleX: number[][], sampleWeights?: number[])` 进行聚类计算，并给出样本本身的类别。

### fitPredict

`fitPredict`的作用是接收样本数据，并对样本数据本身进行聚类。

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

## 原理
Kmeans的核心思想是通过距离将相近的样本划分为一组。在这个过程中主要包含两个过程：
1. 找到每个样本最近的“中心点”，把该样本的类别标记为中心点的类别。
2. 重新根据所有样本新分配的类别，计算每个类别的中心点的位置。

KMeans算法即首先定义一组中心点(一般是随机化)，然后重复1，2。每次迭代计算一个目标函数，即所有样本到其中心点的距离之和，直到这个目标函数在迭代中几乎不再变化为止。

$ \underset{\mathbf{S}} {\operatorname{arg\,min}} \sum_{i=1}^{k} \sum_{\mathbf x \in S_i} \left\| \mathbf x - \boldsymbol\mu_i \right\|^2 = \underset{\mathbf{S}} {\operatorname{arg\,min}}  \sum_{i=1}^k |S_i| \operatorname{Var} S_i $

下图所示的就是算法在冉迭代次数对数据的聚类情况

![](https://ch-resources.oss-cn-shanghai.aliyuncs.com/kanaries-ml/kmeans-example.svg)

Kmeans由于只是简单了依赖的距离来判断分组，在有些情况下会表现很差，如下图所示：

![](https://ch-resources.oss-cn-shanghai.aliyuncs.com/kanaries-ml/kmeans-bad-example.svg)