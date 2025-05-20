# Clusters.DBScan

DBScan（Density-Based Spatial Clustering of Applications with Noise）通过核心点与密度对样本进行聚类，可自动识别噪声点。

```ts
constructor(eps: number = 0.5, minSamples: number = 5, distanceType: Distance.IDistanceType = 'euclidiean')
```

## 参数
+ `eps` `number` 邻域距离阈值，距离小于等于该值的样本视为邻居。
+ `minSamples` `number` 形成核心点所需的最少邻居数量。
+ `distanceType` `Distance.IDistanceType` 距离度量方式，默认使用 `euclidiean`。

## 方法
+ `fitPredict(samplesX: number[][]): number[]` 对样本执行聚类，返回每个样本的类别编号，噪声点返回 `-1`。

### 例子
```ts
const X = makeCircles(20, 20, 1, 5);
const dbscan = new DBScan(0.6, 3);
const labels = dbscan.fitPredict(X);
```
