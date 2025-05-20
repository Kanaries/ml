# Clusters

## Clusters.KMeans

```ts
constructor (n_clusters: number = 2, opt_ratio: number = 0.05, initCenters?: number[][], max_iter: number = 30)
```

| props name | type | default value |
|-|-|-|
| n_clusters | number | 2 |
| opt_ratio | number | 0.05 |
| initCenters | number[][] | undefined |
| max_iter | number | 30 |


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

## Clusters.DBScan

```ts
constructor(eps: number = 0.5, minSamples: number = 5, distanceType: Distance.IDistanceType = 'euclidiean')
```

`fitPredict(samplesX: number[][]): number[]` returns cluster labels for samples. Noise points are marked as `-1`.

```ts
const X = makeCircles(20, 20, 1, 5);
const dbscan = new DBScan(0.6, 3);
const labels = dbscan.fitPredict(X);
```

## Clusters.HDBScan

```ts
constructor(
    min_cluster_size: number = 5,
    min_samples: number | null = null,
    cluster_selection_epsilon: number = 0.5,
    metric: Distance.IDistanceType = 'euclidiean'
)
```

`fitPredict(samplesX: number[][]): number[]` returns cluster labels. Noise points are marked as `-1`.

This is a simplified implementation that internally calls DBSCAN using `cluster_selection_epsilon` as the `eps` parameter.

```ts
const hdb = new HDBScan(5, null, 0.6);
const labels = hdb.fitPredict(X);
```

## Clusters.MeanShift

```ts
constructor(
    bandwidth: number = 1,
    max_iter: number = 300,
    distanceType: Distance.IDistanceType = 'euclidiean'
)
```

Methods:
- `fitPredict(samplesX: number[][]): number[]`
- `getCentroids(): number[][]`

```ts
const ms = new MeanShift(2);
const labels = ms.fitPredict(X);
const centers = ms.getCentroids();
```

## Clusters.OPTICS

```ts
interface OPTICSOptions {
    min_samples?: number;
    max_eps?: number;
    metric?: Distance.IDistanceType;
    p?: number;
    eps?: number;
}
constructor(options: OPTICSOptions = {})
```

`fitPredict(samplesX: number[][]): number[]` returns cluster labels. Noise points are marked as `-1`.

```ts
const optics = new OPTICS({ eps: 0.5, min_samples: 5 });
const labels = optics.fitPredict(X);
```

## Clusters.kmeansPlusPlus

```ts
kmeansPlusPlus(
    X: number[][],
    n_clusters: number,
    sampleWeight?: number[],
    randomState: () => number = Math.random
): { centers: number[][]; indices: number[] }
```

This utility initializes cluster centers using the k-means++ strategy.

```ts
const { centers } = kmeansPlusPlus(X, 3);
```
