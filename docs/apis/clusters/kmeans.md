# Clusters.KMeans

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