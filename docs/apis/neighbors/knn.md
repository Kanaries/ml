## KNearstNeighbors

```ts
constructor(
    kNeighbors: number = 5,
    weightType: IWeightType = 'uniform',
    distanceType: Distance.IDistanceType = 'euclidiean',
    pNorm: number = 2
)
```

### Example
```ts
const knn = new KNearstNeighbors(3, 'distance', '2-norm');
knn.fit(trainX, trainY);
const result = knn.predict(trainX);
```
