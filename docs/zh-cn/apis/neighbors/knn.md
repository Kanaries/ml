## KNearstNeighbors

```ts
constructor(
    kNeighbors: number = 5,
    weightType: IWeightType = 'uniform',
    distanceType: Distance.IDistanceType = 'euclidiean',
    pNorm: number = 2
)
```



### 例子
```ts
const trainX: number[][] = [];
const trainY: number[] = [];
for (let i = 0; i < 100; i++) {
    trainX.push([])
    for (let j = 0; j < 10; j++) {
        trainX[i].push(Math.random())
    }
    trainY.push(i % 8);
}
const knn = new KNearstNeighbors(3, 'distance', '2-norm');
knn.fit(trainX, trainY);
const result = knn.predict(trainX)
// expect(result).toEqual(trainY);
```