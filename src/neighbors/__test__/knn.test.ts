import { KNearstNeighbors } from '../knn';

test('init', () => {
    const knn = new KNearstNeighbors();
    expect(knn).toBeDefined();
})

test('basic', () => {
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
    expect(knn.predict(trainX)).toEqual(trainY);
})