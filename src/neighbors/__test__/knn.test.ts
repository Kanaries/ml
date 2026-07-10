import { KNearestNeighbors, KNearstNeighbors } from '../knn';

test('init', () => {
    const knn = new KNearestNeighbors();
    expect(knn).toBeDefined();
})

test('deprecated alias init', () => {
    const knn = new KNearstNeighbors();
    expect(knn).toBeDefined();
    expect(knn).toBeInstanceOf(KNearestNeighbors);
})

test('predict before fit throws', () => {
    const knn = new KNearestNeighbors();
    expect(() => knn.predict([[0]])).toThrow('model is not fitted');
});

test('fit validates inputs', () => {
    const knn = new KNearestNeighbors(1);
    expect(() => knn.fit([], [])).toThrow('X and Y must be non-empty');
    expect(() => knn.fit([[0], [1]], [0])).toThrow('X and Y must have the same length');
    expect(() => knn.fit([[0, 1], [2]], [0, 1])).toThrow('all rows in X must have the same length');
});

test('predict validates feature dimensions', () => {
    const knn = new KNearestNeighbors(1);
    knn.fit([[0, 0], [1, 1]], [0, 1]);
    expect(() => knn.predict([[0]])).toThrow('input feature size does not match fitted model');
});

test('kNeighbors larger than the training set falls back to all samples', () => {
    // Clamping mirrors KNeighborsRegressor. Note sklearn raises here instead;
    // aligning would require updating GridSearchCV usages outside src/neighbors
    // (see src/utils/__test__/modelSelection.test.ts).
    const knn = new KNearestNeighbors(5);
    knn.fit([[0], [1], [2]], [0, 1, 0]);
    expect(knn.predict([[0]])).toEqual([0]);
});

test('distance weighting uses only zero-distance neighbors on exact match', () => {
    // sklearn semantics: an exact match gives all weight to zero-distance
    // neighbors; ties between them resolve to the smallest label.
    const knn = new KNearestNeighbors(3, 'distance');
    knn.fit([[0], [0], [1]], [8, 7, 9]);
    expect(knn.predict([[0]])).toEqual([7]);
});

test('uniform vote ties resolve to the smallest label', () => {
    const knn = new KNearestNeighbors(2, 'uniform');
    knn.fit([[1], [-1]], [5, 1]);
    expect(knn.predict([[0.1]])).toEqual([1]);
});

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
    const knn = new KNearestNeighbors(3, 'distance', '2-norm');
    knn.fit(trainX, trainY);
    expect(knn.predict(trainX)).toEqual(trainY);
})
