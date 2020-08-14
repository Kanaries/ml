import { IsolationForest } from '../isolationForest';

test('init', () => {
    const iForest = new IsolationForest();
    expect(iForest).toBeDefined();
})

test('example', () => {
    const iForest = new IsolationForest();
    const X = [[-1.1], [0.3], [0.5], [100]];
    iForest.fit(X);
    const result = iForest.predict([[0.1], [0], [90]]);
    expect(result).toEqual([0, 0, 1]);
});

test('toy data', () => {
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
    expect(result).toEqual([0, 0, 0, 0, 0, 0, 1, 1]);
})