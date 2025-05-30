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
    const originalRandom = Math.random;
    let seed = 0;
    Math.random = () => {
        const x = Math.sin(seed++) * 10000;
        return x - Math.floor(x);
    };
    iForest.fit(X);
    const result = iForest.predict(X);
    Math.random = originalRandom;
    expect(result).toEqual([0, 0, 0, 0, 0, 0, 1, 1]);
})