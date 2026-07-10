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

test('predict before fit throws', () => {
    const iForest = new IsolationForest();
    expect(() => iForest.predict([[1], [2]])).toThrow();
});

test('same random_state gives reproducible results', () => {
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
    const a = new IsolationForest(4, 25, 'auto', 7);
    const b = new IsolationForest(4, 25, 'auto', 7);
    a.fit(X);
    b.fit(X);
    expect(X.map(x => a.anomalyScore(x))).toEqual(X.map(x => b.anomalyScore(x)));
    expect(a.predict(X)).toEqual(b.predict(X));
});

test('refit resets the forest: fit(A) then fit(B) equals a fresh fit(B)', () => {
    const A = [[0], [1], [2], [3], [4], [5], [6], [7]];
    const B = [[10], [11], [12], [13], [14], [15], [16], [50]];
    const testX = [[11], [13], [49]];
    const refitted = new IsolationForest(8, 20, 'auto', 42);
    refitted.fit(A);
    refitted.fit(B);
    const fresh = new IsolationForest(8, 20, 'auto', 42);
    fresh.fit(B);
    expect(testX.map(x => refitted.anomalyScore(x))).toEqual(testX.map(x => fresh.anomalyScore(x)));
    expect(refitted.predict(testX)).toEqual(fresh.predict(testX));
});

test('subsampling_size is not overwritten by fitting a small dataset', () => {
    const small = [[0], [1], [2]];
    const large: number[][] = [];
    for (let i = 0; i < 32; i++) {
        large.push([i]);
    }
    const testX = [[3], [15], [100]];
    const reused = new IsolationForest(16, 20, 'auto', 11);
    reused.fit(small);
    reused.fit(large);
    const fresh = new IsolationForest(16, 20, 'auto', 11);
    fresh.fit(large);
    expect(testX.map(x => reused.anomalyScore(x))).toEqual(testX.map(x => fresh.anomalyScore(x)));
});