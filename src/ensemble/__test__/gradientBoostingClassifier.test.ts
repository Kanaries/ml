import { GradientBoostingClassifier } from '../gradientBoostingClassifier';

test('GradientBoostingClassifier init', () => {
    expect(new GradientBoostingClassifier()).toBeDefined();
});

test('separates a linearly separable dataset', () => {
    const X = [[1], [2], [3], [4], [6], [7], [8], [9]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];
    const clf = new GradientBoostingClassifier({ nEstimators: 20, randomState: 0 });
    clf.fit(X, y);
    expect(clf.predict(X)).toEqual(y);
});

test('supports arbitrary binary labels', () => {
    const X = [[1], [2], [3], [4], [6], [7], [8], [9]];
    const y = [-5, -5, -5, -5, 3, 3, 3, 3];
    const clf = new GradientBoostingClassifier({ nEstimators: 20, randomState: 0 });
    clf.fit(X, y);
    expect(clf.predict(X)).toEqual(y);
    expect(clf.predict([[0]])).toEqual([-5]);
});

test('learns XOR (needs depth-2 interactions)', () => {
    const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const y = [0, 1, 1, 0];
    const clf = new GradientBoostingClassifier({ nEstimators: 30, maxDepth: 2, randomState: 0 });
    clf.fit(X, y);
    expect(clf.predict(X)).toEqual(y);
});

test('predictProba is calibrated toward the true class and sums to 1', () => {
    const X = [[1], [2], [3], [4], [6], [7], [8], [9]];
    const y = [0, 0, 0, 0, 1, 1, 1, 1];
    const clf = new GradientBoostingClassifier({ nEstimators: 50, randomState: 0 });
    clf.fit(X, y);
    const proba = clf.predictProba(X);
    for (let i = 0; i < X.length; i++) {
        expect(proba[i][0] + proba[i][1]).toBeCloseTo(1, 9);
        expect(proba[i][y[i]]).toBeGreaterThan(0.8);
    }
});

test('randomState makes subsampled fit reproducible', () => {
    const X = Array.from({ length: 60 }, (_, i) => [i, (i * 7) % 13]);
    const y = X.map(([a, b]) => (a + b > 35 ? 1 : 0));
    const a = new GradientBoostingClassifier({ nEstimators: 20, subsample: 0.6, randomState: 5 });
    const b = new GradientBoostingClassifier({ nEstimators: 20, subsample: 0.6, randomState: 5 });
    a.fit(X, y);
    b.fit(X, y);
    expect(a.predict(X)).toEqual(b.predict(X));
});

test('throws on invalid inputs', () => {
    expect(() => new GradientBoostingClassifier({ nEstimators: 0 })).toThrow();
    expect(() => new GradientBoostingClassifier({ learningRate: -1 })).toThrow();
    expect(() => new GradientBoostingClassifier({ subsample: 2 })).toThrow();
    expect(() => new GradientBoostingClassifier({ maxDepth: 0 })).toThrow();
    const clf = new GradientBoostingClassifier();
    expect(() => clf.predict([[1]])).toThrow();
    expect(() => clf.fit([[1], [2]], [1, 2, 3])).toThrow();
    // more than two classes is not supported yet
    expect(() => clf.fit([[1], [2], [3]], [0, 1, 2])).toThrow();
    // single class is not valid binary classification
    expect(() => clf.fit([[1], [2]], [1, 1])).toThrow();
});
