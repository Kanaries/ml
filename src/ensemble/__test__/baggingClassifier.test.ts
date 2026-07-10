import { BaggingClassifier } from '../baggingClassifier';

test('BaggingClassifier initializes', () => {
    expect(new BaggingClassifier()).toBeDefined();
});

test('BaggingClassifier fits and predicts a separable dataset', () => {
    const X = [[0], [1], [2], [10], [11], [12]];
    const y = [0, 0, 0, 1, 1, 1];

    const clf = new BaggingClassifier({ nEstimators: 7, randomState: 42 });
    clf.fit(X, y);

    expect(clf.predict([[0.2], [11.5]])).toEqual([0, 1]);
});

test('BaggingClassifier validates fit lifecycle', () => {
    const clf = new BaggingClassifier();
    expect(() => clf.predict([[0]])).toThrow('model is not fitted');
});

describe('fractional maxSamples (unified subset semantics)', () => {
    const X = Array.from({ length: 40 }, (_, i) => [i, (i * 13) % 40]);
    const Y = X.map(row => (row[0] > 19 ? 1 : 0));

    test('bootstrap: maxSamples=0.8 trains on 32 samples per estimator, not 1', () => {
        const clf = new BaggingClassifier({ maxSamples: 0.8, nEstimators: 7, randomState: 1 });
        clf.fit(X, Y);
        const acc = clf.predict(X).filter((p, i) => p === Y[i]).length / Y.length;
        expect(acc).toBeGreaterThan(0.9);
    });

    test('no bootstrap: fractional maxSamples subsamples without replacement', () => {
        const clf = new BaggingClassifier({ maxSamples: 0.5, bootstrap: false, nEstimators: 7, randomState: 1 });
        expect(() => clf.fit(X, Y)).not.toThrow();
        const acc = clf.predict(X).filter((p, i) => p === Y[i]).length / Y.length;
        expect(acc).toBeGreaterThan(0.9);
    });

    test('invalid maxSamples throws', () => {
        expect(() => new BaggingClassifier({ maxSamples: 0 }).fit(X, Y)).toThrow();
        expect(() => new BaggingClassifier({ maxSamples: 1.5 }).fit(X, Y)).toThrow();
    });
});
