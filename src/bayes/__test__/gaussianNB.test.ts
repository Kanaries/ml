import { GaussianNB } from '../gaussianNB';

test('init', () => {
    const nb = new GaussianNB();
    expect(nb).toBeDefined();
});

test('GaussianNB matches sklearn reference predictions on a separable dataset', () => {
    const X = [
        [-2, -1],
        [-1, -1],
        [-1, -2],
        [1, 1],
        [2, 1],
        [1, 2],
    ];
    const y = [0, 0, 0, 1, 1, 1];
    const testX = [
        [-1.5, -1.5],
        [1.5, 1.5],
        [0, 0],
    ];

    const nb = new GaussianNB();
    nb.fit(X, y);

    expect(nb.predict(testX)).toEqual([0, 1, 0]);
});

describe('GaussianNB.predictProba', () => {
    it('returns normalized posteriors consistent with predict', () => {
        const X = [[0, 0], [0.5, 0.2], [5, 5], [5.5, 4.8], [0.2, 0.4], [5.1, 5.3]];
        const y = [0, 0, 1, 1, 0, 1];
        const nb = new GaussianNB();
        nb.fit(X, y);
        const proba = nb.predictProba(X);
        const preds = nb.predict(X);
        expect(proba).toHaveLength(X.length);
        for (let i = 0; i < proba.length; i++) {
            const sum = proba[i].reduce((acc, v) => acc + v, 0);
            expect(sum).toBeCloseTo(1, 10);
            expect(nb.getClasses()[proba[i].indexOf(Math.max(...proba[i]))]).toBe(preds[i]);
        }
        // far into class-0 territory the posterior must be near-certain
        expect(nb.predictProba([[0, 0]])[0][0]).toBeGreaterThan(0.99);
    });
});
