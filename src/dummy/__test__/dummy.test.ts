import { DummyClassifier, DummyRegressor } from '../dummy';

const X3 = [[0], [0], [0]];
const Y3 = [0, 0, 1];

describe('DummyClassifier', () => {
    it("'mostFrequent' predicts the mode everywhere and predictProba is its one-hot", () => {
        const clf = new DummyClassifier({ strategy: 'mostFrequent' });
        clf.fit(X3, Y3);
        expect(clf.predict([[1], [2], [3], [4]])).toEqual([0, 0, 0, 0]);
        expect(clf.predictProba([[1], [2]])).toEqual([
            [1, 0],
            [1, 0],
        ]);
    });

    it("'prior' predicts the mode and predictProba returns the class prior for every row", () => {
        const clf = new DummyClassifier({ strategy: 'prior' });
        clf.fit(X3, Y3);
        expect(clf.predict([[9], [9]])).toEqual([0, 0]);
        expect(clf.predictProba([[9], [9]])).toEqual([
            [2 / 3, 1 / 3],
            [2 / 3, 1 / 3],
        ]);
    });

    it('ties in the prior resolve to the smallest class (first argmax)', () => {
        const clf = new DummyClassifier({ strategy: 'mostFrequent' });
        clf.fit([[0], [0], [0], [0]], [3, 1, 3, 1]);
        expect(clf.predict([[0]])).toEqual([1]);
    });

    it('supports sample weights for the prior', () => {
        const clf = new DummyClassifier({ strategy: 'prior' });
        clf.fit(X3, Y3, [1, 1, 6]);
        expect(clf.predict([[0]])).toEqual([1]);
        expect(clf.predictProba([[0]])).toEqual([[0.25, 0.75]]);
    });

    it("'uniform' predictProba is 1/nClasses and predictions are seeded-deterministic", () => {
        const clf = new DummyClassifier({ strategy: 'uniform', randomState: 42 });
        clf.fit(X3, Y3);
        expect(clf.predictProba([[0], [0]])).toEqual([
            [0.5, 0.5],
            [0.5, 0.5],
        ]);
        const X = Array.from({ length: 50 }, () => [0]);
        const first = clf.predict(X);
        expect(clf.predict(X)).toEqual(first); // fresh seeded RNG per call
        expect(new Set(first)).toEqual(new Set([0, 1]));
    });

    it("'stratified' samples from the empirical distribution, deterministically per seed", () => {
        const clf = new DummyClassifier({ strategy: 'stratified', randomState: 7 });
        clf.fit(X3, Y3);
        const X = Array.from({ length: 3000 }, () => [0]);
        const first = clf.predict(X);
        expect(clf.predict(X)).toEqual(first);
        const fractionZero = first.filter((v) => v === 0).length / first.length;
        expect(fractionZero).toBeGreaterThan(2 / 3 - 0.05);
        expect(fractionZero).toBeLessThan(2 / 3 + 0.05);

        const proba = clf.predictProba([[0], [0], [0]]);
        expect(clf.predictProba([[0], [0], [0]])).toEqual(proba);
        for (const row of proba) {
            expect(row.reduce((a, b) => a + b, 0)).toBe(1);
            expect(row.filter((v) => v === 1)).toHaveLength(1);
            expect(row.filter((v) => v === 0)).toHaveLength(1);
        }

        const other = new DummyClassifier({ strategy: 'stratified', randomState: 8 });
        other.fit(X3, Y3);
        expect(other.predict(X)).not.toEqual(first);
    });

    it("'constant' always predicts the constant with a one-hot predictProba", () => {
        const clf = new DummyClassifier({ strategy: 'constant', constant: 1 });
        clf.fit(X3, Y3);
        expect(clf.predict([[5], [6]])).toEqual([1, 1]);
        expect(clf.predictProba([[5]])).toEqual([[0, 1]]);
    });

    it('validates strategy, constant and fit inputs', () => {
        expect(() => new DummyClassifier({ strategy: 'mode' as never })).toThrow(/Unknown strategy/);
        const missing = new DummyClassifier({ strategy: 'constant' });
        expect(() => missing.fit(X3, Y3)).toThrow(/constant/);
        const absent = new DummyClassifier({ strategy: 'constant', constant: 2 });
        expect(() => absent.fit(X3, Y3)).toThrow(/not present/);
        const clf = new DummyClassifier();
        expect(() => clf.fit([[1]], [1, 2])).toThrow(/same length/);
        expect(() => clf.predict([[1]])).toThrow(/must be fitted/);
        expect(() => clf.predictProba([[1]])).toThrow(/must be fitted/);
    });
});

describe('DummyRegressor', () => {
    const X4 = [[0], [0], [0], [0]];

    it("'mean' predicts the target mean (weighted when sampleWeight is given)", () => {
        const reg = new DummyRegressor();
        reg.fit(X4, [1, 2, 3, 4]);
        expect(reg.predict([[9], [9]])).toEqual([2.5, 2.5]);

        const weighted = new DummyRegressor({ strategy: 'mean' });
        weighted.fit(X4, [1, 2, 3, 4], [0, 0, 1, 3]);
        expect(weighted.predict([[0]])).toEqual([3.75]);
    });

    it("'median' averages the two middle values for even n and takes the middle for odd n", () => {
        const even = new DummyRegressor({ strategy: 'median' });
        even.fit(X4, [4, 1, 3, 2]);
        expect(even.predict([[0]])).toEqual([2.5]);

        const odd = new DummyRegressor({ strategy: 'median' });
        odd.fit([[0], [0], [0]], [30, 10, 20]);
        expect(odd.predict([[0]])).toEqual([20]);
    });

    it("'quantile' interpolates linearly", () => {
        const q9 = new DummyRegressor({ strategy: 'quantile', quantile: 0.9 });
        q9.fit(X4, [1, 2, 3, 4]);
        // position 0.9 * 3 = 2.7 -> 3 + 0.7 * (4 - 3)
        expect(q9.predict([[0]])[0]).toBeCloseTo(3.7, 12);

        const q0 = new DummyRegressor({ strategy: 'quantile', quantile: 0 });
        q0.fit(X4, [4, 1, 3, 2]);
        expect(q0.predict([[0]])).toEqual([1]);

        const q1 = new DummyRegressor({ strategy: 'quantile', quantile: 1 });
        q1.fit(X4, [4, 1, 3, 2]);
        expect(q1.predict([[0]])).toEqual([4]);

        const q5 = new DummyRegressor({ strategy: 'quantile', quantile: 0.5 });
        q5.fit(X4, [4, 1, 3, 2]);
        expect(q5.predict([[0]])).toEqual([2.5]); // matches median
    });

    it("'constant' predicts the given value", () => {
        const reg = new DummyRegressor({ strategy: 'constant', constant: 7 });
        reg.fit(X4, [1, 2, 3, 4]);
        expect(reg.predict([[0], [1], [2]])).toEqual([7, 7, 7]);
    });

    it('validates params and inputs', () => {
        expect(() => new DummyRegressor({ strategy: 'mode' as never })).toThrow(/Unknown strategy/);
        expect(() => new DummyRegressor({ strategy: 'quantile' })).toThrow(/quantile/);
        expect(() => new DummyRegressor({ strategy: 'quantile', quantile: 1.5 })).toThrow(/quantile/);
        const missing = new DummyRegressor({ strategy: 'constant' });
        expect(() => missing.fit(X4, [1, 2, 3, 4])).toThrow(/constant/);
        const med = new DummyRegressor({ strategy: 'median' });
        expect(() => med.fit(X4, [1, 2, 3, 4], [1, 1, 1, 1])).toThrow(/sampleWeight is not supported/);
        const reg = new DummyRegressor();
        expect(() => reg.predict([[0]])).toThrow(/must be fitted/);
    });
});
