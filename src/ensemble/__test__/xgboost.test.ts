import { XGBoostClassifier, XGBoostRegressor } from '../xgboost';

describe('XGBoostRegressor', () => {
    test('init', () => {
        expect(new XGBoostRegressor()).toBeDefined();
    });

    test('fits a nonlinear function well', () => {
        const X = Array.from({ length: 100 }, (_, i) => [i / 10]);
        const y = X.map(([v]) => v * v - 3 * v + 2);
        const reg = new XGBoostRegressor({ nEstimators: 50, learningRate: 0.3, maxDepth: 4, randomState: 0 });
        reg.fit(X, y);
        const pred = reg.predict(X);
        const mse = pred.reduce((acc, p, i) => acc + (p - y[i]) ** 2, 0) / pred.length;
        expect(mse).toBeLessThan(0.1);
    });

    test('constant target converges to the constant', () => {
        const X = [[1], [2], [3], [4]];
        const y = [7, 7, 7, 7];
        // lambda shrinks each step, so convergence is geometric: (1 - eta * n/(n+lambda))^rounds
        const reg = new XGBoostRegressor({ nEstimators: 60, randomState: 0 });
        reg.fit(X, y);
        for (const p of reg.predict([[1.5], [3.5]])) {
            expect(p).toBeCloseTo(7, 3);
        }
    });

    test('lambda regularization shrinks leaf weights', () => {
        const X = Array.from({ length: 20 }, (_, i) => [i]);
        const y = X.map(([v]) => v);
        const fitWith = (lambda: number) => {
            const reg = new XGBoostRegressor({ nEstimators: 1, learningRate: 1, maxDepth: 1, lambda, randomState: 0 });
            reg.fit(X, y);
            return reg.predict(X);
        };
        const spread = (pred: number[]) => Math.max(...pred) - Math.min(...pred);
        // heavier regularization pulls the two leaf values closer together
        expect(spread(fitWith(100))).toBeLessThan(spread(fitWith(0)));
    });

    test('gamma prunes low-gain splits', () => {
        const X = Array.from({ length: 20 }, (_, i) => [i]);
        const y = X.map(([v]) => (v < 10 ? 0 : 10));
        const fitWith = (gamma: number) => {
            const reg = new XGBoostRegressor({ nEstimators: 1, learningRate: 1, maxDepth: 3, gamma, randomState: 0 });
            reg.fit(X, y);
            return new Set(reg.predict(X)).size;
        };
        expect(fitWith(1000)).toBe(1); // gain of the best split is ~219 < gamma: no split survives
        expect(fitWith(0)).toBeGreaterThan(1);
    });

    test('randomState makes subsampled fit reproducible', () => {
        const X = Array.from({ length: 60 }, (_, i) => [i, (i * 3) % 11]);
        const y = X.map(([a, b]) => a + 2 * b);
        const a = new XGBoostRegressor({ nEstimators: 20, subsample: 0.7, colsampleByTree: 0.5, randomState: 9 });
        const b = new XGBoostRegressor({ nEstimators: 20, subsample: 0.7, colsampleByTree: 0.5, randomState: 9 });
        a.fit(X, y);
        b.fit(X, y);
        expect(a.predict(X)).toEqual(b.predict(X));
    });

    test('throws on invalid inputs', () => {
        expect(() => new XGBoostRegressor({ nEstimators: 0 })).toThrow();
        expect(() => new XGBoostRegressor({ learningRate: 0 })).toThrow();
        expect(() => new XGBoostRegressor({ lambda: -1 })).toThrow();
        expect(() => new XGBoostRegressor({ gamma: -1 })).toThrow();
        expect(() => new XGBoostRegressor({ minChildWeight: -1 })).toThrow();
        expect(() => new XGBoostRegressor({ subsample: 0 })).toThrow();
        expect(() => new XGBoostRegressor({ colsampleByTree: 1.5 })).toThrow();
        expect(() => new XGBoostRegressor({ maxDepth: 0 })).toThrow();
        const reg = new XGBoostRegressor();
        expect(() => reg.predict([[1]])).toThrow();
        expect(() => reg.fit([], [])).toThrow();
        expect(() => reg.fit([[1]], [1, 2])).toThrow();
    });
});

describe('XGBoostClassifier', () => {
    test('separates a linearly separable dataset', () => {
        const X = [[1], [2], [3], [4], [6], [7], [8], [9]];
        const y = [0, 0, 0, 0, 1, 1, 1, 1];
        const clf = new XGBoostClassifier({ nEstimators: 20, randomState: 0 });
        clf.fit(X, y);
        expect(clf.predict(X)).toEqual(y);
    });

    test('supports arbitrary binary labels', () => {
        const X = [[1], [2], [3], [4], [6], [7], [8], [9]];
        const y = [-5, -5, -5, -5, 3, 3, 3, 3];
        const clf = new XGBoostClassifier({ nEstimators: 20, randomState: 0 });
        clf.fit(X, y);
        expect(clf.predict(X)).toEqual(y);
    });

    test('learns a two-feature interaction (AND)', () => {
        // perfectly symmetric XOR has G_L = G_R = 0 for every first split
        // (zero gain), which exact-greedy xgboost cannot split either; AND
        // still requires depth-2 interactions but has usable marginal gain
        const X: number[][] = [];
        const y: number[] = [];
        for (let a = 0; a < 4; a++) {
            for (let b = 0; b < 4; b++) {
                X.push([a, b]);
                y.push(a >= 2 && b >= 2 ? 1 : 0);
            }
        }
        const clf = new XGBoostClassifier({ nEstimators: 30, maxDepth: 2, minChildWeight: 0, randomState: 0 });
        clf.fit(X, y);
        expect(clf.predict(X)).toEqual(y);
    });

    test('predictProba sums to 1 and favors the true class', () => {
        const X = [[1], [2], [3], [4], [6], [7], [8], [9]];
        const y = [0, 0, 0, 0, 1, 1, 1, 1];
        const clf = new XGBoostClassifier({ nEstimators: 30, minChildWeight: 0, randomState: 0 });
        clf.fit(X, y);
        const proba = clf.predictProba(X);
        for (let i = 0; i < X.length; i++) {
            expect(proba[i][0] + proba[i][1]).toBeCloseTo(1, 9);
            expect(proba[i][y[i]]).toBeGreaterThan(0.5);
        }
    });

    test('throws on non-binary targets and invalid baseScore', () => {
        const clf = new XGBoostClassifier();
        expect(() => clf.fit([[1], [2], [3]], [0, 1, 2])).toThrow();
        expect(() => clf.fit([[1], [2]], [1, 1])).toThrow();
        expect(() => new XGBoostClassifier({ baseScore: 1 })).toThrow();
        expect(() => new XGBoostClassifier({ baseScore: 0 })).toThrow();
    });

    test('lambda=0 with saturated probabilities stays finite', () => {
        const X = [[1], [2], [3], [4], [6], [7], [8], [9]];
        const y = [0, 0, 0, 0, 1, 1, 1, 1];
        const clf = new XGBoostClassifier({ nEstimators: 300, lambda: 0, minChildWeight: 0, randomState: 0 });
        clf.fit(X, y);
        expect(clf.predict(X)).toEqual(y);
        for (const [p0, p1] of clf.predictProba(X)) {
            expect(Number.isFinite(p0)).toBe(true);
            expect(Number.isFinite(p1)).toBe(true);
        }
    });

    test('a failed refit does not corrupt a fitted model', () => {
        const X = [[1], [2], [3], [4], [6], [7], [8], [9]];
        const y = [5, 5, 5, 5, 9, 9, 9, 9];
        const clf = new XGBoostClassifier({ nEstimators: 10, randomState: 0 });
        clf.fit(X, y);
        expect(() => clf.fit([[1], [2], [3]], [0, 1, 2])).toThrow();
        // still predicts with the original label set
        expect(clf.predict([[1], [9]])).toEqual([5, 9]);
    });

    test('rejects eta > 1 and NaN inputs', () => {
        expect(() => new XGBoostRegressor({ learningRate: 2 })).toThrow();
        const reg = new XGBoostRegressor();
        expect(() => reg.fit([[1], [NaN]], [1, 2])).toThrow();
        expect(() => reg.fit([[1], [2]], [1, NaN])).toThrow();
    });
});
