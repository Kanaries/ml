import { GradientBoostingRegressor } from '../gradientBoostingRegressor';

test('GradientBoostingRegressor init', () => {
    expect(new GradientBoostingRegressor()).toBeDefined();
});

test('fits a nonlinear function well', () => {
    const X = Array.from({ length: 100 }, (_, i) => [i / 10]);
    const y = X.map(([v]) => v * v - 3 * v + 2);
    const reg = new GradientBoostingRegressor({ nEstimators: 100, learningRate: 0.1, maxDepth: 3, randomState: 0 });
    reg.fit(X, y);
    const pred = reg.predict(X);
    const mse = pred.reduce((acc, p, i) => acc + (p - y[i]) ** 2, 0) / pred.length;
    expect(mse).toBeLessThan(0.1);
});

test('constant target converges to the constant', () => {
    const X = [[1], [2], [3], [4]];
    const y = [7, 7, 7, 7];
    const reg = new GradientBoostingRegressor({ nEstimators: 10, randomState: 0 });
    reg.fit(X, y);
    for (const p of reg.predict([[1.5], [3.5]])) {
        expect(p).toBeCloseTo(7, 6);
    }
});

test('more estimators reduce training error', () => {
    const X = Array.from({ length: 60 }, (_, i) => [i]);
    const y = X.map(([v]) => Math.sin(v / 5) * 10 + v);
    const mseOf = (nEstimators: number) => {
        const reg = new GradientBoostingRegressor({ nEstimators, learningRate: 0.1, maxDepth: 2, randomState: 0 });
        reg.fit(X, y);
        const pred = reg.predict(X);
        return pred.reduce((acc, p, i) => acc + (p - y[i]) ** 2, 0) / pred.length;
    };
    expect(mseOf(100)).toBeLessThan(mseOf(5));
});

test('randomState makes subsampled fit reproducible', () => {
    const X = Array.from({ length: 50 }, (_, i) => [i, i % 7]);
    const y = X.map(([a, b]) => a + 3 * b);
    const a = new GradientBoostingRegressor({ nEstimators: 20, subsample: 0.6, randomState: 11 });
    const b = new GradientBoostingRegressor({ nEstimators: 20, subsample: 0.6, randomState: 11 });
    a.fit(X, y);
    b.fit(X, y);
    expect(a.predict(X)).toEqual(b.predict(X));
});

test('subsample < 1 still fits reasonably', () => {
    const X = Array.from({ length: 100 }, (_, i) => [i / 10]);
    const y = X.map(([v]) => 2 * v + 5);
    const reg = new GradientBoostingRegressor({ nEstimators: 50, subsample: 0.5, randomState: 3 });
    reg.fit(X, y);
    const pred = reg.predict(X);
    const mse = pred.reduce((acc, p, i) => acc + (p - y[i]) ** 2, 0) / pred.length;
    expect(mse).toBeLessThan(1);
});

test('throws on invalid inputs', () => {
    expect(() => new GradientBoostingRegressor({ subsample: 0 })).toThrow();
    expect(() => new GradientBoostingRegressor({ subsample: NaN })).toThrow();
    expect(() => new GradientBoostingRegressor({ nEstimators: 0 })).toThrow();
    expect(() => new GradientBoostingRegressor({ nEstimators: 2.5 })).toThrow();
    expect(() => new GradientBoostingRegressor({ nEstimators: Infinity })).toThrow();
    expect(() => new GradientBoostingRegressor({ learningRate: 0 })).toThrow();
    expect(() => new GradientBoostingRegressor({ learningRate: NaN })).toThrow();
    expect(() => new GradientBoostingRegressor({ maxDepth: 0 })).toThrow();
    expect(() => new GradientBoostingRegressor({ minSamplesSplit: 1 })).toThrow();
    const reg = new GradientBoostingRegressor();
    expect(() => reg.predict([[1]])).toThrow();
    expect(() => reg.fit([], [])).toThrow();
    expect(() => reg.fit([[1]], [1, 2])).toThrow();
});
