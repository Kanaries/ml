import { GaussianProcessRegressor } from '../gaussianProcessRegressor';

test('prior covariance uses anisotropic RBF lengths in the original feature order', () => {
    const model = new GaussianProcessRegressor({ lengthScale: [1, 2], signalVariance: 4 });
    const X = [[0, 0], [0, 2], [2, 0]];
    const { mean, cov } = model.predict(X, { returnCov: true });
    expect(mean).toEqual([0, 0, 0]);
    expect(model.predict(X, { returnStd: true }).std).toEqual([2, 2, 2]);
    expect(cov[0][1]).toBeCloseTo(4 * Math.exp(-0.5), 14);
    expect(cov[0][2]).toBeCloseTo(4 * Math.exp(-2), 14);
    expect(() => model.predict([[1]])).toThrow(/lengthScale/);
    expect(() => model.predict([[NaN, 0]])).toThrow(/finite/);
});

test('a failed refit preserves a usable trained posterior', () => {
    const model = new GaussianProcessRegressor({ alpha: 0 });
    model.fit([[0], [1]], [3, 4]);
    const expected = model.predict([[0.5]], { returnStd: true });
    expect(() => model.fit([[0], [0]], [3, 4])).toThrow(/alpha/);
    expect(model.predict([[0.5]], { returnStd: true })).toEqual(expected);
    expect(() => model.predict([[0, 1]])).toThrow(/feature count/);
});

test('constant targets normalize exactly and observation-specific noise may span many orders', () => {
    const model = new GaussianProcessRegressor({ normalizeY: true, alpha: [1e30, 0, 0.1] });
    model.fit([[0], [1], [2]], [7.1, 7.1, 7.1]);
    const { mean, std } = model.predict([[1], [2]], { returnStd: true });
    expect(mean).toEqual([7.1, 7.1]);
    expect(std[0]).toBeLessThan(1e-7);
    expect(std[1]).toBeGreaterThan(0);
    expect(() => model.fit([[1]], [2])).toThrow(/one noise variance/);
});

test('optional bounded search improves likelihood and leaves the initial parameters reusable', () => {
    const X = Array.from({ length: 18 }, (_, i) => [i / 5]);
    const y = X.map(([x]) => Math.sin(3 * x));
    const initial = { lengthScale: 3, signalVariance: 0.3, alpha: 0.01 };
    const fixed = new GaussianProcessRegressor(initial);
    fixed.fit(X, y);
    const model = new GaussianProcessRegressor({
        ...initial,
        optimizer: { lengthScaleBounds: [0.1, 3], signalVarianceBounds: [0.3, 2], maxIter: 30 },
    });
    model.fit(X, y);
    expect(model.logMarginalLikelihood()).toBeGreaterThan(fixed.logMarginalLikelihood() + 1);
    expect(model.fittedKernel.lengthScale).toBeLessThan(3);
    expect(model.fittedKernel.lengthScale).toBeGreaterThanOrEqual(0.1);
    expect(model.getParams().lengthScale).toBe(3);
    expect(model.getParams().signalVariance).toBe(0.3);
});

test('constructor rejects invalid kernel, noise and optimizer settings', () => {
    expect(() => new GaussianProcessRegressor({ lengthScale: 0 })).toThrow(/lengthScale/);
    expect(() => new GaussianProcessRegressor({ signalVariance: Infinity })).toThrow(/signalVariance/);
    expect(() => new GaussianProcessRegressor({ alpha: [-1] })).toThrow(/alpha/);
    expect(() => new GaussianProcessRegressor({ optimizer: { maxIter: 0 } })).toThrow(/maxIter/);
    expect(() => new GaussianProcessRegressor({ optimizer: { lengthScaleBounds: [2, 1] } })).toThrow(/bounds/);
});
