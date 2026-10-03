import { GammaRegressor, PoissonRegressor, TweedieRegressor } from '../glm';
import { loadModel } from '../../base';

const cases = [
    { Model: TweedieRegressor, expected: [.3, 2.6, 4.9, 7.2] },
    { Model: PoissonRegressor, expected: [1, 2, 4, 8] },
    { Model: GammaRegressor, expected: [1, 2, 4, 8] },
];

describe.each(cases)('$Model rank-deficient fitting', ({ Model, expected }) => {
    test.each([
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0]],
    ])('fits redundant and constant columns', (...X: number[][]) => {
        const model = new Model({ alpha: 0, tol: 1e-9, maxIter: 200 });
        model.fit(X, [1, 2, 4, 8]);
        const actual = model.predict(X);
        actual.forEach((value, i) => expect(value).toBeCloseTo(expected[i], 6));
        expect(model.converged).toBe(true);
        expect(model.terminationReason).toBe('converged');
        const restored = loadModel(JSON.parse(JSON.stringify(model.toJSON()))) as typeof model;
        expect(restored.predict(X)).toEqual(actual);
        expect(restored.terminationReason).toBe('converged');
    });

    test('supports more coefficients than observations', () => {
        const X = [[1, 0, 1, 0], [0, 1, 0, 1]], y = [2, 4];
        const model = new Model({ alpha: 0, tol: 1e-9, maxIter: 200 });
        model.fit(X, y);
        model.predict(X).forEach((value, i) => expect(value).toBeCloseTo(y[i], 6));
    });
});

test('GLM distinguishes an iteration limit from convergence', () => {
    const model = new PoissonRegressor({ alpha: 0, tol: 1e-12, maxIter: 1 });
    model.fit([[0], [1], [2], [3]], [1, 2, 4, 8]);
    expect(model.nIter).toBe(1);
    expect(model.converged).toBe(false);
    expect(model.terminationReason).toBe('maxIter');
    expect(model.predict([[0], [1]]).every(Number.isFinite)).toBe(true);
});

test('non-finite numerical systems fail explicitly and invalidate a previous fit', () => {
    const model = new TweedieRegressor({ alpha: 0 });
    model.fit([[0], [1]], [0, 1]);
    expect(() => model.fit([[1e200], [-1e200]], [1, 2])).toThrow(/GLM optimization failed/);
    expect(model.converged).toBe(false);
    expect(model.terminationReason).toBe('numericalFailure');
    expect(() => model.predict([[0]])).toThrow(/not fitted/);
});
