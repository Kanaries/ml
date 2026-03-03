import { LinearRegression } from '../linearRegression';
import { LassoRegression } from '../lassoRegression';

test('init', () => {
    const lr = new LassoRegression();
    expect(lr).toBeDefined();
});

test('alpha zero behaves like linear regression', () => {
    const X: number[][] = [];
    const Y: number[] = [];
    for (let i = 0; i < 20; i++) {
        X.push([i]);
        Y.push(2 * i + 3);
    }

    const ols = new LinearRegression();
    ols.fit(X, Y);
    const olsPred = ols.predict([[30], [40]]);

    const lasso = new LassoRegression({ alpha: 0, maxIter: 20000, tol: 1e-12 });
    lasso.fit(X, Y);
    const pred = lasso.predict([[30], [40]]);

    expect(pred[0]).toBeCloseTo(olsPred[0], 6);
    expect(pred[1]).toBeCloseTo(olsPred[1], 6);
});

test('stronger regularization shrinks prediction magnitude', () => {
    const X = [[-3], [-2], [-1], [0], [1], [2], [3]];
    const Y = X.map(([x]) => 1 + 3 * x);

    const weak = new LassoRegression({ alpha: 0.01, maxIter: 20000, tol: 1e-12 });
    weak.fit(X, Y);
    const weakPred = weak.predict([[10]])[0];

    const strong = new LassoRegression({ alpha: 1, maxIter: 20000, tol: 1e-12 });
    strong.fit(X, Y);
    const strongPred = strong.predict([[10]])[0];

    expect(Math.abs(strongPred - 1)).toBeLessThan(Math.abs(weakPred - 1));
});

test('fitIntercept=false keeps zero intercept model', () => {
    const X = [[1], [2], [3], [4], [5]];
    const Y = X.map(([x]) => 2 * x);

    const lasso = new LassoRegression({ alpha: 0.01, fitIntercept: false, maxIter: 20000, tol: 1e-12 });
    lasso.fit(X, Y);
    const pred = lasso.predict([[10]])[0];

    expect(pred).toBeCloseTo(19.99090909, 6);
});

test('throws on invalid constructor args', () => {
    expect(() => new LassoRegression({ alpha: -1 })).toThrow('alpha must be a finite number >= 0');
    expect(() => new LassoRegression({ maxIter: 0 })).toThrow('maxIter must be an integer > 0');
    expect(() => new LassoRegression({ tol: 0 })).toThrow('tol must be a finite number > 0');
});

test('throws on invalid fit data', () => {
    const lasso = new LassoRegression();
    expect(() => lasso.fit([], [])).toThrow('X and Y must be non-empty');
    expect(() => lasso.fit([[1], [2]], [1])).toThrow('X and Y must have the same length');
    expect(() => lasso.fit([[], []], [1, 2])).toThrow('X must have at least one feature');
    expect(() => lasso.fit([[1], [1, 2]], [1, 2])).toThrow('all rows in X must have the same length');
});

test('throws when predict called before fit', () => {
    const lasso = new LassoRegression();
    expect(() => lasso.predict([[1]])).toThrow('model is not fitted');
});
