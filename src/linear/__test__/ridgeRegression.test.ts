import { LinearRegression } from '../linearRegression';
import { RidgeRegression } from '../ridgeRegression';

test('init', () => {
    const rr = new RidgeRegression();
    expect(rr).toBeDefined();
});

test('alpha zero behaves like linear regression', () => {
    const X: number[][] = [];
    const Y: number[] = [];
    for (let i = 0; i < 20; i++) {
        X.push([i]);
        Y.push(2 * i + 3);
    }

    const lr = new LinearRegression();
    lr.fit(X, Y);
    const lrPred = lr.predict([[30], [40]]);

    const rr = new RidgeRegression({ alpha: 0 });
    rr.fit(X, Y);
    const rrPred = rr.predict([[30], [40]]);

    expect(rrPred[0]).toBeCloseTo(lrPred[0], 8);
    expect(rrPred[1]).toBeCloseTo(lrPred[1], 8);
});

test('stronger regularization shrinks prediction magnitude', () => {
    const X = [[-3], [-2], [-1], [0], [1], [2], [3]];
    const Y = X.map(([x]) => 1 + 3 * x);

    const weak = new RidgeRegression({ alpha: 0.01 });
    weak.fit(X, Y);
    const weakPred = weak.predict([[10]])[0];

    const strong = new RidgeRegression({ alpha: 100 });
    strong.fit(X, Y);
    const strongPred = strong.predict([[10]])[0];

    expect(Math.abs(strongPred - 1)).toBeLessThan(Math.abs(weakPred - 1));
});

test('fitIntercept=false keeps zero intercept model', () => {
    const X = [[1], [2], [3], [4], [5]];
    const Y = X.map(([x]) => 2 * x);

    const rr = new RidgeRegression({ alpha: 0, fitIntercept: false });
    rr.fit(X, Y);
    const pred = rr.predict([[10]])[0];

    expect(pred).toBeCloseTo(20, 8);
});

test('throws on invalid alpha', () => {
    expect(() => new RidgeRegression({ alpha: -1 })).toThrow('alpha must be a finite number >= 0');
});

test('throws on invalid fit data', () => {
    const rr = new RidgeRegression();
    expect(() => rr.fit([], [])).toThrow('X and Y must be non-empty');
    expect(() => rr.fit([[1], [2]], [1])).toThrow('X and Y must have the same length');
    expect(() => rr.fit([[], []], [1, 2])).toThrow('X must have at least one feature');
    expect(() => rr.fit([[1], [1, 2]], [1, 2])).toThrow('all rows in X must have the same length');
});

test('throws when predict called before fit', () => {
    const rr = new RidgeRegression();
    expect(() => rr.predict([[1]])).toThrow('model is not fitted');
});
