import { LinearRegression } from '../linearRegression';
import { PolynomialRegression } from '../polynomialRegression';

test('init', () => {
    const pr = new PolynomialRegression();
    expect(pr).toBeDefined();
});

test('fits quadratic one variable', () => {
    const X: number[][] = [];
    const Y: number[] = [];
    for (let i = -5; i <= 5; i++) {
        X.push([i]);
        Y.push(1 + 2 * i + 3 * i * i);
    }

    const pr = new PolynomialRegression({ degree: 2 });
    pr.fit(X, Y);
    const pred = pr.predict([[6], [-6]]);

    expect(pred[0]).toBeCloseTo(1 + 2 * 6 + 3 * 6 * 6, 6);
    expect(pred[1]).toBeCloseTo(1 + 2 * -6 + 3 * 36, 6);
});

test('fits polynomial terms for multiple variables without interaction terms', () => {
    const X = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2]
    ];
    const Y = X.map(([x1, x2]) => 4 + 2 * x1 + 3 * x1 * x1 - 5 * x2 + 2 * x2 * x2);

    const pr = new PolynomialRegression({ degree: 2 });
    pr.fit(X, Y);
    const pred = pr.predict([[3, 1], [1, 3]]);

    expect(pred[0]).toBeCloseTo(4 + 2 * 3 + 3 * 9 - 5 * 1 + 2 * 1, 6);
    expect(pred[1]).toBeCloseTo(4 + 2 * 1 + 3 * 1 - 5 * 3 + 2 * 9, 6);
});

test('degree one matches linear regression', () => {
    const X = [[0], [1], [2], [3], [4], [5]];
    const Y = X.map(([x]) => 7 - 4 * x);

    const lr = new LinearRegression();
    lr.fit(X, Y);
    const lrPred = lr.predict([[10], [-2]]);

    const pr = new PolynomialRegression({ degree: 1 });
    pr.fit(X, Y);
    const prPred = pr.predict([[10], [-2]]);

    expect(prPred[0]).toBeCloseTo(lrPred[0], 6);
    expect(prPred[1]).toBeCloseTo(lrPred[1], 6);
});

test('throws on invalid degree', () => {
    expect(() => new PolynomialRegression({ degree: 0 })).toThrow('degree must be an integer >= 1');
    expect(() => new PolynomialRegression({ degree: 1.5 })).toThrow('degree must be an integer >= 1');
});
