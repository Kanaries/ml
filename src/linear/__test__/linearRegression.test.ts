import { LinearRegression } from '../linearRegression';

test('init', () => {
    const lr = new LinearRegression();
    expect(lr).toBeDefined();
});

test('basic one variable', () => {
    const X: number[][] = [];
    const Y: number[] = [];
    for (let i = 0; i < 20; i++) {
        const x = i;
        X.push([x]);
        Y.push(2 * x + 3);
    }
    const lr = new LinearRegression();
    lr.fit(X, Y);
    const pred = lr.predict([[30], [40]]);
    expect(pred[0]).toBeCloseTo(63);
    expect(pred[1]).toBeCloseTo(83);
});

test('two variables', () => {
    const X = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ];
    const Y = [4, 6, 1, 3]; // y = 4 + 2*x1 -3*x2
    const lr = new LinearRegression();
    lr.fit(X, Y);
    const pred = lr.predict([[2, 3]]);
    expect(pred[0]).toBeCloseTo(4 + 2 * 2 - 3 * 3);
});

describe('collinear and near-collinear inputs (QR least squares)', () => {
    test('exactly collinear features throw an informative error', () => {
        const m = new LinearRegression();
        expect(() => m.fit([[1, 1], [2, 2], [3, 3]], [1, 2, 3])).toThrow(/collinear|singular/i);
    });

    test('near-collinear features do not silently produce garbage', () => {
        const m = new LinearRegression();
        m.fit(
            [
                [1, 1.0000001],
                [2, 2.0000002],
                [3, 3.0000004],
            ],
            [1, 2, 3]
        );
        expect(m.predict([[4, 4]])[0]).toBeCloseTo(4, 2);
    });

    test('predict before fit throws', () => {
        expect(() => new LinearRegression().predict([[1]])).toThrow();
    });

    test('empty input throws', () => {
        expect(() => new LinearRegression().fit([], [])).toThrow();
    });
});
