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
