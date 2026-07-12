import { SVR } from '../svr';

// 40 evenly-spaced samples of y = sin(x) on [0, 6]
const sineX: number[][] = [];
const sineY: number[] = [];
for (let i = 0; i < 40; i++) {
    const x = (6 * i) / 39;
    sineX.push([x]);
    sineY.push(Math.sin(x));
}

describe('SVR (epsilon-SVR, SMO)', () => {
    test('rbf kernel fits a sine wave with R^2 > 0.95', () => {
        const reg = new SVR({ kernel: 'rbf', C: 10 });
        reg.fit(sineX, sineY);
        expect(reg.score(sineX, sineY)).toBeGreaterThan(0.95);
    });

    test('linear kernel recovers y = 2x + 1 up to the epsilon-tube slack', () => {
        // Exact linear data on [0, 10] with epsilon = 0.1 and large C: the
        // optimum is the *flattest* line through the tube (b is
        // unregularized, so only the slope is shrunk). A line inside the
        // tube must satisfy |f(0) - 1| <= 0.1 and |f(10) - 21| <= 0.1, so
        // the minimal-|w| solution has slope (20.9 - 1.1)/10 = 1.98 and
        // intercept 1.1 (tube edges at both ends). Assert around that.
        const X = Array.from({ length: 21 }, (_, i) => [i * 0.5]);
        const y = X.map(([x]) => 2 * x + 1);
        const reg = new SVR({ kernel: 'linear', C: 100, epsilon: 0.1 });
        reg.fit(X, y);
        const [p0, p10] = reg.predict([[0], [10]]);
        const slope = (p10 - p0) / 10;
        expect(slope).toBeGreaterThan(1.97);
        expect(slope).toBeLessThan(2.01);
        // intercept within the epsilon tube around the true intercept 1
        expect(Math.abs(p0 - 1)).toBeLessThanOrEqual(0.1 + 0.02);
        // every training point stays inside the tube (plus solver tolerance):
        // the tube is satisfiable with zero slack and C is large, so no
        // point may end up outside it at the optimum
        const pred = reg.predict(X);
        pred.forEach((p, i) => {
            expect(Math.abs(p - y[i])).toBeLessThanOrEqual(0.1 + 0.02);
        });
    });

    test('widening epsilon reduces the number of support vectors', () => {
        const narrow = new SVR({ kernel: 'rbf', C: 10, epsilon: 0.01 });
        narrow.fit(sineX, sineY);
        const wide = new SVR({ kernel: 'rbf', C: 10, epsilon: 0.5 });
        wide.fit(sineX, sineY);
        const nNarrow = narrow.getSupportVectors().length;
        const nWide = wide.getSupportVectors().length;
        expect(nNarrow).toBeGreaterThan(0);
        expect(nWide).toBeGreaterThan(0);
        // only points on/outside the tube keep a nonzero dual coefficient,
        // so a wider tube must strictly drop support vectors here
        expect(nWide).toBeLessThan(nNarrow);
        // sin has amplitude 1, so a tube of half-width 1.5 around y contains
        // a constant function with zero slack: every alpha stays 0
        const huge = new SVR({ kernel: 'rbf', C: 10, epsilon: 1.5 });
        huge.fit(sineX, sineY);
        expect(huge.getSupportVectors().length).toBe(0);
    });

    test('support-vector indices are sorted training indices', () => {
        const reg = new SVR({ kernel: 'rbf', C: 10, epsilon: 0.2 });
        reg.fit(sineX, sineY);
        const sv = reg.getSupportVectors();
        for (let i = 1; i < sv.length; i++) {
            expect(sv[i]).toBeGreaterThan(sv[i - 1]);
        }
        expect(sv[0]).toBeGreaterThanOrEqual(0);
        expect(sv[sv.length - 1]).toBeLessThan(sineX.length);
    });

    test('predict before fit throws, invalid params throw', () => {
        expect(() => new SVR().predict([[0]])).toThrow();
        expect(() => new SVR({ C: 0 })).toThrow();
        expect(() => new SVR({ C: -1 })).toThrow();
        expect(() => new SVR({ epsilon: -0.1 })).toThrow();
        expect(() => new SVR().fit([], [])).toThrow();
        expect(() => new SVR().fit([[0]], [1, 2])).toThrow();
    });
});
