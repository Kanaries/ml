import { NuSVR } from '../nuSVR';

// 40 evenly-spaced samples of y = sin(x) on [0, 6]
const sineX: number[][] = [];
const sineY: number[] = [];
for (let i = 0; i < 40; i++) {
    const x = (6 * i) / 39;
    sineX.push([x]);
    sineY.push(Math.sin(x));
}

describe('NuSVR (nu-SVR, SMO)', () => {
    test('rbf kernel fits a sine wave with R^2 > 0.9', () => {
        const reg = new NuSVR({ kernel: 'rbf', C: 10, nu: 0.5 });
        reg.fit(sineX, sineY);
        expect(reg.score(sineX, sineY)).toBeGreaterThan(0.9);
    });

    test('nu lower-bounds the support-vector fraction', () => {
        // Schölkopf: at the optimum, (#points outside the tube)/n <= nu
        // <= (#SVs)/n. SMO stops at a tol-approximate optimum, so allow a
        // 2/n slack on the finite-sample bound.
        const n = sineX.length;
        for (const nu of [0.2, 0.5, 0.8]) {
            const reg = new NuSVR({ kernel: 'rbf', C: 10, nu });
            reg.fit(sineX, sineY);
            const frac = reg.getSupportVectors().length / n;
            expect(frac).toBeGreaterThanOrEqual(nu - 2 / n);
            expect(frac).toBeLessThanOrEqual(1);
        }
    });

    test('larger nu keeps more support vectors', () => {
        const small = new NuSVR({ kernel: 'rbf', C: 10, nu: 0.2 });
        small.fit(sineX, sineY);
        const large = new NuSVR({ kernel: 'rbf', C: 10, nu: 0.8 });
        large.fit(sineX, sineY);
        expect(large.getSupportVectors().length).toBeGreaterThan(small.getSupportVectors().length);
    });

    test('the optimizer chooses a sensible tube width', () => {
        // small nu -> few points allowed outside the tube -> wide tube;
        // large nu -> narrow tube
        const small = new NuSVR({ kernel: 'rbf', C: 10, nu: 0.1 });
        small.fit(sineX, sineY);
        const large = new NuSVR({ kernel: 'rbf', C: 10, nu: 0.9 });
        large.fit(sineX, sineY);
        expect(small.getFittedEpsilon()).toBeGreaterThan(0);
        expect(large.getFittedEpsilon()).toBeGreaterThanOrEqual(0);
        expect(large.getFittedEpsilon()).toBeLessThan(small.getFittedEpsilon());
    });

    test('invalid nu throws, predict before fit throws', () => {
        expect(() => new NuSVR({ nu: 0 })).toThrow();
        expect(() => new NuSVR({ nu: 1.1 })).toThrow();
        expect(() => new NuSVR({ nu: -0.5 })).toThrow();
        expect(() => new NuSVR().predict([[0]])).toThrow();
    });
});
