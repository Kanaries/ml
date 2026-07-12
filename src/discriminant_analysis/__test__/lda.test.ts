import { LinearDiscriminantAnalysis } from '../lda';
import { blobsDataset } from '../../__test__/conformance/datasets';

const SOLVERS = ['svd', 'eigen'] as const;

function accuracy(pred: number[], truth: number[]): number {
    let ok = 0;
    for (let i = 0; i < truth.length; i++) if (pred[i] === truth[i]) ok++;
    return ok / truth.length;
}

/**
 * Hand-verified 2-class 2-D dataset with equal, diagonal within-class
 * covariance. Class means (1, 0) and (5, 0); per-class centered points
 * (+-1, 0), (0, +-1) give pooled scatter diag(4, 4).
 *
 * svd solver (bias-corrected pooled covariance, denominator n - K = 6):
 *   Sw = diag(2/3, 2/3), w = Sw^-1 (mu1 - mu0) = (6, 0),
 *   b = -w . midpoint = -18 (equal priors).
 * eigen solver (prior-weighted biased covariance, denominator n_k = 4):
 *   Sw = diag(1/2, 1/2), w = (8, 0), b = -0.5 (mu1.c1 - mu0.c0) = -24.
 * Both give the decision boundary x0 = 3.
 */
const HAND_X = [
    [0, 0], [2, 0], [1, 1], [1, -1],
    [4, 0], [6, 0], [5, 1], [5, -1],
];
const HAND_Y = [0, 0, 0, 0, 1, 1, 1, 1];

describe('LinearDiscriminantAnalysis: hand-verified 2-class case', () => {
    it("svd solver reproduces the closed-form coef/intercept w = Sw^-1 (mu1 - mu0)", () => {
        const lda = new LinearDiscriminantAnalysis({ solver: 'svd' });
        lda.fit(HAND_X, HAND_Y);
        const coef = lda.getCoef();
        expect(coef.length).toBe(1);
        expect(coef[0][0]).toBeCloseTo(6, 8);
        expect(coef[0][1]).toBeCloseTo(0, 8);
        expect(lda.getIntercept()[0]).toBeCloseTo(-18, 8);
    });

    it('eigen solver reproduces the closed-form coef/intercept', () => {
        const lda = new LinearDiscriminantAnalysis({ solver: 'eigen' });
        lda.fit(HAND_X, HAND_Y);
        const coef = lda.getCoef();
        expect(coef[0][0]).toBeCloseTo(8, 8);
        expect(coef[0][1]).toBeCloseTo(0, 8);
        expect(lda.getIntercept()[0]).toBeCloseTo(-24, 8);
    });

    SOLVERS.forEach((solver) => {
        it(`${solver}: decision boundary sits at x0 = 3 (midpoint, equal priors)`, () => {
            const lda = new LinearDiscriminantAnalysis({ solver });
            lda.fit(HAND_X, HAND_Y);
            const dec = lda.decisionFunction([[3, 0]]) as number[];
            expect(Math.abs(dec[0])).toBeLessThan(1e-8);
            expect(lda.predict([[2.9, 0], [3.1, 5]])).toEqual([0, 1]);
            expect(lda.predict(HAND_X)).toEqual(HAND_Y);
        });
    });

    it('binary transform projects onto a single discriminant axis', () => {
        const lda = new LinearDiscriminantAnalysis({ solver: 'svd' });
        lda.fit(HAND_X, HAND_Y);
        const T = lda.transform(HAND_X);
        expect(T.length).toBe(HAND_X.length);
        expect(T[0].length).toBe(1);
        expect(lda.getExplainedVarianceRatio()).toEqual([1]);
    });
});

describe('LinearDiscriminantAnalysis: sklearn docs example', () => {
    const X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]];
    const y = [1, 1, 1, 2, 2, 2];
    SOLVERS.forEach((solver) => {
        it(`${solver}: predict([[-0.8, -1]]) == [1]`, () => {
            const lda = new LinearDiscriminantAnalysis({ solver });
            lda.fit(X, y);
            expect(lda.predict([[-0.8, -1]])).toEqual([1]);
            expect(lda.predict(X)).toEqual(y);
        });
    });
});

describe('LinearDiscriminantAnalysis: three blobs', () => {
    const { X, y } = blobsDataset();

    SOLVERS.forEach((solver) => {
        it(`${solver}: accuracy > 0.95`, () => {
            const lda = new LinearDiscriminantAnalysis({ solver });
            lda.fit(X, y);
            expect(accuracy(lda.predict(X), y)).toBeGreaterThan(0.95);
        });

        it(`${solver}: predictProba rows sum to 1 and argmax matches predict`, () => {
            const lda = new LinearDiscriminantAnalysis({ solver });
            lda.fit(X, y);
            const proba = lda.predictProba(X);
            const pred = lda.predict(X);
            const classes = lda.getClasses();
            proba.forEach((row, i) => {
                expect(row.length).toBe(3);
                const sum = row.reduce((a, b) => a + b, 0);
                expect(sum).toBeCloseTo(1, 10);
                row.forEach((v) => {
                    expect(v).toBeGreaterThanOrEqual(0);
                    expect(v).toBeLessThanOrEqual(1);
                });
                let best = 0;
                row.forEach((v, k) => { if (v > row[best]) best = k; });
                expect(classes[best]).toBe(pred[i]);
            });
        });

        it(`${solver}: transform separates the classes`, () => {
            const lda = new LinearDiscriminantAnalysis({ solver });
            lda.fit(X, y);
            const T = lda.transform(X);
            expect(T.length).toBe(X.length);
            // maxComponents = min(nClasses - 1, nFeatures) = 2
            expect(T[0].length).toBe(2);
            T.forEach((row) => row.forEach((v) => expect(Number.isFinite(v)).toBe(true)));

            // class centroids in the transformed space
            const centroids: number[][] = [0, 1, 2].map((c) => {
                const rows = T.filter((_, i) => y[i] === c);
                return [0, 1].map((j) => rows.reduce((a, r) => a + r[j], 0) / rows.length);
            });
            let minBetween = Infinity;
            for (let a = 0; a < 3; a++) {
                for (let b = a + 1; b < 3; b++) {
                    const d = Math.hypot(centroids[a][0] - centroids[b][0], centroids[a][1] - centroids[b][1]);
                    minBetween = Math.min(minBetween, d);
                }
            }
            let within = 0;
            T.forEach((row, i) => {
                const c = centroids[y[i]];
                within += Math.hypot(row[0] - c[0], row[1] - c[1]);
            });
            within /= T.length;
            expect(minBetween).toBeGreaterThan(within);
        });

        it(`${solver}: explainedVarianceRatio has maxComponents entries summing to ~1 and descending`, () => {
            const lda = new LinearDiscriminantAnalysis({ solver });
            lda.fit(X, y);
            const evr = lda.getExplainedVarianceRatio();
            expect(evr.length).toBe(2);
            expect(evr[0]).toBeGreaterThanOrEqual(evr[1]);
            expect(evr.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
        });
    });

    it('svd and eigen solvers agree on predictions', () => {
        const svd = new LinearDiscriminantAnalysis({ solver: 'svd' });
        const eigen = new LinearDiscriminantAnalysis({ solver: 'eigen' });
        svd.fit(X, y);
        eigen.fit(X, y);
        expect(eigen.predict(X)).toEqual(svd.predict(X));
    });

    it('nComponents narrows the transform output', () => {
        const lda = new LinearDiscriminantAnalysis({ solver: 'svd', nComponents: 1 });
        lda.fit(X, y);
        expect(lda.transform(X)[0].length).toBe(1);
    });

    it('nComponents > min(nFeatures, nClasses - 1) throws', () => {
        const lda = new LinearDiscriminantAnalysis({ solver: 'svd', nComponents: 3 });
        expect(() => lda.fit(X, y)).toThrow(/nComponents/);
    });
});

describe('LinearDiscriminantAnalysis: shrinkage', () => {
    const { X, y } = blobsDataset();

    it('eigen + shrinkage fits and stays accurate', () => {
        const lda = new LinearDiscriminantAnalysis({ solver: 'eigen', shrinkage: 0.2 });
        lda.fit(X, y);
        expect(accuracy(lda.predict(X), y)).toBeGreaterThan(0.95);
    });

    it("svd + shrinkage throws like sklearn", () => {
        const lda = new LinearDiscriminantAnalysis({ solver: 'svd', shrinkage: 0.2 });
        expect(() => lda.fit(X, y)).toThrow(/shrinkage not supported/);
    });

    it('shrinkage outside [0, 1] throws', () => {
        const lda = new LinearDiscriminantAnalysis({ solver: 'eigen', shrinkage: 1.5 });
        expect(() => lda.fit(X, y)).toThrow(/between 0 and 1/);
    });
});

describe('LinearDiscriminantAnalysis: degenerate inputs', () => {
    it('rank-deficient input (duplicate + constant feature) does not produce NaN (svd)', () => {
        const Xdup = HAND_X.map((row) => [row[0], row[1], row[0], 7]);
        const lda = new LinearDiscriminantAnalysis({ solver: 'svd' });
        lda.fit(Xdup, HAND_Y);
        const scores = lda.predictProba(Xdup);
        scores.forEach((row) => row.forEach((v) => expect(Number.isFinite(v)).toBe(true)));
        expect(lda.predict(Xdup)).toEqual(HAND_Y);
        lda.transform(Xdup).forEach((row) => row.forEach((v) => expect(Number.isFinite(v)).toBe(true)));
    });

    it('eigen solver reports a singular within-class covariance instead of NaN', () => {
        const Xdup = HAND_X.map((row) => [row[0], row[1], row[0]]);
        const lda = new LinearDiscriminantAnalysis({ solver: 'eigen' });
        expect(() => lda.fit(Xdup, HAND_Y)).toThrow(/singular/);
        // ... and shrinkage rescues it
        const shrunk = new LinearDiscriminantAnalysis({ solver: 'eigen', shrinkage: 0.1 });
        shrunk.fit(Xdup, HAND_Y);
        expect(shrunk.predict(Xdup)).toEqual(HAND_Y);
    });

    it('nSamples <= nClasses throws for the svd solver', () => {
        const lda = new LinearDiscriminantAnalysis({ solver: 'svd' });
        expect(() => lda.fit([[0, 1], [2, 3]], [0, 1])).toThrow(/nSamples > nClasses/);
    });

    it('single-class y throws', () => {
        const lda = new LinearDiscriminantAnalysis();
        expect(() => lda.fit([[0, 1], [2, 3]], [1, 1])).toThrow(/at least 2 classes/);
    });

    it('predict before fit throws', () => {
        const lda = new LinearDiscriminantAnalysis();
        expect(() => lda.predict([[0, 0]])).toThrow(/fitted/);
    });
});

describe('LinearDiscriminantAnalysis: priors', () => {
    it('default priors are class frequencies', () => {
        const lda = new LinearDiscriminantAnalysis();
        lda.fit(HAND_X.concat([[1, 2], [1, -2]]), HAND_Y.concat([0, 0]));
        expect(lda.getPriors()[0]).toBeCloseTo(0.6, 12);
        expect(lda.getPriors()[1]).toBeCloseTo(0.4, 12);
    });

    it('explicit priors shift the decision boundary toward the rarer class', () => {
        const even = new LinearDiscriminantAnalysis({ priors: [0.5, 0.5] });
        const skewed = new LinearDiscriminantAnalysis({ priors: [0.999, 0.001] });
        even.fit(HAND_X, HAND_Y);
        skewed.fit(HAND_X, HAND_Y);
        // the midpoint sample flips to the heavily favored class 0
        expect(even.predict([[3.1, 0]])).toEqual([1]);
        expect(skewed.predict([[3.1, 0]])).toEqual([0]);
    });

    it('invalid priors throw', () => {
        expect(() => new LinearDiscriminantAnalysis({ priors: [1] }).fit(HAND_X, HAND_Y)).toThrow(/one entry per class/);
        expect(() => new LinearDiscriminantAnalysis({ priors: [-1, 2] }).fit(HAND_X, HAND_Y)).toThrow(/positive/);
    });
});
