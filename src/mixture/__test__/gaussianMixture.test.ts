import { GaussianMixture } from '../gaussianMixture';
import { CovarianceType } from '../common';

// ---------------------------------------------------------------------------
// deterministic test data: three well-separated 2-D blobs
// ---------------------------------------------------------------------------

function lcg(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

const BLOB_CENTERS = [
    [0, 0],
    [7, 7],
    [0, 9],
];

function makeBlobs(perBlob: number, spread: number, seed = 42): { X: number[][]; y: number[] } {
    const rand = lcg(seed);
    const X: number[][] = [];
    const y: number[] = [];
    BLOB_CENTERS.forEach((c, label) => {
        for (let i = 0; i < perBlob; i++) {
            X.push(c.map((v) => v + (rand() * 2 - 1) * spread));
            y.push(label);
        }
    });
    return { X, y };
}

/** fraction of samples whose predicted cluster matches the blob, best label permutation implied by majority vote */
function majorityMatchAccuracy(labels: number[], y: number[], nBlobs: number): number {
    const majority: number[] = [];
    for (let blob = 0; blob < nBlobs; blob++) {
        const counts = new Map<number, number>();
        for (let i = 0; i < y.length; i++) {
            if (y[i] !== blob) continue;
            counts.set(labels[i], (counts.get(labels[i]) ?? 0) + 1);
        }
        let bestLabel = -1;
        let bestCount = -1;
        counts.forEach((count, label) => {
            if (count > bestCount) {
                bestCount = count;
                bestLabel = label;
            }
        });
        majority.push(bestLabel);
    }
    // the mapping must be a bijection for the match to count as recovering the blobs
    expect(new Set(majority).size).toBe(nBlobs);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
        if (labels[i] === majority[y[i]]) correct++;
    }
    return correct / y.length;
}

describe('GaussianMixture on three well-separated blobs', () => {
    const { X, y } = makeBlobs(40, 0.8);

    it('recovers the blob means within tolerance', () => {
        const gm = new GaussianMixture({ nComponents: 3, randomState: 1 });
        gm.fit(X);
        const means = gm.getMeans()!;
        const used = new Set<number>();
        for (const center of BLOB_CENTERS) {
            let bestIdx = -1;
            let bestDist = Infinity;
            for (let c = 0; c < means.length; c++) {
                const dist = Math.hypot(means[c][0] - center[0], means[c][1] - center[1]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = c;
                }
            }
            expect(bestDist).toBeLessThan(0.4);
            used.add(bestIdx);
        }
        expect(used.size).toBe(3); // each blob matched a distinct component
        expect(gm.converged).toBe(true);
        expect(gm.nIter).toBeGreaterThan(0);
        expect(gm.nIter).toBeLessThanOrEqual(100);
        const weights = gm.getWeights()!;
        expect(weights.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 10);
    });

    it('labels match blob membership up to permutation', () => {
        const gm = new GaussianMixture({ nComponents: 3, randomState: 1 });
        const labels = gm.fitPredict(X);
        expect(labels).toHaveLength(X.length);
        expect(majorityMatchAccuracy(labels, y, 3)).toBe(1);
    });

    it('predictProba rows are valid distributions summing to 1', () => {
        const gm = new GaussianMixture({ nComponents: 3, randomState: 1 }).fit(X);
        const proba = gm.predictProba(X);
        expect(proba).toHaveLength(X.length);
        for (const row of proba) {
            expect(row).toHaveLength(3);
            let sum = 0;
            for (const p of row) {
                expect(p).toBeGreaterThanOrEqual(0);
                expect(p).toBeLessThanOrEqual(1 + 1e-12);
                sum += p;
            }
            expect(sum).toBeCloseTo(1, 9);
        }
        // predict is the argmax of predictProba
        const labels = gm.predict(X);
        proba.forEach((row, i) => {
            expect(row.indexOf(Math.max(...row))).toBe(labels[i]);
        });
    });

    it('log-likelihood is non-decreasing across EM iterations', () => {
        const gm = new GaussianMixture({ nComponents: 3, randomState: 5, initParams: 'random', maxIter: 200 }).fit(X);
        const history = gm.getLowerBoundHistory();
        expect(history.length).toBeGreaterThan(1);
        for (let i = 1; i < history.length; i++) {
            expect(history[i]).toBeGreaterThanOrEqual(history[i - 1] - 1e-9);
        }
        expect(history[history.length - 1]).toBe(gm.getLowerBound());
        // score(X) on the training data equals the final average log-likelihood
        // up to the one M-step taken after the last E-step
        expect(gm.score(X)).toBeGreaterThanOrEqual(gm.getLowerBound() - 1e-9);
    });

    const covarianceTypes: CovarianceType[] = ['full', 'tied', 'diag', 'spherical'];
    it.each(covarianceTypes)('covarianceType "%s" fits and predicts sensibly', (covarianceType) => {
        const gm = new GaussianMixture({ nComponents: 3, covarianceType, randomState: 2 });
        const labels = gm.fitPredict(X);
        expect(majorityMatchAccuracy(labels, y, 3)).toBeGreaterThanOrEqual(0.95);
        const proba = gm.predictProba(X);
        for (const row of proba) {
            expect(row.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 9);
        }
        expect(Number.isFinite(gm.score(X))).toBe(true);
    });

    it('seeded runs are fully deterministic', () => {
        const a = new GaussianMixture({ nComponents: 3, randomState: 7, nInit: 3 });
        const b = new GaussianMixture({ nComponents: 3, randomState: 7, nInit: 3 });
        expect(a.fitPredict(X)).toEqual(b.fitPredict(X));
        expect(a.getMeans()).toEqual(b.getMeans());
        expect(a.getWeights()).toEqual(b.getWeights());
        expect(a.getCovariances()).toEqual(b.getCovariances());
        expect(a.getLowerBound()).toBe(b.getLowerBound());
    });

    it('honors weightsInit/meansInit/precisionsInit', () => {
        const meansInit = BLOB_CENTERS.map((c) => c.slice());
        const weightsInit = [1 / 3, 1 / 3, 1 / 3];
        const precisionsInit = [
            [
                [1, 0],
                [0, 1],
            ],
            [
                [1, 0],
                [0, 1],
            ],
            [
                [1, 0],
                [0, 1],
            ],
        ];
        const gm = new GaussianMixture({ nComponents: 3, randomState: 3, weightsInit, meansInit, precisionsInit });
        const labels = gm.fitPredict(X);
        // component order is pinned by meansInit: component i owns blob i
        const means = gm.getMeans()!;
        BLOB_CENTERS.forEach((center, i) => {
            expect(Math.hypot(means[i][0] - center[0], means[i][1] - center[1])).toBeLessThan(0.4);
        });
        for (let i = 0; i < y.length; i++) {
            expect(labels[i]).toBe(y[i]);
        }
        // params round-trip keeps the init arrays
        expect(gm.getParams()).toMatchObject({ weightsInit, meansInit, precisionsInit });
    });
});

describe('GaussianMixture information criteria', () => {
    // aic = -2·logL·n + 2·nParams and bic = -2·logL·n + ln(n)·nParams, where
    // logL is the average per-sample log-likelihood and nParams counts free
    // parameters exactly like sklearn: covParams + k·d + (k − 1) with
    // covParams = k·d(d+1)/2 (full) | d(d+1)/2 (tied) | k·d (diag) | k (spherical).
    const tiny = [
        [0, 0.2],
        [0.3, -0.1],
        [-0.2, 0.4],
        [0.1, 0.1],
        [4.9, 5.2],
        [5.3, 4.8],
        [5.1, 5.0],
        [4.7, 4.9],
    ];

    it('aic/bic match a fully hand-computed single-Gaussian fit', () => {
        const regCovar = 1e-6;
        const gm = new GaussianMixture({ nComponents: 1, covarianceType: 'full', regCovar, randomState: 0 }).fit(tiny);
        const n = tiny.length;
        // hand-computed MLE of a single Gaussian: sample mean and biased covariance (+ regCovar on the diagonal)
        const mean = [0, 1].map((j) => tiny.reduce((s, row) => s + row[j], 0) / n);
        const cov = [
            [regCovar, 0],
            [0, regCovar],
        ];
        for (const row of tiny) {
            const d0 = row[0] - mean[0];
            const d1 = row[1] - mean[1];
            cov[0][0] += (d0 * d0) / n;
            cov[0][1] += (d0 * d1) / n;
            cov[1][1] += (d1 * d1) / n;
        }
        cov[1][0] = cov[0][1];
        const det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
        const inv = [
            [cov[1][1] / det, -cov[0][1] / det],
            [-cov[1][0] / det, cov[0][0] / det],
        ];
        let totalLogL = 0;
        for (const row of tiny) {
            const d0 = row[0] - mean[0];
            const d1 = row[1] - mean[1];
            const quad = d0 * (inv[0][0] * d0 + inv[0][1] * d1) + d1 * (inv[1][0] * d0 + inv[1][1] * d1);
            totalLogL += -0.5 * (2 * Math.log(2 * Math.PI) + Math.log(det) + quad);
        }
        // k = 1, d = 2, 'full' → nParams = d(d+1)/2 + d + 0 = 5
        const nParams = 5;
        expect(gm.score(tiny)).toBeCloseTo(totalLogL / n, 8);
        expect(gm.aic(tiny)).toBeCloseTo(-2 * totalLogL + 2 * nParams, 6);
        expect(gm.bic(tiny)).toBeCloseTo(-2 * totalLogL + Math.log(n) * nParams, 6);
    });

    it.each([
        ['full' as CovarianceType, 3 * 3 + 3 * 2 + 2], // k·d(d+1)/2 + k·d + (k−1) = 9 + 6 + 2
        ['tied' as CovarianceType, 3 + 3 * 2 + 2], //     d(d+1)/2 + k·d + (k−1) = 3 + 6 + 2
        ['diag' as CovarianceType, 3 * 2 + 3 * 2 + 2], // k·d      + k·d + (k−1) = 6 + 6 + 2
        ['spherical' as CovarianceType, 3 + 3 * 2 + 2], // k       + k·d + (k−1) = 3 + 6 + 2
    ])('parameter count for covarianceType "%s" equals %i', (covarianceType, expectedParams) => {
        const { X } = makeBlobs(20, 0.8);
        const gm = new GaussianMixture({ nComponents: 3, covarianceType, randomState: 4 }).fit(X);
        const n = X.length;
        // nParams = (aic + 2·logL·n) / 2, so this pins the exact parameter count
        const inferredParams = (gm.aic(X) + 2 * gm.score(X) * n) / 2;
        expect(inferredParams).toBeCloseTo(expectedParams, 6);
        // bic uses the same count with a ln(n) penalty
        expect(gm.bic(X) - gm.aic(X)).toBeCloseTo((Math.log(n) - 2) * expectedParams, 6);
    });
});

describe('GaussianMixture validation and edge cases', () => {
    it('rejects invalid construction parameters', () => {
        expect(() => new GaussianMixture({ nComponents: 0 })).toThrow(/nComponents/);
        expect(() => new GaussianMixture({ covarianceType: 'banana' as CovarianceType })).toThrow(/covarianceType/);
        expect(() => new GaussianMixture({ maxIter: 0 })).toThrow(/maxIter/);
        expect(() => new GaussianMixture({ nInit: 0 })).toThrow(/nInit/);
        expect(() => new GaussianMixture({ initParams: 'nope' as 'kmeans' })).toThrow(/initParams/);
        expect(() => new GaussianMixture({ nComponents: 2, weightsInit: [1] })).toThrow(/weightsInit/);
    });

    it('throws when predicting before fit and when n_samples < nComponents', () => {
        const gm = new GaussianMixture({ nComponents: 3 });
        expect(() => gm.predict([[0, 0]])).toThrow(/not fitted/);
        expect(() => gm.fit([[0, 0], [1, 1]])).toThrow(/n_samples/);
    });

    it('nInit picks the best of several runs (lower bound never worse than a single run)', () => {
        const { X } = makeBlobs(15, 0.8);
        const single = new GaussianMixture({ nComponents: 3, randomState: 11, nInit: 1, initParams: 'random' }).fit(X);
        const multi = new GaussianMixture({ nComponents: 3, randomState: 11, nInit: 5, initParams: 'random' }).fit(X);
        expect(multi.getLowerBound()).toBeGreaterThanOrEqual(single.getLowerBound() - 1e-12);
    });
});
