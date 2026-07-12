import { BayesianGaussianMixture } from '../bayesianGaussianMixture';

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

/** each blob must map to its own majority component and the mapping must be a bijection */
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
    expect(new Set(majority).size).toBe(nBlobs);
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
        if (labels[i] === majority[y[i]]) correct++;
    }
    return correct / y.length;
}

describe('BayesianGaussianMixture with a Dirichlet-process prior', () => {
    const { X, y } = makeBlobs(50, 0.8);

    it('drives extra components toward zero weight on 3 blobs with nComponents=6', () => {
        const bgm = new BayesianGaussianMixture({ nComponents: 6, randomState: 3, maxIter: 300, tol: 1e-6 });
        bgm.fit(X);
        const weights = bgm.getWeights()!;
        expect(weights).toHaveLength(6);
        expect(weights.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 8);
        const sorted = weights.slice().sort((a, b) => b - a);
        // the three real blobs dominate ...
        expect(sorted[0] + sorted[1] + sorted[2]).toBeGreaterThan(0.9);
        // ... and every surplus component individually decays to (near) nothing
        for (let c = 3; c < 6; c++) {
            expect(sorted[c]).toBeLessThan(0.04);
        }
    });

    it('predictions still recover the 3 groups', () => {
        const bgm = new BayesianGaussianMixture({ nComponents: 6, randomState: 3, maxIter: 300, tol: 1e-6 });
        const labels = bgm.fitPredict(X);
        expect(labels).toHaveLength(X.length);
        expect(majorityMatchAccuracy(labels, y, 3)).toBeGreaterThanOrEqual(0.98);
        // effectively only 3 components receive samples
        expect(new Set(labels).size).toBe(3);
    });

    it('variational lower bound is non-decreasing', () => {
        const bgm = new BayesianGaussianMixture({
            nComponents: 6,
            randomState: 9,
            initParams: 'random',
            maxIter: 300,
            tol: 1e-8,
        }).fit(X);
        const history = bgm.getLowerBoundHistory();
        expect(history.length).toBeGreaterThan(2);
        for (let i = 1; i < history.length; i++) {
            expect(history[i]).toBeGreaterThanOrEqual(history[i - 1] - 1e-6 * Math.max(1, Math.abs(history[i - 1])));
        }
        expect(history[history.length - 1]).toBe(bgm.getLowerBound());
    });

    it('seeded runs are fully deterministic', () => {
        const a = new BayesianGaussianMixture({ nComponents: 6, randomState: 5, nInit: 2 });
        const b = new BayesianGaussianMixture({ nComponents: 6, randomState: 5, nInit: 2 });
        expect(a.fitPredict(X)).toEqual(b.fitPredict(X));
        expect(a.getWeights()).toEqual(b.getWeights());
        expect(a.getMeans()).toEqual(b.getMeans());
        expect(a.getCovariances()).toEqual(b.getCovariances());
        expect(a.getLowerBound()).toBe(b.getLowerBound());
    });

    it('predictProba rows sum to 1 and agree with predict', () => {
        const bgm = new BayesianGaussianMixture({ nComponents: 6, randomState: 3 }).fit(X);
        const proba = bgm.predictProba(X);
        const labels = bgm.predict(X);
        proba.forEach((row, i) => {
            expect(row.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 9);
            expect(row.indexOf(Math.max(...row))).toBe(labels[i]);
        });
        const scores = bgm.scoreSamples(X);
        expect(scores).toHaveLength(X.length);
        scores.forEach((s) => expect(Number.isFinite(s)).toBe(true));
        expect(bgm.score(X)).toBeCloseTo(scores.reduce((s, v) => s + v, 0) / scores.length, 12);
    });
});

describe('BayesianGaussianMixture variants', () => {
    const { X, y } = makeBlobs(40, 0.8, 7);

    it("covarianceType 'diag' fits and recovers the blobs", () => {
        const bgm = new BayesianGaussianMixture({ nComponents: 6, covarianceType: 'diag', randomState: 1, maxIter: 300 });
        const labels = bgm.fitPredict(X);
        expect(majorityMatchAccuracy(labels, y, 3)).toBeGreaterThanOrEqual(0.95);
        const weights = bgm.getWeights()!;
        const sorted = weights.slice().sort((a, b) => b - a);
        expect(sorted[0] + sorted[1] + sorted[2]).toBeGreaterThan(0.85);
    });

    it("weightConcentrationPriorType 'dirichletDistribution' fits the blobs", () => {
        const bgm = new BayesianGaussianMixture({
            nComponents: 3,
            weightConcentrationPriorType: 'dirichletDistribution',
            randomState: 2,
            maxIter: 300,
        });
        const labels = bgm.fitPredict(X);
        expect(majorityMatchAccuracy(labels, y, 3)).toBeGreaterThanOrEqual(0.98);
        expect(bgm.getWeights()!.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 8);
        expect(bgm.converged).toBe(true);
    });

    it('honors explicit priors', () => {
        const bgm = new BayesianGaussianMixture({
            nComponents: 3,
            randomState: 2,
            weightConcentrationPrior: 0.5,
            meanPrecisionPrior: 2,
            meanPrior: [2, 5],
            degreesOfFreedomPrior: 4,
            covariancePrior: [
                [1, 0],
                [0, 1],
            ],
        });
        const labels = bgm.fitPredict(X);
        expect(majorityMatchAccuracy(labels, y, 3)).toBeGreaterThanOrEqual(0.95);
        expect(bgm.getParams()).toMatchObject({
            weightConcentrationPrior: 0.5,
            meanPrecisionPrior: 2,
            meanPrior: [2, 5],
            degreesOfFreedomPrior: 4,
        });
    });
});

describe('BayesianGaussianMixture validation', () => {
    it("throws a clear not-implemented error for 'tied' and 'spherical'", () => {
        for (const covarianceType of ['tied', 'spherical']) {
            expect(() => new BayesianGaussianMixture({ covarianceType: covarianceType as 'full' })).toThrow(
                /not implemented.*full, diag/s
            );
        }
    });

    it('rejects invalid parameters', () => {
        expect(() => new BayesianGaussianMixture({ nComponents: 0 })).toThrow(/nComponents/);
        expect(() => new BayesianGaussianMixture({ weightConcentrationPrior: 0 })).toThrow(/weightConcentrationPrior/);
        expect(() => new BayesianGaussianMixture({ meanPrecisionPrior: -1 })).toThrow(/meanPrecisionPrior/);
        expect(
            () => new BayesianGaussianMixture({ weightConcentrationPriorType: 'nope' as 'dirichletProcess' })
        ).toThrow(/weightConcentrationPriorType/);
        const bad = new BayesianGaussianMixture({ nComponents: 2, degreesOfFreedomPrior: 0.5 });
        expect(() => bad.fit([[0, 0], [1, 1], [2, 2]])).toThrow(/degreesOfFreedomPrior/);
        expect(() => new BayesianGaussianMixture().predict([[0, 0]])).toThrow(/not fitted/);
    });
});
