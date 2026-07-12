import { CalibratedClassifierCV, SigmoidCalibration } from '../calibratedClassifierCV';
import { KNearestNeighbors } from '../../neighbors';
import { logLoss } from '../../metrics';
import { loadModel } from '../../base/estimator';
import { GaussianNBProba, MarginClassifier } from './testEstimators';

function lcg(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function gaussian(rand: () => number): number {
    // Box-Muller
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Overlapping binary problem with four nearly-duplicated features: naive
 * Bayes' independence assumption is badly violated, so its probabilities are
 * strongly overconfident — the classic case calibration fixes.
 */
function overconfidentData(seed: number, n: number): { X: number[][]; y: number[] } {
    const rand = lcg(seed);
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < n; i++) {
        const label = i % 2;
        const base = (label === 0 ? -1 : 1) + 1.5 * gaussian(rand);
        X.push([
            base + 0.01 * gaussian(rand),
            base + 0.01 * gaussian(rand),
            base + 0.01 * gaussian(rand),
            base + 0.01 * gaussian(rand),
        ]);
        y.push(label);
    }
    return { X, y };
}

/** Three overlapping 2-D gaussian blobs. */
function multiclassData(seed: number, perClass: number): { X: number[][]; y: number[] } {
    const rand = lcg(seed);
    const centers = [
        [0, 0],
        [2, 2],
        [0, 3],
    ];
    const X: number[][] = [];
    const y: number[] = [];
    centers.forEach((c, label) => {
        for (let i = 0; i < perClass; i++) {
            X.push([c[0] + 1.2 * gaussian(rand), c[1] + 1.2 * gaussian(rand)]);
            y.push(label);
        }
    });
    return { X, y };
}

const train = overconfidentData(7, 400);
const test = overconfidentData(99, 400);

function rawTestLogLoss(): number {
    const nb = new GaussianNBProba();
    nb.fit(train.X, train.y);
    return logLoss(test.y, nb.predictProba(test.X));
}

describe('CalibratedClassifierCV', () => {
    const rawLoss = rawTestLogLoss();

    it.each([
        ['sigmoid', true],
        ['sigmoid', false],
        ['isotonic', true],
        ['isotonic', false],
    ] as ['sigmoid' | 'isotonic', boolean][])(
        'method=%s ensemble=%s reduces held-out logLoss vs the raw overconfident classifier',
        (method, ensemble) => {
            const cal = new CalibratedClassifierCV({ estimator: new GaussianNBProba(), method, ensemble, cv: 5 });
            cal.fit(train.X, train.y);
            const proba = cal.predictProba(test.X);
            for (const row of proba) {
                expect(row).toHaveLength(2);
                expect(row[0] + row[1]).toBeCloseTo(1, 9);
                expect(row[0]).toBeGreaterThanOrEqual(0);
                expect(row[1]).toBeGreaterThanOrEqual(0);
            }
            const calibratedLoss = logLoss(test.y, proba);
            expect(calibratedLoss).toBeLessThan(rawLoss);
        },
    );

    it('predict is the argmax of predictProba over the sorted classes', () => {
        const cal = new CalibratedClassifierCV({ estimator: new GaussianNBProba(), cv: 5 });
        cal.fit(train.X, train.y);
        const proba = cal.predictProba(test.X);
        const pred = cal.predict(test.X);
        pred.forEach((label, i) => {
            expect(label).toBe(proba[i][1] > proba[i][0] ? 1 : 0);
        });
        // sanity: the calibrated model still classifies well above chance
        expect(cal.score(test.X, test.y)).toBeGreaterThan(0.7);
    });

    it('handles multiclass problems one-vs-rest with normalized rows', () => {
        const data = multiclassData(3, 60);
        for (const method of ['sigmoid', 'isotonic'] as const) {
            const cal = new CalibratedClassifierCV({ estimator: new GaussianNBProba(), method, cv: 3 });
            cal.fit(data.X, data.y);
            const proba = cal.predictProba(data.X);
            for (const row of proba) {
                expect(row).toHaveLength(3);
                expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
            }
            expect(cal.score(data.X, data.y)).toBeGreaterThan(0.6);
        }
    });

    it('falls back to decisionFunction when the base has no predictProba', () => {
        const cal = new CalibratedClassifierCV({ estimator: new MarginClassifier(), cv: 4 });
        cal.fit(train.X, train.y);
        const proba = cal.predictProba(test.X);
        for (const row of proba) {
            expect(row[0] + row[1]).toBeCloseTo(1, 9);
        }
        // Platt-scaled margins must be monotone in the margin itself
        const low = cal.predictProba([[-3, 0, 0, 0]])[0][1];
        const high = cal.predictProba([[3, 0, 0, 0]])[0][1];
        expect(high).toBeGreaterThan(low);
        expect(logLoss(test.y, proba)).toBeLessThan(rawTestLogLoss());
    });

    it('rejects a base classifier with neither predictProba nor decisionFunction', () => {
        // KNN exposes neither predictProba nor decisionFunction
        const cal = new CalibratedClassifierCV({ estimator: new KNearestNeighbors({ kNeighbors: 3 }), cv: 3 });
        expect(() => cal.fit(train.X, train.y)).toThrow(/predictProba or decisionFunction/);
    });

    it('validates constructor params and fit inputs', () => {
        expect(() => new CalibratedClassifierCV({} as never)).toThrow(/estimator/);
        expect(
            () => new CalibratedClassifierCV({ estimator: new GaussianNBProba(), method: 'platt' as never }),
        ).toThrow(/method/);
        expect(() => new CalibratedClassifierCV({ estimator: new GaussianNBProba(), cv: 1 })).toThrow(/cv/);
        const cal = new CalibratedClassifierCV({ estimator: new GaussianNBProba(), cv: 3 });
        expect(() => cal.fit([[1], [2], [3]], [1, 1, 1])).toThrow(/two classes/);
        expect(() => cal.predictProba([[1]])).toThrow(/must be fitted/);
    });

    it.each([
        ['sigmoid', true],
        ['isotonic', false],
    ] as ['sigmoid' | 'isotonic', boolean][])(
        'serialize -> revive -> predictProba parity (method=%s ensemble=%s)',
        (method, ensemble) => {
            const cal = new CalibratedClassifierCV({ estimator: new GaussianNBProba(), method, ensemble, cv: 5 });
            cal.fit(train.X, train.y);
            const revived = loadModel(JSON.stringify(cal)) as CalibratedClassifierCV;
            expect(revived.constructor).toBe(CalibratedClassifierCV);
            expect(revived.predictProba(test.X)).toEqual(cal.predictProba(test.X));
            expect(revived.predict(test.X)).toEqual(cal.predict(test.X));
        },
    );
});

describe('SigmoidCalibration (Platt scaling)', () => {
    it("fits Platt's smoothed targets via Newton's method and stays in (0, 1)", () => {
        // scores separating the labels imperfectly
        const scores = [-2, -1.5, -1, -0.5, 0.2, -0.1, 0.5, 1, 1.5, 2];
        const targets = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        const cal = new SigmoidCalibration();
        cal.fit(scores, targets);
        const p = cal.predict(scores);
        for (const v of p) {
            expect(v).toBeGreaterThan(0);
            expect(v).toBeLessThan(1);
        }
        // monotone increasing in the score => fitted slope a < 0
        // (probability = 1 / (1 + exp(a * score + b)))
        expect(cal.a).toBeLessThan(0);
        const sorted = scores.slice().sort((a, b) => a - b);
        for (let i = 1; i < sorted.length; i++) {
            expect(cal.predict([sorted[i]])[0]).toBeGreaterThan(cal.predict([sorted[i - 1]])[0]);
        }
    });

    it('uses the Bayesian-prior label smoothing targets (bounded away from 0/1 on separable data)', () => {
        // perfectly separable: without smoothing, the MLE diverges
        const scores = [-2, -1, 1, 2];
        const targets = [0, 0, 1, 1];
        const cal = new SigmoidCalibration();
        cal.fit(scores, targets);
        const p = cal.predict([-2, 2]);
        expect(p[0]).toBeGreaterThan(0);
        expect(p[1]).toBeLessThan(1);
        expect(Number.isFinite(cal.a)).toBe(true);
        expect(Number.isFinite(cal.b)).toBe(true);
    });
});
