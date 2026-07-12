import { SGDClassifier } from '../sgdClassifier';
import { SGDRegressor } from '../sgdRegressor';
import { Perceptron } from '../perceptron';
import { accuracyScore } from '../../metrics';

// ---------------------------------------------------------------------------
// Deterministic data generators (fixed-seed LCG, platform independent)
// ---------------------------------------------------------------------------

function lcg(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function blob(rand: () => number, center: number[], n: number, spread: number): number[][] {
    const points: number[][] = [];
    for (let i = 0; i < n; i++) {
        points.push(center.map(c => c + (rand() * 2 - 1) * spread));
    }
    return points;
}

/** Two well-separated 2-D blobs, labels 0/1. */
function separableBinary(): { X: number[][]; y: number[] } {
    const rand = lcg(7);
    const X: number[][] = [];
    const y: number[] = [];
    for (const p of blob(rand, [0, 0], 25, 1)) { X.push(p); y.push(0); }
    for (const p of blob(rand, [6, 6], 25, 1)) { X.push(p); y.push(1); }
    return { X, y };
}

/** Three well-separated 2-D blobs, labels 0/1/2. */
function blobs3(): { X: number[][]; y: number[] } {
    const rand = lcg(42);
    const centers = [[0, 0], [7, 7], [0, 9]];
    const X: number[][] = [];
    const y: number[] = [];
    centers.forEach((c, label) => {
        for (const p of blob(rand, c, 20, 1)) { X.push(p); y.push(label); }
    });
    return { X, y };
}

/**
 * Sparse-signal classification data: the label depends only on feature 0,
 * features 1..5 are pure noise.
 */
function sparseSignal(): { X: number[][]; y: number[] } {
    const rand = lcg(99);
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 200; i++) {
        const signal = rand() * 4 - 2;
        const row = [signal];
        for (let j = 0; j < 5; j++) row.push(rand() * 2 - 1);
        X.push(row);
        y.push(signal > 0 ? 1 : 0);
    }
    return { X, y };
}

/** y = 3x - 2 on a single feature, optionally with one gross outlier. */
function line3xMinus2(outlier = false): { X: number[][]; y: number[] } {
    const rand = lcg(23);
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 100; i++) {
        const x = rand() * 4 - 2;
        X.push([x]);
        y.push(3 * x - 2);
    }
    if (outlier) y[0] += 100;
    return { X, y };
}

function l2Norm(v: number[]): number {
    return Math.sqrt(v.reduce((s, w) => s + w * w, 0));
}

// ---------------------------------------------------------------------------
// SGDClassifier
// ---------------------------------------------------------------------------

describe('SGDClassifier', () => {
    it('reaches perfect accuracy on linearly separable data with hinge loss', () => {
        const { X, y } = separableBinary();
        const clf = new SGDClassifier({ loss: 'hinge', randomState: 42 });
        clf.fit(X, y);
        expect(accuracyScore(clf.predict(X), y)).toBe(1);
    });

    it('logLoss predictProba rows sum to 1, lie in [0, 1], and argmax matches predict', () => {
        const { X, y } = separableBinary();
        const clf = new SGDClassifier({ loss: 'logLoss', randomState: 42 });
        clf.fit(X, y);
        const proba = clf.predictProba(X);
        const pred = clf.predict(X);
        const classes = clf.getClasses();
        proba.forEach((row, i) => {
            expect(row).toHaveLength(2);
            const sum = row.reduce((s, p) => s + p, 0);
            expect(sum).toBeCloseTo(1, 9);
            for (const p of row) {
                expect(p).toBeGreaterThanOrEqual(0);
                expect(p).toBeLessThanOrEqual(1);
            }
            const argmax = row[1] > row[0] ? classes[1] : classes[0];
            expect(argmax).toBe(pred[i]);
        });
        // and the calibration is at least directionally right on separated blobs
        proba.forEach((row, i) => {
            expect(row[classes.indexOf(y[i])]).toBeGreaterThan(0.5);
        });
    });

    it('modifiedHuber supports predictProba; hinge / squaredHinge / perceptron throw', () => {
        const { X, y } = separableBinary();
        const mh = new SGDClassifier({ loss: 'modifiedHuber', randomState: 42 });
        mh.fit(X, y);
        const proba = mh.predictProba(X);
        proba.forEach(row => expect(row.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 9));
        for (const loss of ['hinge', 'squaredHinge', 'perceptron'] as const) {
            const clf = new SGDClassifier({ loss, randomState: 42 });
            clf.fit(X, y);
            // margin losses expose NO predictProba capability (sklearn's
            // available_if semantics) so meta-estimators can feature-detect
            expect(clf.predictProba).toBeUndefined();
        }
    });

    it('handles 3 classes one-vs-rest with accuracy > 0.9 and per-class decision values', () => {
        const { X, y } = blobs3();
        const clf = new SGDClassifier({ loss: 'hinge', randomState: 1 });
        clf.fit(X, y);
        expect(accuracyScore(clf.predict(X), y)).toBeGreaterThan(0.9);
        expect(clf.getClasses()).toEqual([0, 1, 2]);
        expect(clf.getCoef()).toHaveLength(3);
        const scores = clf.decisionFunction(X) as number[][];
        expect(scores).toHaveLength(X.length);
        expect(scores[0]).toHaveLength(3);
        // argmax of the decision values is exactly predict
        const pred = clf.predict(X);
        scores.forEach((row, i) => {
            const argmax = row.indexOf(Math.max(...row));
            expect(clf.getClasses()[argmax]).toBe(pred[i]);
        });
    });

    it('multiclass logLoss predictProba is OvR-normalized and argmax matches predict', () => {
        const { X, y } = blobs3();
        const clf = new SGDClassifier({ loss: 'logLoss', randomState: 1 });
        clf.fit(X, y);
        const proba = clf.predictProba(X);
        const pred = clf.predict(X);
        proba.forEach((row, i) => {
            expect(row).toHaveLength(3);
            expect(row.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 9);
            const argmax = row.indexOf(Math.max(...row));
            expect(clf.getClasses()[argmax]).toBe(pred[i]);
        });
        expect(accuracyScore(pred, y)).toBeGreaterThan(0.9);
    });

    it('binary decisionFunction is 1-D and its sign is the prediction', () => {
        const { X, y } = separableBinary();
        const clf = new SGDClassifier({ loss: 'hinge', randomState: 42 });
        clf.fit(X, y);
        const scores = clf.decisionFunction(X) as number[];
        const pred = clf.predict(X);
        expect(typeof scores[0]).toBe('number');
        scores.forEach((s, i) => {
            expect(pred[i]).toBe(s > 0 ? 1 : 0);
        });
    });

    it('L1 penalty drives irrelevant-feature weights to exactly 0 (cumulative penalty)', () => {
        const { X, y } = sparseSignal();
        const clf = new SGDClassifier({ loss: 'hinge', penalty: 'l1', alpha: 0.05, randomState: 3 });
        clf.fit(X, y);
        const w = clf.getCoef()[0];
        // the informative feature survives ...
        expect(Math.abs(w[0])).toBeGreaterThan(0.1);
        // ... while every noise feature is truncated to exactly zero
        for (let j = 1; j < w.length; j++) {
            expect(w[j]).toBe(0);
        }
        // and the model still works
        expect(accuracyScore(clf.predict(X), y)).toBeGreaterThan(0.9);
    });

    it('elasticnet also produces exact zeros on noise features', () => {
        const { X, y } = sparseSignal();
        const clf = new SGDClassifier({ loss: 'hinge', penalty: 'elasticnet', l1Ratio: 0.8, alpha: 0.01, randomState: 3 });
        clf.fit(X, y);
        const w = clf.getCoef()[0];
        const zeroNoise = w.slice(1).filter(v => v === 0).length;
        expect(zeroNoise).toBeGreaterThanOrEqual(4);
    });

    it('stronger L2 regularization shrinks the weight norm', () => {
        const { X, y } = separableBinary();
        const weak = new SGDClassifier({ loss: 'hinge', penalty: 'l2', alpha: 1e-4, randomState: 5 });
        const strong = new SGDClassifier({ loss: 'hinge', penalty: 'l2', alpha: 1e-1, randomState: 5 });
        weak.fit(X, y);
        strong.fit(X, y);
        expect(l2Norm(strong.getCoef()[0])).toBeLessThan(l2Norm(weak.getCoef()[0]));
    });

    it('is deterministic under a fixed randomState', () => {
        const { X, y } = blobs3();
        const a = new SGDClassifier({ loss: 'hinge', randomState: 11 });
        const b = new SGDClassifier({ loss: 'hinge', randomState: 11 });
        a.fit(X, y);
        b.fit(X, y);
        expect(b.getCoef()).toEqual(a.getCoef());
        expect(b.getIntercept()).toEqual(a.getIntercept());
        expect(b.predict(X)).toEqual(a.predict(X));
    });

    it('validates loss, penalty and schedule parameters', () => {
        const { X, y } = separableBinary();
        expect(() => {
            const clf = new SGDClassifier({ loss: 'squaredEpsilon' as never });
            clf.fit(X, y);
        }).toThrow(/Unknown classification loss/);
        expect(() => new SGDClassifier({ penalty: 'ridge' as never })).toThrow(/Unknown penalty/);
        expect(() => {
            const clf = new SGDClassifier({ learningRate: 'constant' }); // eta0 defaults to 0
            clf.fit(X, y);
        }).toThrow(/eta0 must be > 0/);
        expect(() => {
            const clf = new SGDClassifier({ alpha: 0 }); // optimal schedule needs alpha > 0
            clf.fit(X, y);
        }).toThrow(/alpha must be > 0/);
    });
});

// ---------------------------------------------------------------------------
// SGDRegressor
// ---------------------------------------------------------------------------

describe('SGDRegressor', () => {
    it('fits y = 3x - 2 with R² > 0.99 (invscaling default schedule)', () => {
        const { X, y } = line3xMinus2();
        const reg = new SGDRegressor({ randomState: 42 });
        reg.fit(X, y);
        expect(reg.score(X, y)).toBeGreaterThan(0.99);
        expect(reg.getCoef()[0]).toBeCloseTo(3, 0);
        expect(reg.getIntercept()).toBeCloseTo(-2, 0);
    });

    it('huber loss is robust to a single gross outlier where squaredError is not', () => {
        const { X, y } = line3xMinus2(true);
        const squared = new SGDRegressor({ loss: 'squaredError', randomState: 42 });
        const huber = new SGDRegressor({ loss: 'huber', epsilon: 1, eta0: 0.05, maxIter: 2000, randomState: 42 });
        squared.fit(X, y);
        huber.fit(X, y);
        const squaredErr = Math.abs(squared.getCoef()[0] - 3) + Math.abs(squared.getIntercept() + 2);
        const huberErr = Math.abs(huber.getCoef()[0] - 3) + Math.abs(huber.getIntercept() + 2);
        expect(huberErr).toBeLessThan(squaredErr);
        expect(huberErr).toBeLessThan(0.5);
    });

    it('epsilonInsensitive and squaredEpsilonInsensitive losses train and predict', () => {
        const { X, y } = line3xMinus2();
        for (const loss of ['epsilonInsensitive', 'squaredEpsilonInsensitive'] as const) {
            const reg = new SGDRegressor({ loss, eta0: 0.05, randomState: 42 });
            reg.fit(X, y);
            const pred = reg.predict(X);
            expect(pred).toHaveLength(X.length);
            pred.forEach(p => expect(Number.isFinite(p)).toBe(true));
            expect(reg.score(X, y)).toBeGreaterThan(0.8);
        }
    });

    it('supports L1 penalty and produces exact zeros on irrelevant features', () => {
        const rand = lcg(17);
        const X: number[][] = [];
        const y: number[] = [];
        for (let i = 0; i < 200; i++) {
            const row = [rand() * 4 - 2, rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1];
            X.push(row);
            y.push(2 * row[0] + 1);
        }
        const reg = new SGDRegressor({ penalty: 'l1', alpha: 0.01, randomState: 8 });
        reg.fit(X, y);
        const w = reg.getCoef();
        expect(Math.abs(w[0])).toBeGreaterThan(1);
        const zeroNoise = w.slice(1).filter(v => v === 0).length;
        expect(zeroNoise).toBeGreaterThanOrEqual(2);
    });

    it('is deterministic under a fixed randomState', () => {
        const { X, y } = line3xMinus2();
        const a = new SGDRegressor({ randomState: 9 });
        const b = new SGDRegressor({ randomState: 9 });
        a.fit(X, y);
        b.fit(X, y);
        expect(b.getCoef()).toEqual(a.getCoef());
        expect(b.getIntercept()).toBe(a.getIntercept());
        expect(b.predict(X)).toEqual(a.predict(X));
    });
});

// ---------------------------------------------------------------------------
// Perceptron
// ---------------------------------------------------------------------------

describe('Perceptron', () => {
    it('reaches perfect accuracy on linearly separable data', () => {
        const { X, y } = separableBinary();
        const clf = new Perceptron({ randomState: 42 });
        clf.fit(X, y);
        expect(accuracyScore(clf.predict(X), y)).toBe(1);
    });

    it('handles 3 classes one-vs-rest', () => {
        const { X, y } = blobs3();
        const clf = new Perceptron({ randomState: 42 });
        clf.fit(X, y);
        expect(accuracyScore(clf.predict(X), y)).toBeGreaterThan(0.9);
    });

    it('respects eta0: doubling it scales the learned weights linearly', () => {
        const { X, y } = separableBinary();
        // perceptron updates fire on sign mistakes only, so with no penalty the
        // mistake sequence is scale-invariant and weights scale exactly with eta0
        const one = new Perceptron({ eta0: 1, tol: null, maxIter: 20, randomState: 13 });
        const two = new Perceptron({ eta0: 2, tol: null, maxIter: 20, randomState: 13 });
        one.fit(X, y);
        two.fit(X, y);
        const w1 = one.getCoef()[0];
        const w2 = two.getCoef()[0];
        expect(l2Norm(w1)).toBeGreaterThan(0);
        w2.forEach((w, j) => expect(w).toBeCloseTo(2 * w1[j], 10));
        expect(two.getIntercept()[0]).toBeCloseTo(2 * one.getIntercept()[0], 10);
    });

    it('has no probability model: predictProba throws', () => {
        const { X, y } = separableBinary();
        const clf = new Perceptron({ randomState: 42 });
        clf.fit(X, y);
        expect(clf.predictProba).toBeUndefined();
    });

    it('exposes exactly the Perceptron parameter set', () => {
        const clf = new Perceptron({ randomState: 42 });
        expect(Object.keys(clf.getParams()).sort()).toEqual([
            'alpha', 'eta0', 'fitIntercept', 'l1Ratio', 'maxIter',
            'penalty', 'randomState', 'shuffle', 'tol',
        ]);
        expect(clf.getParams().penalty).toBeNull();
        expect(clf.getParams().eta0).toBe(1);
    });

    it('is deterministic under a fixed randomState', () => {
        const { X, y } = blobs3();
        const a = new Perceptron({ randomState: 21 });
        const b = new Perceptron({ randomState: 21 });
        a.fit(X, y);
        b.fit(X, y);
        expect(b.getCoef()).toEqual(a.getCoef());
        expect(b.predict(X)).toEqual(a.predict(X));
    });
});
