import { ClassifierBase } from '../base';
import { BaseEstimator, Params, registerEstimator, registerSerializableClass } from '../base/estimator';
import { StratifiedKFold } from '../utils/modelSelection';
import { IsotonicRegression } from '../isotonic';

export type CalibrationMethod = 'sigmoid' | 'isotonic';

/**
 * Any contract classifier exposing a continuous score:
 * `predictProba` (preferred) or `decisionFunction`.
 */
export interface CalibratableClassifier extends BaseEstimator {
    fit(X: number[][], y: number[], sampleWeight?: number[]): void;
    predict(X: number[][]): number[];
    predictProba?(X: number[][]): number[][];
    decisionFunction?(X: number[][]): number[] | number[][];
}

export interface CalibratedClassifierCVProps {
    /** The base classifier to calibrate. It is cloned, never mutated. */
    estimator: CalibratableClassifier;
    /** 'sigmoid' (Platt scaling, default) or 'isotonic'. */
    method?: CalibrationMethod;
    /** Number of StratifiedKFold folds (default 5). */
    cv?: number;
    /**
     * `true` (default, sklearn's default): keep one (classifier, calibrators)
     * pair per fold and average their calibrated probabilities at inference.
     * `false`: pool the out-of-fold scores, fit a single calibrator set on
     * them, and refit the base classifier on the full data (sklearn's
     * `ensemble=False`).
     */
    ensemble?: boolean;
}

/** Common surface of the two calibrator kinds. */
interface CalibratorLike {
    predict(scores: number[]): number[];
}

interface CalibratedPair {
    classifier: CalibratableClassifier;
    /** One calibrator per class (a single one for binary problems). */
    calibrators: CalibratorLike[];
}

/**
 * Platt scaling: probability = 1 / (1 + exp(a * score + b)) with (a, b)
 * fitted by minimizing the cross-entropy against Bayesian-prior-smoothed
 * targets t+ = (n+ + 1)/(n+ + 2), t- = 1/(n- + 2) (Platt 1999), using the
 * Newton method with backtracking line search from Lin, Lin & Weng (2007) —
 * the same algorithm as sklearn's `_sigmoid_calibration` / LIBSVM's
 * `sigmoid_train`.
 */
export class SigmoidCalibration {
    public a = 0;
    public b = 0;

    public fit(scores: number[], targets: number[], sampleWeight?: number[]): void {
        const n = scores.length;
        if (n === 0 || targets.length !== n) {
            throw new Error('scores and targets must be non-empty arrays of the same length');
        }
        const w = sampleWeight ?? new Array<number>(n).fill(1);
        if (w.length !== n) throw new Error('sampleWeight must have the same length as scores');

        let prior0 = 0;
        let prior1 = 0;
        for (let i = 0; i < n; i++) {
            if (targets[i] > 0) prior1 += w[i];
            else prior0 += w[i];
        }
        const hiTarget = (prior1 + 1) / (prior1 + 2);
        const loTarget = 1 / (prior0 + 2);
        const T = targets.map((t) => (t > 0 ? hiTarget : loTarget));

        const maxIter = 100;
        const minStep = 1e-10;
        const sigma = 1e-12; // Hessian ridge for numerical stability
        let A = 0;
        let B = Math.log((prior0 + 1) / (prior1 + 1));

        const objective = (a: number, b: number): number => {
            let f = 0;
            for (let i = 0; i < n; i++) {
                const fApB = scores[i] * a + b;
                f +=
                    fApB >= 0
                        ? w[i] * (T[i] * fApB + Math.log1p(Math.exp(-fApB)))
                        : w[i] * ((T[i] - 1) * fApB + Math.log1p(Math.exp(fApB)));
            }
            return f;
        };
        let fval = objective(A, B);

        for (let it = 0; it < maxIter; it++) {
            // gradient and Hessian of the cross-entropy in (A, B)
            let h11 = sigma;
            let h22 = sigma;
            let h21 = 0;
            let g1 = 0;
            let g2 = 0;
            for (let i = 0; i < n; i++) {
                const fApB = scores[i] * A + B;
                let p: number;
                let q: number;
                if (fApB >= 0) {
                    const e = Math.exp(-fApB);
                    p = e / (1 + e);
                    q = 1 / (1 + e);
                } else {
                    const e = Math.exp(fApB);
                    p = 1 / (1 + e);
                    q = e / (1 + e);
                }
                const d2 = w[i] * p * q;
                h11 += scores[i] * scores[i] * d2;
                h22 += d2;
                h21 += scores[i] * d2;
                const d1 = w[i] * (T[i] - p);
                g1 += scores[i] * d1;
                g2 += d1;
            }
            if (Math.abs(g1) < 1e-5 && Math.abs(g2) < 1e-5) break;
            // Newton direction: -H^{-1} g
            const det = h11 * h22 - h21 * h21;
            const dA = -(h22 * g1 - h21 * g2) / det;
            const dB = -(-h21 * g1 + h11 * g2) / det;
            const gd = g1 * dA + g2 * dB;
            let stepSize = 1;
            while (stepSize >= minStep) {
                const newA = A + stepSize * dA;
                const newB = B + stepSize * dB;
                const newF = objective(newA, newB);
                if (newF < fval + 1e-4 * stepSize * gd) {
                    A = newA;
                    B = newB;
                    fval = newF;
                    break;
                }
                stepSize /= 2;
            }
            if (stepSize < minStep) break; // line search failed
        }
        this.a = A;
        this.b = B;
    }

    public predict(scores: number[]): number[] {
        return scores.map((s) => {
            const fApB = s * this.a + this.b;
            return fApB >= 0 ? Math.exp(-fApB) / (1 + Math.exp(-fApB)) : 1 / (1 + Math.exp(fApB));
        });
    }
}
registerSerializableClass('calibration.SigmoidCalibration', SigmoidCalibration);

/**
 * Probability calibration with out-of-fold cross-validation, mirroring
 * `sklearn.calibration.CalibratedClassifierCV`.
 *
 * `fit` splits (X, y) with `StratifiedKFold(cv)` (no shuffle, deterministic);
 * per fold, a clone of the base classifier is fitted on the train part and
 * scored on the held-out part. Scores are the `predictProba` output when the
 * base classifier provides it (positive-class column for binary problems),
 * else `decisionFunction`. Calibrators are fitted one-vs-rest — one per class,
 * or a single one for binary problems.
 *
 * With `ensemble=true` (the default, matching sklearn) each fold keeps its own
 * (classifier, calibrators) pair and `predictProba` averages the calibrated
 * probabilities of all folds. With `ensemble=false` the out-of-fold scores are
 * pooled into a single calibrator set and the base classifier is refit on the
 * full data (sklearn's pooled variant).
 *
 * `predictProba` returns per-class calibrated scores normalized to sum to 1
 * (binary problems return [1 - p, p] directly); `predict` is the argmax.
 *
 * Omitted vs sklearn: `cv='prefit'` / `FrozenEstimator` support, arbitrary
 * splitter objects for `cv` (only an integer fold count is accepted), and
 * fold-classifiers trained on a subset of the classes (StratifiedKFold
 * guarantees every class appears in every training part, and errors out when
 * a class has fewer than `cv` samples).
 */
export class CalibratedClassifierCV extends ClassifierBase {
    private estimator: CalibratableClassifier;
    private method: CalibrationMethod;
    private cv: number;
    private ensemble: boolean;

    private classes: number[] = [];
    private calibratedPairs: CalibratedPair[] = [];
    private fitted = false;

    constructor(props: CalibratedClassifierCVProps) {
        super();
        const { estimator, method = 'sigmoid', cv = 5, ensemble = true } = props ?? {};
        if (!(estimator instanceof BaseEstimator) || typeof (estimator as CalibratableClassifier).fit !== 'function') {
            throw new Error('estimator must be a contract classifier (a BaseEstimator with fit/predict)');
        }
        if (method !== 'sigmoid' && method !== 'isotonic') {
            throw new Error("method must be 'sigmoid' or 'isotonic'");
        }
        if (!Number.isInteger(cv) || cv < 2) {
            throw new Error('cv must be an integer >= 2');
        }
        this.estimator = estimator;
        this.method = method;
        this.cv = cv;
        this.ensemble = ensemble;
    }

    public getParams(): Params {
        return {
            estimator: this.estimator,
            method: this.method,
            cv: this.cv,
            ensemble: this.ensemble,
        };
    }

    /**
     * Continuous per-class scores for the OvR calibrators, shape
     * [nSamples][nClasses] (or [nSamples][1] for binary problems).
     */
    private rawScores(clf: CalibratableClassifier, X: number[][], nClasses: number): number[][] {
        if (typeof clf.predictProba === 'function') {
            const proba = clf.predictProba(X);
            if (proba.length !== X.length || proba[0].length !== nClasses) {
                throw new Error(
                    `base classifier predictProba returned shape [${proba.length}, ${proba[0]?.length}], ` +
                        `expected [${X.length}, ${nClasses}]`,
                );
            }
            return nClasses === 2 ? proba.map((row) => [row[1]]) : proba;
        }
        if (typeof clf.decisionFunction === 'function') {
            const df = clf.decisionFunction(X);
            if (Array.isArray(df[0])) {
                const mat = df as number[][];
                if (nClasses > 2 && mat[0].length !== nClasses) {
                    throw new Error(
                        `base classifier decisionFunction returned ${mat[0].length} columns, expected ${nClasses}`,
                    );
                }
                return nClasses === 2 && mat[0].length !== 1 ? mat.map((row) => [row[mat[0].length - 1]]) : mat;
            }
            if (nClasses > 2) {
                throw new Error('base classifier decisionFunction is 1-D but the problem has more than two classes');
            }
            return (df as number[]).map((v) => [v]);
        }
        throw new Error('base classifier must implement predictProba or decisionFunction to be calibrated');
    }

    private fitCalibrators(
        scores: number[][],
        y: number[],
        classes: number[],
        sampleWeight?: number[],
    ): CalibratorLike[] {
        const nCalibrators = classes.length === 2 ? 1 : classes.length;
        const calibrators: CalibratorLike[] = [];
        for (let k = 0; k < nCalibrators; k++) {
            const positive = classes.length === 2 ? classes[1] : classes[k];
            const target = y.map((v) => (v === positive ? 1 : 0));
            const column = scores.map((row) => row[classes.length === 2 ? 0 : k]);
            if (this.method === 'sigmoid') {
                const cal = new SigmoidCalibration();
                cal.fit(column, target, sampleWeight);
                calibrators.push(cal);
            } else {
                const cal = new IsotonicRegression({ increasing: true, outOfBounds: 'clip' });
                cal.fit(column, target, sampleWeight);
                calibrators.push(cal);
            }
        }
        return calibrators;
    }

    public fit(X: number[][], y: number[], sampleWeight?: number[]): void {
        if (!Array.isArray(X) || X.length === 0 || X.length !== y.length) {
            throw new Error('X and y must be non-empty arrays of the same length');
        }
        const classes = Array.from(new Set(y)).sort((a, b) => a - b);
        if (classes.length < 2) {
            throw new Error('CalibratedClassifierCV requires at least two classes in y');
        }
        const splitter = new StratifiedKFold({ nSplits: this.cv });
        const folds = splitter.split(X, y);
        const pairs: CalibratedPair[] = [];

        if (this.ensemble) {
            for (const fold of folds) {
                const clf = this.estimator.clone();
                const trainX = fold.trainIndices.map((i) => X[i]);
                const trainY = fold.trainIndices.map((i) => y[i]);
                const trainW = sampleWeight ? fold.trainIndices.map((i) => sampleWeight[i]) : undefined;
                clf.fit(trainX, trainY, trainW);
                const testX = fold.testIndices.map((i) => X[i]);
                const testY = fold.testIndices.map((i) => y[i]);
                const testW = sampleWeight ? fold.testIndices.map((i) => sampleWeight[i]) : undefined;
                const scores = this.rawScores(clf, testX, classes.length);
                pairs.push({ classifier: clf, calibrators: this.fitCalibrators(scores, testY, classes, testW) });
            }
        } else {
            // pooled variant: out-of-fold scores for every sample, one
            // calibrator set, base classifier refit on the full data
            const pooled: number[][] = new Array(X.length);
            for (const fold of folds) {
                const clf = this.estimator.clone();
                const trainX = fold.trainIndices.map((i) => X[i]);
                const trainY = fold.trainIndices.map((i) => y[i]);
                const trainW = sampleWeight ? fold.trainIndices.map((i) => sampleWeight[i]) : undefined;
                clf.fit(trainX, trainY, trainW);
                const testX = fold.testIndices.map((i) => X[i]);
                const scores = this.rawScores(clf, testX, classes.length);
                fold.testIndices.forEach((sampleIdx, row) => {
                    pooled[sampleIdx] = scores[row];
                });
            }
            const calibrators = this.fitCalibrators(pooled, y, classes, sampleWeight);
            const finalClf = this.estimator.clone();
            finalClf.fit(X, y, sampleWeight);
            pairs.push({ classifier: finalClf, calibrators });
        }

        this.classes = classes;
        this.calibratedPairs = pairs;
        this.fitted = true;
    }

    public predictProba(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('CalibratedClassifierCV must be fitted before calling predictProba');
        }
        const n = X.length;
        const K = this.classes.length;
        const acc: number[][] = Array.from({ length: n }, () => new Array<number>(K).fill(0));

        for (const pair of this.calibratedPairs) {
            const scores = this.rawScores(pair.classifier, X, K);
            if (K === 2) {
                const p = pair.calibrators[0].predict(scores.map((row) => row[0]));
                for (let i = 0; i < n; i++) {
                    const p1 = Math.min(Math.max(p[i], 0), 1);
                    acc[i][0] += 1 - p1;
                    acc[i][1] += p1;
                }
            } else {
                const perClass: number[][] = [];
                for (let k = 0; k < K; k++) {
                    perClass.push(pair.calibrators[k].predict(scores.map((row) => row[k])));
                }
                for (let i = 0; i < n; i++) {
                    let sum = 0;
                    for (let k = 0; k < K; k++) {
                        perClass[k][i] = Math.min(Math.max(perClass[k][i], 0), 1);
                        sum += perClass[k][i];
                    }
                    for (let k = 0; k < K; k++) {
                        // all-zero rows fall back to the uniform distribution,
                        // as in sklearn
                        acc[i][k] += sum > 0 ? perClass[k][i] / sum : 1 / K;
                    }
                }
            }
        }

        const nPairs = this.calibratedPairs.length;
        return acc.map((row) => row.map((v) => v / nPairs));
    }

    public predict(X: number[][]): number[] {
        const proba = this.predictProba(X);
        return proba.map((row) => {
            let best = 0;
            for (let k = 1; k < row.length; k++) {
                if (row[k] > row[best]) best = k;
            }
            return this.classes[best];
        });
    }
}
registerEstimator('CalibratedClassifierCV', CalibratedClassifierCV);
