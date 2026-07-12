import { ClassifierBase } from '../base';
import { BaseEstimator, Params, registerEstimator } from '../base/estimator';
import {
    ClassifierLike,
    argmax,
    flattenDecision,
    sortedUniqueLabels,
    splitEstimatorParams,
    validateClassifierEstimator,
    validateFitInputs,
} from './common';

export interface OneVsOneClassifierProps {
    /** Prototype classifier; one unfitted clone is fitted per class pair. */
    estimator: BaseEstimator;
}

/**
 * TypeScript port of sklearn's `_ovr_decision_function`: fold per-pair binary
 * predictions and confidences into a per-class decision matrix of
 *
 *     votes + sumOfConfidences / (3 * (|sumOfConfidences| + 1))
 *
 * `predictions[k][s]` is member k's 0/1 prediction for sample s and
 * `confidences[k][s]` its signed confidence (positive favors the pair's
 * second class). Pair k corresponds to the class-index pair (i, j) in the
 * order `for i in [0, K): for j in (i, K)`.
 *
 * The monotone transform maps each confidence sum into (-1/3, 1/3), so the
 * confidence term can only break ties between equal vote counts — it can
 * never overturn a vote.
 */
export function ovrDecisionFunction(predictions: number[][], confidences: number[][], nClasses: number): number[][] {
    const nSamples = predictions.length > 0 ? predictions[0].length : 0;
    const votes: number[][] = Array.from({ length: nSamples }, () => new Array(nClasses).fill(0));
    const sumOfConfidences: number[][] = Array.from({ length: nSamples }, () => new Array(nClasses).fill(0));
    let k = 0;
    for (let i = 0; i < nClasses; i++) {
        for (let j = i + 1; j < nClasses; j++) {
            for (let s = 0; s < nSamples; s++) {
                sumOfConfidences[s][i] -= confidences[k][s];
                sumOfConfidences[s][j] += confidences[k][s];
                if (predictions[k][s] === 0) {
                    votes[s][i] += 1;
                } else {
                    votes[s][j] += 1;
                }
            }
            k += 1;
        }
    }
    return votes.map((row, s) =>
        row.map((v, c) => {
            const sum = sumOfConfidences[s][c];
            return v + sum / (3 * (Math.abs(sum) + 1));
        })
    );
}

/**
 * One-vs-one multiclass strategy, mirroring sklearn's `OneVsOneClassifier`.
 *
 * For every unordered class pair (cI, cJ) with cI < cJ, a clone of
 * `estimator` is fitted on just that pair's samples, relabeled 0 (cI) and 1
 * (cJ). Members are stored in sklearn's pair order
 * `for i in [0, K): for j in (i, K)`.
 *
 * Prediction counts pairwise votes and, like sklearn, adds the per-class sum
 * of signed member confidences squashed into (-1/3, 1/3) (see
 * `ovrDecisionFunction`), so confidences break vote ties but never overturn
 * a vote. Member confidence is `decisionFunction` when available, else
 * `predictProba`'s positive-class column.
 *
 * Deviations from sklearn (documented):
 *  - when the members expose neither `decisionFunction` nor `predictProba`,
 *    sklearn raises; here the confidences are all 0, so vote ties fall back
 *    to the first (lowest-index) tied class.
 *
 * Nested params are addressable as `estimator__<param>` in `setParams`.
 */
export class OneVsOneClassifier extends ClassifierBase {
    private estimator: ClassifierLike;
    private classes_: number[];
    private estimators_: ClassifierLike[];

    constructor(props: OneVsOneClassifierProps) {
        super();
        const { estimator } = props ?? {};
        this.estimator = validateClassifierEstimator(estimator, 'OneVsOneClassifier');
        this.classes_ = [];
        this.estimators_ = [];
    }

    public getParams(): Params {
        return { estimator: this.estimator };
    }

    /** Supports own params plus nested `estimator__param` addressing. */
    public setParams(params: Params): this {
        const { own, nested } = splitEstimatorParams(params, this.constructor.name);
        if (Object.keys(own).length > 0) {
            super.setParams(own);
        }
        if (Object.keys(nested).length > 0) {
            this.estimator.setParams(nested);
        }
        return this;
    }

    /** Sorted class labels seen during fit. */
    public get classes(): number[] {
        return this.classes_.slice();
    }

    /** The fitted per-pair members, in sklearn's (i, j) pair order. */
    public get estimators(): ClassifierLike[] {
        return this.estimators_.slice();
    }

    public fit(X: number[][], y: number[], sampleWeight?: number[]): void {
        validateFitInputs(X, y, 'OneVsOneClassifier');
        const classes = sortedUniqueLabels(y);
        if (classes.length < 2) {
            throw new Error('OneVsOneClassifier requires at least 2 classes in y');
        }
        const estimators: ClassifierLike[] = [];
        for (let i = 0; i < classes.length; i++) {
            for (let j = i + 1; j < classes.length; j++) {
                const cI = classes[i];
                const cJ = classes[j];
                const pairX: number[][] = [];
                const pairY: number[] = [];
                const pairW: number[] = [];
                for (let s = 0; s < y.length; s++) {
                    if (y[s] === cI || y[s] === cJ) {
                        pairX.push(X[s]);
                        pairY.push(y[s] === cJ ? 1 : 0);
                        if (sampleWeight !== undefined) {
                            pairW.push(sampleWeight[s]);
                        }
                    }
                }
                const member = this.estimator.clone() as ClassifierLike;
                member.fit(pairX, pairY, sampleWeight !== undefined ? pairW : undefined);
                estimators.push(member);
            }
        }
        this.classes_ = classes;
        this.estimators_ = estimators;
    }

    private ensureFitted(): void {
        if (this.estimators_.length === 0) {
            throw new Error('OneVsOneClassifier must be fitted before inference');
        }
    }

    /** votes + squashed-confidence matrix, shape [nSamples][nClasses]. */
    private decisionMatrix(X: number[][]): number[][] {
        const predictions = this.estimators_.map((est) => est.predict(X));
        const confidences = this.estimators_.map((est) => {
            if (typeof est.decisionFunction === 'function') {
                return flattenDecision(est.decisionFunction(X));
            }
            if (typeof est.predictProba === 'function') {
                return est.predictProba(X).map((row) => row[row.length - 1]);
            }
            // no confidence source: ties fall back to the first tied class
            return new Array(X.length).fill(0);
        });
        return ovrDecisionFunction(predictions, confidences, this.classes_.length);
    }

    public predict(X: number[][]): number[] {
        this.ensureFitted();
        return this.decisionMatrix(X).map((row) => this.classes_[argmax(row)]);
    }

    /**
     * The vote + squashed-confidence matrix [nSamples][nClasses]; for binary
     * problems the positive-class column is returned as a flat array
     * (sklearn behavior).
     */
    public decisionFunction(X: number[][]): number[] | number[][] {
        this.ensureFitted();
        const Y = this.decisionMatrix(X);
        return this.classes_.length === 2 ? Y.map((row) => row[1]) : Y;
    }
}
registerEstimator('OneVsOneClassifier', OneVsOneClassifier);
