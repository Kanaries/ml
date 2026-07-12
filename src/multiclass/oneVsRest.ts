import { ClassifierBase } from '../base';
import { BaseEstimator, Params, registerEstimator } from '../base/estimator';
import {
    ClassifierLike,
    argmax,
    binaryScores,
    flattenDecision,
    sortedUniqueLabels,
    splitEstimatorParams,
    validateClassifierEstimator,
    validateFitInputs,
} from './common';

export interface OneVsRestClassifierProps {
    /** Prototype classifier; one unfitted clone is fitted per class. */
    estimator: BaseEstimator;
}

/**
 * One-vs-rest (one-vs-all) multiclass strategy, mirroring sklearn's
 * `OneVsRestClassifier`.
 *
 * For each of the K classes, a clone of `estimator` is fitted on a binary
 * problem where samples of that class are relabeled 1 and every other sample
 * 0. This turns any binary-only classifier into a multiclass classifier.
 *
 * Prediction picks the class whose member produces the highest score
 * (first class wins ties). The per-member score is, in order of preference:
 * `predictProba`'s positive-class column, `decisionFunction`, or the raw
 * 0/1 `predict` output (vote).
 *
 * Deviations from sklearn (documented):
 *  - sklearn fits a single estimator when y is binary (LabelBinarizer emits
 *    one column); this implementation always fits one member per class.
 *  - `predictProba` rows whose positive-class probabilities sum to 0 fall
 *    back to a uniform distribution instead of producing NaN.
 *
 * Nested params are addressable as `estimator__<param>` in `setParams`.
 */
export class OneVsRestClassifier extends ClassifierBase {
    private estimator: ClassifierLike;
    private classes_: number[];
    private estimators_: ClassifierLike[];

    constructor(props: OneVsRestClassifierProps) {
        super();
        const { estimator } = props ?? {};
        this.estimator = validateClassifierEstimator(estimator, 'OneVsRestClassifier');
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

    /** The fitted per-class members, ordered like `classes`. */
    public get estimators(): ClassifierLike[] {
        return this.estimators_.slice();
    }

    public fit(X: number[][], y: number[], sampleWeight?: number[]): void {
        validateFitInputs(X, y, 'OneVsRestClassifier');
        const classes = sortedUniqueLabels(y);
        if (classes.length < 2) {
            throw new Error('OneVsRestClassifier requires at least 2 classes in y');
        }
        this.classes_ = classes;
        this.estimators_ = classes.map((cls) => {
            const member = this.estimator.clone() as ClassifierLike;
            member.fit(X, y.map((v) => (v === cls ? 1 : 0)), sampleWeight);
            return member;
        });
    }

    private ensureFitted(): void {
        if (this.estimators_.length === 0) {
            throw new Error('OneVsRestClassifier must be fitted before inference');
        }
    }

    /** Per-class score matrix [nSamples][nClasses] used by predict. */
    private memberScores(X: number[][]): number[][] {
        const cols = this.estimators_.map((est) => binaryScores(est, X));
        return X.map((_, i) => cols.map((col) => col[i]));
    }

    public predict(X: number[][]): number[] {
        this.ensureFitted();
        return this.memberScores(X).map((row) => this.classes_[argmax(row)]);
    }

    /**
     * Positive-class probability of each member, normalized so every row
     * sums to 1 (sklearn behavior). Requires the members to implement
     * `predictProba`.
     */
    public predictProba(X: number[][]): number[][] {
        this.ensureFitted();
        if (typeof this.estimators_[0].predictProba !== 'function') {
            throw new Error('The wrapped estimator does not implement predictProba()');
        }
        const cols = this.estimators_.map((est) => est.predictProba!(X).map((row) => row[row.length - 1]));
        return X.map((_, i) => {
            const row = cols.map((col) => col[i]);
            const total = row.reduce((a, b) => a + b, 0);
            // sklearn divides by the row sum; guard the all-zero row with a
            // uniform distribution instead of NaN
            return total > 0 ? row.map((v) => v / total) : row.map(() => 1 / row.length);
        });
    }

    /**
     * Per-class decision values, shape [nSamples][nClasses]. Requires the
     * members to implement `decisionFunction`.
     */
    public decisionFunction(X: number[][]): number[][] {
        this.ensureFitted();
        if (typeof this.estimators_[0].decisionFunction !== 'function') {
            throw new Error('The wrapped estimator does not implement decisionFunction()');
        }
        const cols = this.estimators_.map((est) => flattenDecision(est.decisionFunction!(X)));
        return X.map((_, i) => cols.map((col) => col[i]));
    }
}
registerEstimator('OneVsRestClassifier', OneVsRestClassifier);
