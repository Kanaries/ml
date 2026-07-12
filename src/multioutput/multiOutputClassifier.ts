import { BaseEstimator, Params, registerEstimator } from '../base/estimator';
import {
    SingleOutputEstimator,
    splitEstimatorParams,
    validateMultiOutputTargets,
    validateSingleOutputEstimator,
} from './common';

export interface MultiOutputClassifierProps {
    /** Prototype classifier; one unfitted clone is fitted per output column. */
    estimator: BaseEstimator;
}

/**
 * Multi-target classification wrapper, mirroring sklearn's
 * `MultiOutputClassifier`: one clone of `estimator` is fitted per output
 * column of the 2-D target matrix `Y` [nSamples][nOutputs].
 *
 * `fit(X, Y)` takes a 2-D target, so this extends `BaseEstimator` directly
 * rather than `ClassifierBase` (whose contract is 1-D y).
 *
 * `score(X, Y)` is **subset accuracy** (exact-match): the fraction of samples
 * whose *entire* output row is predicted correctly — sklearn's
 * `MultiOutputClassifier.score`, which applies `accuracy_score` to 2-D
 * targets. This is deliberately harsher than the mean of per-output
 * accuracies.
 *
 * Nested params are addressable as `estimator__<param>` in `setParams`.
 */
export class MultiOutputClassifier extends BaseEstimator {
    private estimator: SingleOutputEstimator;
    private estimators_: SingleOutputEstimator[];

    constructor(props: MultiOutputClassifierProps) {
        super();
        const { estimator } = props ?? {};
        this.estimator = validateSingleOutputEstimator(estimator, 'MultiOutputClassifier');
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

    /** The fitted per-output members, ordered like Y's columns. */
    public get estimators(): SingleOutputEstimator[] {
        return this.estimators_.slice();
    }

    public fit(X: number[][], Y: number[][], sampleWeight?: number[]): void {
        validateMultiOutputTargets(X, Y, 'MultiOutputClassifier');
        const nOutputs = Y[0].length;
        this.estimators_ = Array.from({ length: nOutputs }, (_, j) => {
            const member = this.estimator.clone() as SingleOutputEstimator;
            member.fit(X, Y.map((row) => row[j]), sampleWeight);
            return member;
        });
    }

    private ensureFitted(): void {
        if (this.estimators_.length === 0) {
            throw new Error('MultiOutputClassifier must be fitted before inference');
        }
    }

    /** Predicted target matrix, shape [nSamples][nOutputs]. */
    public predict(X: number[][]): number[][] {
        this.ensureFitted();
        const cols = this.estimators_.map((est) => est.predict(X));
        return X.map((_, i) => cols.map((col) => col[i]));
    }

    /**
     * Per-output class-membership probabilities, shape
     * [nOutputs][nSamples][nClasses(output)] — one probability matrix per
     * output, like sklearn's list of arrays. Requires the members to
     * implement `predictProba`.
     */
    public predictProba(X: number[][]): number[][][] {
        this.ensureFitted();
        if (typeof this.estimators_[0].predictProba !== 'function') {
            throw new Error('The wrapped estimator does not implement predictProba()');
        }
        return this.estimators_.map((est) => est.predictProba!(X));
    }

    /**
     * Subset (exact-match) accuracy: the fraction of rows where every output
     * is predicted correctly. Matches sklearn's `MultiOutputClassifier.score`.
     */
    public score(X: number[][], Y: number[][]): number {
        validateMultiOutputTargets(X, Y, 'MultiOutputClassifier');
        this.ensureFitted();
        if (Y[0].length !== this.estimators_.length) {
            throw new Error(
                `MultiOutputClassifier: Y has ${Y[0].length} outputs but the model was fitted on ${this.estimators_.length}`
            );
        }
        const pred = this.predict(X);
        let exact = 0;
        for (let i = 0; i < Y.length; i++) {
            if (Y[i].every((v, j) => v === pred[i][j])) {
                exact += 1;
            }
        }
        return exact / Y.length;
    }
}
registerEstimator('MultiOutputClassifier', MultiOutputClassifier);
