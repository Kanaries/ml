import { BaseEstimator, Params, registerEstimator } from '../base/estimator';
import { r2Score } from '../metrics';
import {
    SingleOutputEstimator,
    splitEstimatorParams,
    validateMultiOutputTargets,
    validateSingleOutputEstimator,
} from './common';

export interface MultiOutputRegressorProps {
    /** Prototype regressor; one unfitted clone is fitted per output column. */
    estimator: BaseEstimator;
}

/**
 * Multi-target regression wrapper, mirroring sklearn's
 * `MultiOutputRegressor`: one clone of `estimator` is fitted per output
 * column of the 2-D target matrix `Y` [nSamples][nOutputs].
 *
 * `fit(X, Y)` takes a 2-D target, so this extends `BaseEstimator` directly
 * rather than `RegressorBase` (whose contract is 1-D y).
 *
 * `score(X, Y)` is the **uniform-average R²** over the outputs: R² is
 * computed per output column and the unweighted mean is returned — sklearn's
 * `MultiOutputRegressor.score` (`r2_score` with
 * `multioutput='uniform_average'`).
 *
 * Nested params are addressable as `estimator__<param>` in `setParams`.
 */
export class MultiOutputRegressor extends BaseEstimator {
    private estimator: SingleOutputEstimator;
    private estimators_: SingleOutputEstimator[];

    constructor(props: MultiOutputRegressorProps) {
        super();
        const { estimator } = props ?? {};
        this.estimator = validateSingleOutputEstimator(estimator, 'MultiOutputRegressor');
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
        validateMultiOutputTargets(X, Y, 'MultiOutputRegressor');
        const nOutputs = Y[0].length;
        this.estimators_ = Array.from({ length: nOutputs }, (_, j) => {
            const member = this.estimator.clone() as SingleOutputEstimator;
            member.fit(X, Y.map((row) => row[j]), sampleWeight);
            return member;
        });
    }

    private ensureFitted(): void {
        if (this.estimators_.length === 0) {
            throw new Error('MultiOutputRegressor must be fitted before inference');
        }
    }

    /** Predicted target matrix, shape [nSamples][nOutputs]. */
    public predict(X: number[][]): number[][] {
        this.ensureFitted();
        const cols = this.estimators_.map((est) => est.predict(X));
        return X.map((_, i) => cols.map((col) => col[i]));
    }

    /**
     * Uniform-average R² over the outputs (mean of per-column R²), matching
     * sklearn's `MultiOutputRegressor.score`.
     */
    public score(X: number[][], Y: number[][]): number {
        validateMultiOutputTargets(X, Y, 'MultiOutputRegressor');
        this.ensureFitted();
        if (Y[0].length !== this.estimators_.length) {
            throw new Error(
                `MultiOutputRegressor: Y has ${Y[0].length} outputs but the model was fitted on ${this.estimators_.length}`
            );
        }
        const pred = this.predict(X);
        const nOutputs = this.estimators_.length;
        let total = 0;
        for (let j = 0; j < nOutputs; j++) {
            total += r2Score(pred.map((row) => row[j]), Y.map((row) => row[j]));
        }
        return total / nOutputs;
    }
}
registerEstimator('MultiOutputRegressor', MultiOutputRegressor);
