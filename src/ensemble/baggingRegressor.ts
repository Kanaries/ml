import { BaseEstimator, RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { DecisionTreeRegressor } from '../tree';
import { createRandomGenerator } from '../utils';
import { resolveSubsetSize, SubsetSizeOption } from '../utils/paramResolvers';

interface RegressorLike extends BaseEstimator {
    fit(X: number[][], y: number[]): void;
    predict(X: number[][]): number[];
}

export interface BaggingRegressorProps {
    estimator?: RegressorLike;
    nEstimators?: number;
    maxSamples?: SubsetSizeOption;
    bootstrap?: boolean;
    randomState?: number;
}

export class BaggingRegressor extends RegressorBase {
    private estimator?: RegressorLike;
    private nEstimators: number;
    private maxSamples?: SubsetSizeOption;
    private bootstrap: boolean;
    private randomState?: number;
    private estimators: RegressorLike[];

    constructor(props: BaggingRegressorProps = {}) {
        super();
        const { estimator, nEstimators = 10, maxSamples, bootstrap = true, randomState } = props;
        this.estimator = estimator;
        this.nEstimators = nEstimators;
        this.maxSamples = maxSamples;
        this.bootstrap = bootstrap;
        this.randomState = randomState;
        this.estimators = [];
    }

    public getParams(): Params {
        return {
            estimator: this.estimator,
            nEstimators: this.nEstimators,
            maxSamples: this.maxSamples,
            bootstrap: this.bootstrap,
            randomState: this.randomState,
        };
    }

    public fit(X: number[][], y: number[]): void {
        if (X.length === 0 || X.length !== y.length) throw new Error('X and y must be non-empty and have the same length');
        if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) throw new Error('nEstimators must be a positive integer');
        const count = resolveSubsetSize(this.maxSamples, X.length, 'maxSamples');
        const random = createRandomGenerator(this.randomState);
        this.estimators = [];
        for (let n = 0; n < this.nEstimators; n++) {
            const seed = Math.floor(random() * 1e9);
            const estimator = this.estimator
                ? this.estimator.clone() as RegressorLike
                : new DecisionTreeRegressor({ randomState: seed });
            if (this.estimator) {
                const params = estimator.getParams();
                if (Object.prototype.hasOwnProperty.call(params, 'randomState')) estimator.setParams({ randomState: seed });
                else if (Object.prototype.hasOwnProperty.call(params, 'random_state')) estimator.setParams({ random_state: seed });
            }
            const indices = Array.from({ length: X.length }, (_, i) => i);
            if (this.bootstrap) {
                for (let i = 0; i < count; i++) indices[i] = Math.floor(random() * X.length);
            } else {
                for (let i = indices.length - 1; i > 0; i--) {
                    const j = Math.floor(random() * (i + 1));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }
            }
            const selected = indices.slice(0, count);
            estimator.fit(selected.map(i => X[i]), selected.map(i => y[i]));
            this.estimators.push(estimator);
        }
    }

    public predict(X: number[][]): number[] {
        if (this.estimators.length === 0) throw new Error('model is not fitted');
        const rows = this.estimators.map(estimator => estimator.predict(X));
        return X.map((_, i) => rows.reduce((sum, row) => sum + row[i], 0) / rows.length);
    }
}
registerEstimator('BaggingRegressor', BaggingRegressor);
