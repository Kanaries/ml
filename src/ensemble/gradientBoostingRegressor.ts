import { RegressorBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { DecisionTreeRegressor } from '../tree';
import { createRandomGenerator } from '../utils';
import { mean } from '../utils/stat';

export interface GradientBoostingRegressorProps {
    nEstimators?: number;
    n_estimators?: number;
    learningRate?: number;
    learning_rate?: number;
    maxDepth?: number;
    max_depth?: number;
    minSamplesSplit?: number;
    min_samples_split?: number;
    subsample?: number;
    maxFeatures?: number | 'sqrt' | 'log2';
    max_features?: number | 'sqrt' | 'log2';
    randomState?: number;
    random_state?: number;
}

/**
 * Gradient boosting with squared-error loss, following sklearn's
 * GradientBoostingRegressor: F_0 = mean(y), then each round fits a
 * regression tree to the residuals y - F and updates
 * F += learning_rate * tree(x). subsample < 1 draws rows without
 * replacement per round (stochastic gradient boosting); trees are still
 * used to update F over all samples.
 */
export class GradientBoostingRegressor extends RegressorBase {
    private nEstimators: number;
    private learningRate: number;
    private maxDepth: number;
    private minSamplesSplit: number;
    private subsample: number;
    private maxFeatures?: number | 'sqrt' | 'log2';
    private randomState?: number;
    private estimators: DecisionTreeRegressor[];
    private initPrediction: number;
    private fitted: boolean;

    constructor(props: GradientBoostingRegressorProps = {}) {
        super();
        this.nEstimators = props.nEstimators ?? props.n_estimators ?? 100;
        this.learningRate = props.learningRate ?? props.learning_rate ?? 0.1;
        this.maxDepth = props.maxDepth ?? props.max_depth ?? 3;
        this.minSamplesSplit = props.minSamplesSplit ?? props.min_samples_split ?? 2;
        this.subsample = props.subsample ?? 1.0;
        this.maxFeatures = props.maxFeatures ?? props.max_features;
        this.randomState = props.randomState ?? props.random_state;
        this.estimators = [];
        this.initPrediction = 0;
        this.fitted = false;
        if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
            throw new Error('nEstimators must be a positive integer');
        }
        if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
            throw new Error('learningRate must be a positive finite number');
        }
        if (!(this.subsample > 0 && this.subsample <= 1)) {
            throw new Error('subsample must be in (0, 1]');
        }
        if (!(Number.isInteger(this.maxDepth) || this.maxDepth === Infinity) || this.maxDepth < 1) {
            throw new Error('maxDepth must be an integer >= 1 (or Infinity)');
        }
        if (!Number.isInteger(this.minSamplesSplit) || this.minSamplesSplit < 2) {
            throw new Error('minSamplesSplit must be an integer >= 2');
        }
    }

    public getParams(): Params {
        // canonical camelCase keys; the snake_case aliases remain accepted
        // by the constructor
        return {
            nEstimators: this.nEstimators,
            learningRate: this.learningRate,
            maxDepth: this.maxDepth,
            minSamplesSplit: this.minSamplesSplit,
            subsample: this.subsample,
            maxFeatures: this.maxFeatures,
            randomState: this.randomState,
        };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        if (trainX.length === 0 || trainY.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (trainX.length !== trainY.length) {
            throw new Error('X and y must have the same length');
        }
        const n = trainX.length;
        const random = createRandomGenerator(this.randomState);
        this.initPrediction = mean(trainY);
        this.estimators = [];
        const F = new Array(n).fill(this.initPrediction);
        const subsampleSize = Math.max(1, Math.floor(this.subsample * n));

        for (let m = 0; m < this.nEstimators; m++) {
            const residuals = trainY.map((y, i) => y - F[i]);
            const tree = new DecisionTreeRegressor({
                max_depth: this.maxDepth,
                min_samples_split: this.minSamplesSplit,
                max_features: this.maxFeatures,
                randomState: Math.floor(random() * 1_000_000_000),
            });
            if (subsampleSize < n) {
                // sample rows without replacement (partial Fisher-Yates)
                const indices = Array.from({ length: n }, (_, i) => i);
                for (let i = 0; i < subsampleSize; i++) {
                    const j = i + Math.floor(random() * (n - i));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }
                const rows = indices.slice(0, subsampleSize);
                tree.fit(rows.map(i => trainX[i]), rows.map(i => residuals[i]));
            } else {
                tree.fit(trainX, residuals);
            }
            const update = tree.predict(trainX);
            for (let i = 0; i < n; i++) {
                F[i] += this.learningRate * update[i];
            }
            this.estimators.push(tree);
        }
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        const result = new Array(testX.length).fill(this.initPrediction);
        for (const tree of this.estimators) {
            const pred = tree.predict(testX);
            for (let i = 0; i < result.length; i++) {
                result[i] += this.learningRate * pred[i];
            }
        }
        return result;
    }
}
registerEstimator('GradientBoostingRegressor', GradientBoostingRegressor);
