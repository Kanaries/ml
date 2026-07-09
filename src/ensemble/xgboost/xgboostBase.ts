import { createRandomGenerator } from '../../utils';
import { XGBTree } from './xgbTree';

export interface XGBoostProps {
    nEstimators?: number;
    n_estimators?: number;
    learningRate?: number;
    learning_rate?: number;
    maxDepth?: number;
    max_depth?: number;
    lambda?: number;
    reg_lambda?: number;
    gamma?: number;
    minChildWeight?: number;
    min_child_weight?: number;
    subsample?: number;
    colsampleByTree?: number;
    colsample_bytree?: number;
    baseScore?: number;
    base_score?: number;
    randomState?: number;
    random_state?: number;
}

/**
 * Shared boosting loop for the XGBoost models (exact greedy, following
 * xgboost's defaults: eta=0.3, max_depth=6, lambda=1, gamma=0,
 * min_child_weight=1, subsample=1, colsample_bytree=1, base_score=0.5).
 * Subclasses provide the objective via gradients() and the initial margin.
 */
export abstract class XGBoostBase {
    protected nEstimators: number;
    protected learningRate: number;
    protected maxDepth: number;
    protected lambda: number;
    protected gamma: number;
    protected minChildWeight: number;
    protected subsample: number;
    protected colsampleByTree: number;
    protected baseScore: number;
    protected randomState?: number;
    protected trees: XGBTree[];
    protected baseMargin: number;
    protected fitted: boolean;

    constructor(props: XGBoostProps = {}) {
        this.nEstimators = props.nEstimators ?? props.n_estimators ?? 100;
        this.learningRate = props.learningRate ?? props.learning_rate ?? 0.3;
        this.maxDepth = props.maxDepth ?? props.max_depth ?? 6;
        this.lambda = props.lambda ?? props.reg_lambda ?? 1;
        this.gamma = props.gamma ?? 0;
        this.minChildWeight = props.minChildWeight ?? props.min_child_weight ?? 1;
        this.subsample = props.subsample ?? 1;
        this.colsampleByTree = props.colsampleByTree ?? props.colsample_bytree ?? 1;
        this.baseScore = props.baseScore ?? props.base_score ?? 0.5;
        this.randomState = props.randomState ?? props.random_state;
        this.trees = [];
        this.baseMargin = 0;
        this.fitted = false;

        if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
            throw new Error('nEstimators must be a positive integer');
        }
        if (!(this.learningRate > 0 && this.learningRate <= 1)) {
            throw new Error('learningRate (eta) must be in (0, 1]');
        }
        if (!(Number.isInteger(this.maxDepth) || this.maxDepth === Infinity) || this.maxDepth < 1) {
            throw new Error('maxDepth must be an integer >= 1 (or Infinity)');
        }
        if (!Number.isFinite(this.lambda) || this.lambda < 0) {
            throw new Error('lambda must be a non-negative finite number');
        }
        if (!Number.isFinite(this.gamma) || this.gamma < 0) {
            throw new Error('gamma must be a non-negative finite number');
        }
        if (!Number.isFinite(this.minChildWeight) || this.minChildWeight < 0) {
            throw new Error('minChildWeight must be a non-negative finite number');
        }
        if (!(this.subsample > 0 && this.subsample <= 1)) {
            throw new Error('subsample must be in (0, 1]');
        }
        if (!(this.colsampleByTree > 0 && this.colsampleByTree <= 1)) {
            throw new Error('colsampleByTree must be in (0, 1]');
        }
        if (!Number.isFinite(this.baseScore)) {
            throw new Error('baseScore must be finite');
        }
    }

    /**
     * Objective: first/second-order gradients of the loss at margin F.
     * yInternal is whatever fitBoosted() was given (raw targets for
     * regression, 0/1 labels for classification).
     */
    protected abstract gradients(F: number[], yInternal: number[]): { g: number[]; h: number[] };

    protected fitBoosted(X: number[][], yInternal: number[], initialMargin: number): void {
        const n = X.length;
        const nFeatures = X[0].length;
        const random = createRandomGenerator(this.randomState);
        this.baseMargin = initialMargin;
        this.trees = [];
        const F = new Array(n).fill(initialMargin);
        const rowSampleSize = Math.max(1, Math.floor(this.subsample * n));
        const colSampleSize = Math.max(1, Math.floor(this.colsampleByTree * nFeatures));

        for (let m = 0; m < this.nEstimators; m++) {
            const { g, h } = this.gradients(F, yInternal);

            let rowsX = X;
            let rowsG = g;
            let rowsH = h;
            if (rowSampleSize < n) {
                const indices = Array.from({ length: n }, (_, i) => i);
                for (let i = 0; i < rowSampleSize; i++) {
                    const j = i + Math.floor(random() * (n - i));
                    [indices[i], indices[j]] = [indices[j], indices[i]];
                }
                const rows = indices.slice(0, rowSampleSize);
                rowsX = rows.map(i => X[i]);
                rowsG = rows.map(i => g[i]);
                rowsH = rows.map(i => h[i]);
            }

            let featureIndices = Array.from({ length: nFeatures }, (_, i) => i);
            if (colSampleSize < nFeatures) {
                for (let i = 0; i < colSampleSize; i++) {
                    const j = i + Math.floor(random() * (nFeatures - i));
                    [featureIndices[i], featureIndices[j]] = [featureIndices[j], featureIndices[i]];
                }
                featureIndices = featureIndices.slice(0, colSampleSize);
            }

            const tree = new XGBTree({
                maxDepth: this.maxDepth,
                lambda: this.lambda,
                gamma: this.gamma,
                minChildWeight: this.minChildWeight,
            });
            tree.fit(rowsX, rowsG, rowsH, featureIndices);

            const update = tree.predict(X);
            for (let i = 0; i < n; i++) {
                F[i] += this.learningRate * update[i];
            }
            this.trees.push(tree);
        }
        this.fitted = true;
    }

    protected predictMargin(testX: number[][]): number[] {
        const F = new Array(testX.length).fill(this.baseMargin);
        for (const tree of this.trees) {
            const pred = tree.predict(testX);
            for (let i = 0; i < F.length; i++) {
                F[i] += this.learningRate * pred[i];
            }
        }
        return F;
    }

    protected validateFitInput(X: number[][], y: number[]): void {
        if (X.length === 0 || y.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }
        // missing values are not supported (no sparsity-aware default
        // directions): reject them instead of silently misrouting NaN
        for (const row of X) {
            for (const v of row) {
                if (!Number.isFinite(v)) {
                    throw new Error('X contains NaN or non-finite values, which are not supported');
                }
            }
        }
        for (const v of y) {
            if (!Number.isFinite(v)) {
                throw new Error('y contains NaN or non-finite values, which are not supported');
            }
        }
    }
}
