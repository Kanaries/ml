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
 * Resolved, validated hyper-parameters shared by the XGBoost models
 * (canonical camelCase keys — exactly the shape returned by getParams()).
 */
export interface XGBoostParams {
    nEstimators: number;
    learningRate: number;
    maxDepth: number;
    lambda: number;
    gamma: number;
    minChildWeight: number;
    subsample: number;
    colsampleByTree: number;
    baseScore: number;
    randomState: number | undefined;
}

/**
 * Resolve props (accepting both camelCase and snake_case aliases) to the
 * canonical parameter set, following xgboost's defaults: eta=0.3,
 * max_depth=6, lambda=1, gamma=0, min_child_weight=1, subsample=1,
 * colsample_bytree=1, base_score=0.5.
 */
export function resolveXGBoostProps(props: XGBoostProps = {}): XGBoostParams {
    const params: XGBoostParams = {
        nEstimators: props.nEstimators ?? props.n_estimators ?? 100,
        learningRate: props.learningRate ?? props.learning_rate ?? 0.3,
        maxDepth: props.maxDepth ?? props.max_depth ?? 6,
        lambda: props.lambda ?? props.reg_lambda ?? 1,
        gamma: props.gamma ?? 0,
        minChildWeight: props.minChildWeight ?? props.min_child_weight ?? 1,
        subsample: props.subsample ?? 1,
        colsampleByTree: props.colsampleByTree ?? props.colsample_bytree ?? 1,
        baseScore: props.baseScore ?? props.base_score ?? 0.5,
        randomState: props.randomState ?? props.random_state,
    };

    if (!Number.isInteger(params.nEstimators) || params.nEstimators < 1) {
        throw new Error('nEstimators must be a positive integer');
    }
    if (!(params.learningRate > 0 && params.learningRate <= 1)) {
        throw new Error('learningRate (eta) must be in (0, 1]');
    }
    if (!(Number.isInteger(params.maxDepth) || params.maxDepth === Infinity) || params.maxDepth < 1) {
        throw new Error('maxDepth must be an integer >= 1 (or Infinity)');
    }
    if (!Number.isFinite(params.lambda) || params.lambda < 0) {
        throw new Error('lambda must be a non-negative finite number');
    }
    if (!Number.isFinite(params.gamma) || params.gamma < 0) {
        throw new Error('gamma must be a non-negative finite number');
    }
    if (!Number.isFinite(params.minChildWeight) || params.minChildWeight < 0) {
        throw new Error('minChildWeight must be a non-negative finite number');
    }
    if (!(params.subsample > 0 && params.subsample <= 1)) {
        throw new Error('subsample must be in (0, 1]');
    }
    if (!(params.colsampleByTree > 0 && params.colsampleByTree <= 1)) {
        throw new Error('colsampleByTree must be in (0, 1]');
    }
    if (!Number.isFinite(params.baseScore)) {
        throw new Error('baseScore must be finite');
    }
    return params;
}

/**
 * Objective: first/second-order gradients of the loss at margin F.
 * yInternal is whatever fitBoostedTrees() was given (raw targets for
 * regression, 0/1 labels for classification).
 */
export type GradientFn = (F: number[], yInternal: number[]) => { g: number[]; h: number[] };

/**
 * Shared boosting loop for the XGBoost models (exact greedy). Returns the
 * fitted trees; the caller stores them together with the initial margin.
 */
export function fitBoostedTrees(
    X: number[][],
    yInternal: number[],
    initialMargin: number,
    params: XGBoostParams,
    gradients: GradientFn
): XGBTree[] {
    const n = X.length;
    const nFeatures = X[0].length;
    const random = createRandomGenerator(params.randomState);
    const trees: XGBTree[] = [];
    const F = new Array(n).fill(initialMargin);
    const rowSampleSize = Math.max(1, Math.floor(params.subsample * n));
    const colSampleSize = Math.max(1, Math.floor(params.colsampleByTree * nFeatures));

    for (let m = 0; m < params.nEstimators; m++) {
        const { g, h } = gradients(F, yInternal);

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
            maxDepth: params.maxDepth,
            lambda: params.lambda,
            gamma: params.gamma,
            minChildWeight: params.minChildWeight,
        });
        tree.fit(rowsX, rowsG, rowsH, featureIndices);

        const update = tree.predict(X);
        for (let i = 0; i < n; i++) {
            F[i] += params.learningRate * update[i];
        }
        trees.push(tree);
    }
    return trees;
}

export function predictBoostedMargin(
    trees: XGBTree[],
    baseMargin: number,
    learningRate: number,
    testX: number[][]
): number[] {
    const F = new Array(testX.length).fill(baseMargin);
    for (const tree of trees) {
        const pred = tree.predict(testX);
        for (let i = 0; i < F.length; i++) {
            F[i] += learningRate * pred[i];
        }
    }
    return F;
}

export function validateXGBoostFitInput(X: number[][], y: number[]): void {
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
