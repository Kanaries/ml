import { BaseEstimator } from '../base';
import { coefficientImportances, hasCoefficientCapability, hasFeatureImportanceCapability } from '../base/capabilities';

export interface FeatureSelectingEstimator extends BaseEstimator {
    fit(X: number[][], y: number[]): void;
    predict?(X: number[][]): number[];
    score?(X: number[][], y: number[]): number;
}

export function estimatorImportances(estimator: FeatureSelectingEstimator): number[] {
    if (hasFeatureImportanceCapability(estimator)) return estimator.featureImportances.slice();
    if (hasCoefficientCapability(estimator)) return coefficientImportances(estimator.coef);
    throw new Error(`${estimator.constructor.name} must expose coef or featureImportances`);
}

/** sklearn RFE transform_func="square": sum squared multiclass coefficients. */
export function estimatorImportancesSquared(estimator: FeatureSelectingEstimator): number[] {
    if (hasFeatureImportanceCapability(estimator)) return estimator.featureImportances.slice();
    if (!hasCoefficientCapability(estimator)) {
        throw new Error(`${estimator.constructor.name} must expose coef or featureImportances`);
    }
    const coef = estimator.coef;
    if (coef.length === 0) return [];
    if (!Array.isArray(coef[0])) return (coef as number[]).map(value => value * value);
    const rows = coef as number[][];
    const nFeatures = rows[0].length;
    if (rows.some(row => row.length !== nFeatures)) throw new Error('coef rows must all have the same number of features');
    const result = new Array(nFeatures).fill(0);
    for (const row of rows) for (let j = 0; j < nFeatures; j++) result[j] += row[j] * row[j];
    return result;
}

export function selectColumns(X: number[][], indices: number[]): number[][] {
    return X.map(row => indices.map(index => row[index]));
}

export function validateSelectionInput(X: number[][], y?: number[]): number {
    if (X.length === 0 || X[0].length === 0) throw new Error('X must be non-empty with at least one feature');
    if (y && y.length !== X.length) throw new Error('X and y must have the same length');
    const nFeatures = X[0].length;
    if (X.some(row => row.length !== nFeatures)) throw new Error('all rows in X must have the same length');
    return nFeatures;
}

export function resolveFeatureCount(value: number | undefined, nFeatures: number, fallback: number): number {
    if (value === undefined) return fallback;
    const count = value > 0 && value < 1 ? Math.floor(value * nFeatures) : value;
    if (!Number.isInteger(count) || count < 1 || count > nFeatures) {
        throw new Error('feature count must resolve to an integer between 1 and nFeatures');
    }
    return count;
}

export function resolveStep(step: number, initialFeatureCount: number): number {
    const amount = step > 0 && step < 1 ? Math.max(1, Math.floor(step * initialFeatureCount)) : step;
    if (!Number.isInteger(amount) || amount < 1) throw new Error('step must be a positive integer or fraction in (0, 1)');
    return amount;
}
