import type { BaseEstimator } from './estimator';

export type Coefficients = number[] | number[][];
export type EstimatorCapability = 'coef' | 'featureImportances';

export interface CoefficientCapability {
    /** Fitted input-feature coefficients. A copy is returned to protect model state. */
    readonly coef: Coefficients;
}

export interface FeatureImportanceCapability {
    /** Normalized non-negative input-feature importances. */
    readonly featureImportances: number[];
}

/**
 * Capability declaration is structural: an estimator opts in by defining the
 * named public property/getter. BaseEstimator intentionally does not provide
 * placeholders, so unsupported estimators (for example kernel SVMs) remain
 * distinguishable from supported-but-unfitted estimators.
 */
export function declaresEstimatorCapability(
    estimator: BaseEstimator,
    capability: EstimatorCapability,
): boolean {
    let target: object | null = estimator;
    while (target !== null) {
        if (Object.prototype.hasOwnProperty.call(target, capability)) return true;
        target = Object.getPrototypeOf(target);
    }
    return false;
}

export function hasCoefficientCapability(estimator: BaseEstimator): estimator is BaseEstimator & CoefficientCapability {
    return declaresEstimatorCapability(estimator, 'coef');
}

export function hasFeatureImportanceCapability(
    estimator: BaseEstimator,
): estimator is BaseEstimator & FeatureImportanceCapability {
    return declaresEstimatorCapability(estimator, 'featureImportances');
}

/** Flatten one- or two-dimensional coefficients into one importance per input feature. */
export function coefficientImportances(coef: Coefficients): number[] {
    if (coef.length === 0) return [];
    if (!Array.isArray(coef[0])) return (coef as number[]).map(Math.abs);
    const rows = coef as number[][];
    const nFeatures = rows[0].length;
    if (rows.some(row => row.length !== nFeatures)) {
        throw new Error('coef rows must all have the same number of features');
    }
    const out = new Array(nFeatures).fill(0);
    for (const row of rows) for (let j = 0; j < nFeatures; j++) out[j] += Math.abs(row[j]);
    return out;
}
