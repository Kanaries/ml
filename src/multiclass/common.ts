import { BaseEstimator, Params } from '../base/estimator';

/**
 * Structural contract for the wrapped classifier. Validated at runtime by
 * shape so any estimator following the BaseEstimator contract works,
 * including binary-only classifiers (that is the point of these wrappers).
 */
export interface ClassifierLike extends BaseEstimator {
    fit(X: number[][], y: number[], sampleWeight?: number[]): void;
    predict(X: number[][]): number[];
    predictProba?(X: number[][]): number[][];
    decisionFunction?(X: number[][]): number[] | number[][];
}

export function validateClassifierEstimator(estimator: unknown, wrapper: string): ClassifierLike {
    if (
        !(estimator instanceof BaseEstimator) ||
        typeof (estimator as ClassifierLike).fit !== 'function' ||
        typeof (estimator as ClassifierLike).predict !== 'function'
    ) {
        throw new Error(`${wrapper} requires an "estimator" param implementing fit() and predict()`);
    }
    return estimator as ClassifierLike;
}

export function validateFitInputs(X: number[][], y: number[], wrapper: string): void {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length === 0 || y.length === 0) {
        throw new Error(`${wrapper}: X and y must be non-empty arrays`);
    }
    if (X.length !== y.length) {
        throw new Error(`${wrapper}: X and y must have the same length (${X.length} !== ${y.length})`);
    }
}

export function sortedUniqueLabels(y: number[]): number[] {
    return Array.from(new Set(y)).sort((a, b) => a - b);
}

/**
 * Split params into the wrapper's own params and nested params addressed as
 * `estimator__<param>` (grid-search style, mirroring Pipeline's
 * `step__param` addressing).
 */
export function splitEstimatorParams(params: Params, className: string): { own: Params; nested: Params } {
    const own: Params = {};
    const nested: Params = {};
    for (const key of Object.keys(params)) {
        const idx = key.indexOf('__');
        if (idx > 0) {
            const prefix = key.slice(0, idx);
            if (prefix !== 'estimator') {
                throw new Error(
                    `Invalid parameter "${key}" for estimator ${className}. ` +
                        'Nested params must use the "estimator__<param>" form.'
                );
            }
            nested[key.slice(idx + 2)] = params[key];
        } else {
            own[key] = params[key];
        }
    }
    return { own, nested };
}

/** Flatten a member's decisionFunction output to one score per sample. */
export function flattenDecision(d: number[] | number[][]): number[] {
    if (d.length === 0 || typeof d[0] === 'number') {
        return d as number[];
    }
    // a binary member returning a matrix: take the positive-class column
    return (d as number[][]).map((row) => row[row.length - 1]);
}

/**
 * Per-sample score of the positive class from a fitted binary member trained
 * on labels {0, 1}: predictProba's positive-class column when available, else
 * decisionFunction (flattened), else the raw predict output (0/1 votes).
 */
export function binaryScores(est: ClassifierLike, X: number[][]): number[] {
    if (typeof est.predictProba === 'function') {
        return est.predictProba(X).map((row) => row[row.length - 1]);
    }
    if (typeof est.decisionFunction === 'function') {
        return flattenDecision(est.decisionFunction(X));
    }
    return est.predict(X);
}

/** Index of the row maximum; the first maximum wins on ties. */
export function argmax(row: number[]): number {
    let best = 0;
    for (let i = 1; i < row.length; i++) {
        if (row[i] > row[best]) {
            best = i;
        }
    }
    return best;
}
