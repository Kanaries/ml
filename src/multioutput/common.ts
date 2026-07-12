import { BaseEstimator, Params } from '../base/estimator';

/**
 * Structural contract for the wrapped single-output estimator. Validated at
 * runtime by shape so any estimator following the BaseEstimator contract
 * works.
 */
export interface SingleOutputEstimator extends BaseEstimator {
    fit(X: number[][], y: number[], sampleWeight?: number[]): void;
    predict(X: number[][]): number[];
    predictProba?(X: number[][]): number[][];
}

export function validateSingleOutputEstimator(estimator: unknown, wrapper: string): SingleOutputEstimator {
    if (
        !(estimator instanceof BaseEstimator) ||
        typeof (estimator as SingleOutputEstimator).fit !== 'function' ||
        typeof (estimator as SingleOutputEstimator).predict !== 'function'
    ) {
        throw new Error(`${wrapper} requires an "estimator" param implementing fit() and predict()`);
    }
    return estimator as SingleOutputEstimator;
}

/** Validate the 2-D multi-output target matrix against X. */
export function validateMultiOutputTargets(X: number[][], Y: number[][], wrapper: string): void {
    if (!Array.isArray(X) || !Array.isArray(Y) || X.length === 0 || Y.length === 0) {
        throw new Error(`${wrapper}: X and Y must be non-empty arrays`);
    }
    if (X.length !== Y.length) {
        throw new Error(`${wrapper}: X and Y must have the same length (${X.length} !== ${Y.length})`);
    }
    if (!Array.isArray(Y[0]) || Y[0].length === 0) {
        throw new Error(`${wrapper}: Y must be a 2-D array [nSamples][nOutputs] with at least one output`);
    }
    const nOutputs = Y[0].length;
    for (const row of Y) {
        if (!Array.isArray(row) || row.length !== nOutputs) {
            throw new Error(`${wrapper}: all rows of Y must have the same number of outputs (${nOutputs})`);
        }
    }
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
