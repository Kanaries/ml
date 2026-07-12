/** Shared validation and small numeric helpers for discriminant analysis. */

export function validateXy(X: number[][], y: number[], name: string): void {
    if (!Array.isArray(X) || X.length === 0 || !Array.isArray(y) || y.length === 0) {
        throw new Error(`${name}: X and y must be non-empty`);
    }
    if (X.length !== y.length) {
        throw new Error(`${name}: X and y must have the same length (got ${X.length} and ${y.length})`);
    }
    const p = X[0].length;
    if (p === 0) {
        throw new Error(`${name}: X must have at least one feature`);
    }
    for (const row of X) {
        if (row.length !== p) {
            throw new Error(`${name}: all rows of X must have the same number of features`);
        }
    }
}

export function validatePredictInput(X: number[][], nFeatures: number, name: string): void {
    for (const row of X) {
        if (row.length !== nFeatures) {
            throw new Error(`${name}: expected ${nFeatures} features, got ${row.length}`);
        }
    }
}

export function sortedUniqueLabels(y: number[]): number[] {
    return Array.from(new Set(y)).sort((a, b) => a - b);
}

/**
 * Resolve class priors: from explicit user priors (validated, normalized to
 * sum to 1 like sklearn) or from class frequencies.
 */
export function resolvePriors(
    priors: number[] | undefined,
    counts: number[],
    nSamples: number,
    name: string
): number[] {
    if (priors == null) {
        return counts.map((c) => c / nSamples);
    }
    if (priors.length !== counts.length) {
        throw new Error(`${name}: priors must have one entry per class (${counts.length}), got ${priors.length}`);
    }
    let sum = 0;
    for (const p of priors) {
        if (!Number.isFinite(p) || p <= 0) {
            throw new Error(`${name}: priors must be positive finite numbers`);
        }
        sum += p;
    }
    return priors.map((p) => p / sum);
}

/** Row-wise softmax with max-subtraction for numerical stability. */
export function softmaxRows(scores: number[][]): number[][] {
    return scores.map((row) => {
        let max = -Infinity;
        for (const v of row) if (v > max) max = v;
        const exps = row.map((v) => Math.exp(v - max));
        let total = 0;
        for (const e of exps) total += e;
        return exps.map((e) => e / total);
    });
}

export function argmaxRow(row: number[]): number {
    let best = 0;
    for (let i = 1; i < row.length; i++) {
        if (row[i] > row[best]) best = i;
    }
    return best;
}

/** Per-class means; `classIdx[i]` maps sample i to its class index. */
export function classMeans(X: number[][], classIdx: number[], nClasses: number): number[][] {
    const p = X[0].length;
    const means: number[][] = [];
    const counts = new Array<number>(nClasses).fill(0);
    for (let k = 0; k < nClasses; k++) means.push(new Array<number>(p).fill(0));
    for (let i = 0; i < X.length; i++) {
        const k = classIdx[i];
        counts[k] += 1;
        for (let j = 0; j < p; j++) means[k][j] += X[i][j];
    }
    for (let k = 0; k < nClasses; k++) {
        for (let j = 0; j < p; j++) means[k][j] /= counts[k];
    }
    return means;
}
