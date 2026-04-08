export function validateMatrix(X: number[][]): void {
    if (X.length === 0) {
        throw new Error('X must be non-empty');
    }
    const nFeatures = X[0].length;
    if (nFeatures === 0) {
        throw new Error('X must contain at least one feature');
    }
    for (let i = 1; i < X.length; i++) {
        if (X[i].length !== nFeatures) {
            throw new Error('all rows in X must have the same length');
        }
    }
}

export function validateXY(X: number[][], y: number[]): void {
    if (X.length !== y.length) {
        throw new Error('X and y must have the same length');
    }
    if (X.length === 0 || y.length === 0) {
        throw new Error('X and y must be non-empty');
    }
    validateMatrix(X);
}

export function sortedUniqueLabels(y: number[]): number[] {
    return Array.from(new Set(y)).sort((a, b) => a - b);
}

export function ensureClassPrior(
    classPrior: number[] | null | undefined,
    nClasses: number,
    label: 'priors' | 'classPrior',
): number[] | null {
    if (classPrior === null || classPrior === undefined) {
        return null;
    }
    if (classPrior.length !== nClasses) {
        throw new Error(`${label} must match number of classes`);
    }
    let sum = 0;
    for (const prior of classPrior) {
        if (!Number.isFinite(prior) || prior < 0) {
            throw new Error(`${label} must contain non-negative finite values`);
        }
        sum += prior;
    }
    if (Math.abs(sum - 1) > 1e-9) {
        throw new Error(`${label} must sum to 1`);
    }
    return classPrior.slice();
}

export function classLogPriorFromCounts(
    classCount: number[],
    fitPrior: boolean,
    classPrior: number[] | null,
): number[] {
    if (classPrior) {
        return classPrior.map(p => Math.log(p));
    }
    if (fitPrior) {
        const total = classCount.reduce((acc, v) => acc + v, 0);
        return classCount.map(c => Math.log(c / total));
    }
    return new Array(classCount.length).fill(Math.log(1 / classCount.length));
}

export function argmax(values: number[]): number {
    if (values.length === 0) {
        throw new Error('cannot compute argmax of an empty array');
    }
    let bestIndex = 0;
    let bestValue = values[0];
    for (let i = 1; i < values.length; i++) {
        if (values[i] > bestValue) {
            bestValue = values[i];
            bestIndex = i;
        }
    }
    return bestIndex;
}
