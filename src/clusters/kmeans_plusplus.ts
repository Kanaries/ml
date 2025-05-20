export interface KMeansPlusPlusResult {
    centers: number[][];
    indices: number[];
}

function euclideanSquare(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * Initialize cluster centers according to the kmeans++ strategy.
 * @param X samples matrix
 * @param n_clusters number of clusters
 * @param sampleWeight weights for each sample
 * @param randomState optional random generator
 * @returns selected centers and their indices
 */
export function kmeansPlusPlus(
    X: number[][],
    n_clusters: number,
    sampleWeight?: number[],
    randomState: () => number = Math.random
): KMeansPlusPlusResult {
    if (X.length === 0) {
        return { centers: [], indices: [] };
    }
    const nSamples = X.length;
    const weights = sampleWeight ? sampleWeight.slice() : X.map(() => 1);
    const centers: number[][] = [];
    const indices: number[] = [];

    // choose first center
    let weightSum = weights.reduce((a, b) => a + b, 0);
    let r = randomState() * weightSum;
    let cum = 0;
    let firstIdx = 0;
    for (let i = 0; i < nSamples; i++) {
        cum += weights[i];
        if (r <= cum) {
            firstIdx = i;
            break;
        }
    }
    centers.push(X[firstIdx]);
    indices.push(firstIdx);

    const closestDistSq = X.map((x) => euclideanSquare(x, centers[0]));

    while (centers.length < n_clusters) {
        let distSqSum = 0;
        for (let i = 0; i < nSamples; i++) {
            distSqSum += closestDistSq[i] * weights[i];
        }
        if (distSqSum === 0) {
            // all points the same, pick random
            let r2 = randomState() * weightSum;
            let cum2 = 0;
            let idx = 0;
            for (let i = 0; i < nSamples; i++) {
                cum2 += weights[i];
                if (r2 <= cum2) {
                    idx = i;
                    break;
                }
            }
            if (indices.indexOf(idx) === -1) {
                centers.push(X[idx]);
                indices.push(idx);
            }
            break;
        }
        let r2 = randomState() * distSqSum;
        let cum2 = 0;
        let nextIdx = 0;
        for (let i = 0; i < nSamples; i++) {
            cum2 += closestDistSq[i] * weights[i];
            if (r2 <= cum2) {
                nextIdx = i;
                break;
            }
        }
        centers.push(X[nextIdx]);
        indices.push(nextIdx);
        for (let i = 0; i < nSamples; i++) {
            const d = euclideanSquare(X[i], X[nextIdx]);
            if (d < closestDistSq[i]) {
                closestDistSq[i] = d;
            }
        }
    }

    return { centers, indices };
}
