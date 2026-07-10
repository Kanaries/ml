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
 * Weighted choice over `masses` for a draw `r` in [0, totalMass).
 * Entries with zero (or negative) mass are skipped so that points with
 * zero probability can never be selected, even when `r` lands exactly on
 * a cumulative-sum boundary (e.g. randomState() returning 0).
 */
function weightedChoice(masses: number[], r: number): number {
    let cum = 0;
    let lastPositive = -1;
    for (let i = 0; i < masses.length; i++) {
        if (masses[i] <= 0) continue;
        cum += masses[i];
        lastPositive = i;
        if (r <= cum) {
            return i;
        }
    }
    // numerical fallback: return the last point with positive mass
    return lastPositive;
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
    let weights = sampleWeight ? sampleWeight.slice() : X.map(() => 1);
    let weightSum = weights.reduce((a, b) => a + b, 0);
    if (weightSum <= 0) {
        // degenerate weights: fall back to uniform sampling
        weights = X.map(() => 1);
        weightSum = nSamples;
    }
    const centers: number[][] = [];
    const indices: number[] = [];

    // choose first center
    const firstIdx = weightedChoice(weights, randomState() * weightSum);
    centers.push(X[firstIdx]);
    indices.push(firstIdx);

    const closestDistSq = X.map((x) => euclideanSquare(x, centers[0]));

    while (centers.length < n_clusters) {
        const masses: number[] = new Array(nSamples);
        let distSqSum = 0;
        for (let i = 0; i < nSamples; i++) {
            masses[i] = closestDistSq[i] * weights[i];
            distSqSum += masses[i];
        }
        if (distSqSum === 0) {
            // Remaining probability mass is zero (e.g. all points identical
            // or every distinct point already chosen). Fall back to a
            // weighted pick, allowing repeated centers, so that exactly
            // n_clusters centers are returned (sklearn-style behaviour).
            const idx = weightedChoice(weights, randomState() * weightSum);
            centers.push(X[idx]);
            indices.push(idx);
            continue;
        }
        const nextIdx = weightedChoice(masses, randomState() * distSqSum);
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
