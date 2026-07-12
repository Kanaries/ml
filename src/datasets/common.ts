import { createRandomGenerator } from '../utils/random';

export { createRandomGenerator };

/** {X, y} pair returned by the dataset generators */
export interface Dataset {
    /** samples, shape [nSamples][nFeatures] */
    X: number[][];
    /** labels (or targets), length nSamples */
    y: number[];
}

/**
 * Wraps a uniform [0, 1) generator into a standard normal N(0, 1)
 * generator via the Box-Muller transform. Uniform values are consumed
 * in pairs and the spare normal deviate is cached between calls, so the
 * output is deterministic for a seeded `rng`.
 */
export function createGaussianGenerator(rng: () => number): () => number {
    let spare: number | null = null;
    return () => {
        if (spare !== null) {
            const value = spare;
            spare = null;
            return value;
        }
        let u = rng();
        // guard against log(0)
        while (u <= Number.EPSILON) {
            u = rng();
        }
        const v = rng();
        const magnitude = Math.sqrt(-2 * Math.log(u));
        spare = magnitude * Math.sin(2 * Math.PI * v);
        return magnitude * Math.cos(2 * Math.PI * v);
    };
}

/** uniform integer in [0, max) */
export function randInt(rng: () => number, max: number): number {
    return Math.floor(rng() * max);
}

/** uniform real in [low, high) */
export function uniform(rng: () => number, low: number, high: number): number {
    return low + (high - low) * rng();
}

/** matrix of shape [rows][cols] with i.i.d. uniform [low, high) entries */
export function uniformMatrix(rows: number, cols: number, rng: () => number, low: number, high: number): number[][] {
    const m: number[][] = [];
    for (let i = 0; i < rows; i++) {
        const row: number[] = new Array(cols);
        for (let j = 0; j < cols; j++) {
            row[j] = uniform(rng, low, high);
        }
        m.push(row);
    }
    return m;
}

/** matrix of shape [rows][cols] with i.i.d. standard normal entries */
export function gaussianMatrix(rows: number, cols: number, gaussian: () => number): number[][] {
    const m: number[][] = [];
    for (let i = 0; i < rows; i++) {
        const row: number[] = new Array(cols);
        for (let j = 0; j < cols; j++) {
            row[j] = gaussian();
        }
        m.push(row);
    }
    return m;
}

/** Fisher-Yates permutation of [0, size) */
export function permutation(size: number, rng: () => number): number[] {
    const indices = Array.from({ length: size }, (_, i) => i);
    for (let i = size - 1; i > 0; i--) {
        const j = randInt(rng, i + 1);
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices;
}

/** Shuffles the rows of X and the entries of y with one shared permutation. */
export function shuffleInUnison(X: number[][], y: number[], rng: () => number): { X: number[][]; y: number[] } {
    const indices = permutation(X.length, rng);
    return {
        X: indices.map((i) => X[i]),
        y: indices.map((i) => y[i]),
    };
}

/** naive dense matrix product a (n x k) * b (k x m) */
export function matMul(a: number[][], b: number[][]): number[][] {
    const n = a.length;
    const k = b.length;
    const m = b[0].length;
    const out: number[][] = [];
    for (let i = 0; i < n; i++) {
        const row: number[] = new Array(m).fill(0);
        for (let p = 0; p < k; p++) {
            const aip = a[i][p];
            if (aip === 0) {
                continue;
            }
            for (let j = 0; j < m; j++) {
                row[j] += aip * b[p][j];
            }
        }
        out.push(row);
    }
    return out;
}

/**
 * Resolves an `nSamples` that is either a total count or an explicit
 * per-group array into per-group counts. When a total is given, it is
 * split as evenly as possible over `nGroups`, earlier groups receiving
 * the remainder (sklearn behaviour).
 */
export function resolveSamplesPerGroup(nSamples: number | number[], nGroups: number): number[] {
    if (Array.isArray(nSamples)) {
        if (nSamples.length !== nGroups) {
            throw new Error(`nSamples array length (${nSamples.length}) must equal the number of groups (${nGroups})`);
        }
        return nSamples.slice();
    }
    const base = Math.floor(nSamples / nGroups);
    const counts = new Array(nGroups).fill(base);
    for (let i = 0; i < nSamples % nGroups; i++) {
        counts[i] += 1;
    }
    return counts;
}
