/**
 * Small deterministic datasets for the estimator-conformance harness.
 * Everything is generated from a fixed-seed LCG so runs are reproducible on
 * every platform.
 */

export interface Dataset {
    X: number[][];
    y: number[];
}

function lcg(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function blob(rand: () => number, center: number[], n: number, spread: number): number[][] {
    const points: number[][] = [];
    for (let i = 0; i < n; i++) {
        points.push(center.map((c) => c + (rand() * 2 - 1) * spread));
    }
    return points;
}

/** Three well-separated 2-D blobs, labels 0/1/2. 36 samples. */
export function blobsDataset(): Dataset {
    const rand = lcg(42);
    const centers = [[0, 0], [6, 6], [0, 8]];
    const X: number[][] = [];
    const y: number[] = [];
    centers.forEach((c, label) => {
        for (const p of blob(rand, c, 12, 1)) {
            X.push(p);
            y.push(label);
        }
    });
    return { X, y };
}

/** Two well-separated 4-D blobs, labels 0/1. 30 samples. */
export function binaryDataset(): Dataset {
    const rand = lcg(7);
    const X: number[][] = [];
    const y: number[] = [];
    for (const p of blob(rand, [0, 0, 1, 1], 15, 1)) { X.push(p); y.push(0); }
    for (const p of blob(rand, [5, 5, 6, 4], 15, 1)) { X.push(p); y.push(1); }
    return { X, y };
}

/** Three-class version of `binaryDataset`. 36 samples. */
export function multiclassDataset(): Dataset {
    const rand = lcg(11);
    const centers = [[0, 0, 0], [5, 5, 0], [0, 5, 5]];
    const X: number[][] = [];
    const y: number[] = [];
    centers.forEach((c, label) => {
        for (const p of blob(rand, c, 12, 1)) { X.push(p); y.push(label); }
    });
    return { X, y };
}

/** Linear target with small deterministic noise. 30 samples, 3 features. */
export function regressionDataset(): Dataset {
    const rand = lcg(23);
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 30; i++) {
        const row = [rand() * 4 - 2, rand() * 4 - 2, rand() * 4 - 2];
        X.push(row);
        y.push(3 * row[0] - 2 * row[1] + 0.5 * row[2] + 1 + (rand() - 0.5) * 0.05);
    }
    return { X, y };
}

/** Non-negative small-integer count features (for Multinomial/Complement NB). */
export function countsDataset(): Dataset {
    const rand = lcg(31);
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 30; i++) {
        const label = i < 15 ? 0 : 1;
        const bias = label === 0 ? [4, 1, 0, 2] : [0, 3, 5, 1];
        X.push(bias.map((b) => Math.floor(rand() * 3) + b));
        y.push(label);
    }
    return { X, y };
}

/** Binary 0/1 features (for BernoulliNB, CategoricalNB, BernoulliRBM). */
export function binaryFeaturesDataset(): Dataset {
    const rand = lcg(53);
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 30; i++) {
        const label = i < 15 ? 0 : 1;
        const probs = label === 0 ? [0.9, 0.8, 0.1, 0.2, 0.7] : [0.1, 0.2, 0.9, 0.8, 0.3];
        X.push(probs.map((p) => (rand() < p ? 1 : 0)));
        y.push(label);
    }
    return { X, y };
}
