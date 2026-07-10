import { createRandomGenerator } from '../utils/random';

export function dot(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        s += a[i] * b[i];
    }
    return s;
}

export function matVecMul(A: number[][], v: number[]): number[] {
    return A.map(row => dot(row, v));
}

export function outer(v1: number[], v2: number[]): number[][] {
    const res: number[][] = [];
    for (let i = 0; i < v1.length; i++) {
        res.push([]);
        for (let j = 0; j < v2.length; j++) {
            res[i].push(v1[i] * v2[j]);
        }
    }
    return res;
}

export function norm2(v: number[]): number {
    return Math.sqrt(dot(v, v));
}

/**
 * Returns v / ||v||, or null when ||v|| is (numerically) zero so callers
 * can handle degenerate vectors explicitly instead of propagating NaN.
 */
export function normalizeOrNull(v: number[]): number[] | null {
    const norm = norm2(v);
    if (!Number.isFinite(norm) || norm === 0) {
        return null;
    }
    return v.map(x => x / norm);
}

/**
 * Returns v / ||v||. Throws on a zero (or non-finite) norm rather than
 * silently producing NaN entries.
 */
export function normalize(v: number[]): number[] {
    const normalized = normalizeOrNull(v);
    if (normalized === null) {
        throw new Error('cannot normalize a zero-norm vector');
    }
    return normalized;
}

export interface SymmetricEigenOptions {
    maxIter?: number;
    tol?: number;
    seed?: number;
}

function randomVector(n: number, rand: () => number): number[] {
    const v: number[] = [];
    for (let i = 0; i < n; i++) {
        v.push(rand() * 2 - 1);
    }
    return v;
}

function projectOutInPlace(v: number[], basis: number[][]): void {
    for (const q of basis) {
        const c = dot(v, q);
        for (let i = 0; i < v.length; i++) {
            v[i] -= c * q[i];
        }
    }
}

function matrixScale(A: number[][]): number {
    let scale = 0;
    for (const row of A) {
        for (const x of row) {
            const abs = Math.abs(x);
            if (abs > scale) {
                scale = abs;
            }
        }
    }
    return scale;
}

/**
 * Deterministic unit vector orthogonal to every vector in `basis`: the first
 * standard basis vector with a non-negligible residual after projection.
 */
function orthogonalComplementVector(n: number, basis: number[][]): number[] {
    for (let i = 0; i < n; i++) {
        const e = new Array(n).fill(0);
        e[i] = 1;
        projectOutInPlace(e, basis);
        const normalized = normalizeOrNull(e);
        if (normalized !== null && norm2(e) > 1e-8) {
            return normalized;
        }
    }
    // basis already spans the space; callers only reach this when k <= n
    throw new Error('no orthogonal complement direction available');
}

/**
 * Computes the top-k eigenpairs of a symmetric matrix via power iteration
 * with deflation. Compared to a naive power method it adds:
 * - seeded pseudo-random initialization (deterministic, but avoids the
 *   all-ones vector stalling when (1, ..., 1) is an exact eigenvector);
 * - Gram-Schmidt re-orthogonalization against previously extracted vectors
 *   on every iteration, so components stay orthogonal even for
 *   near-degenerate spectra;
 * - a convergence criterion (vector change < tol, up to sign) with early
 *   stopping instead of a fixed iteration count;
 * - zero-norm guards: when the deflated matrix annihilates the search space
 *   the eigenvalue is reported as exactly 0 with a deterministic vector from
 *   the orthogonal complement, never NaN.
 */
export function symmetricEigen(
    A: number[][],
    k: number,
    options: SymmetricEigenOptions = {}
): { values: number[]; vectors: number[][] } {
    const { maxIter = 1000, tol = 1e-10, seed = 1 } = options;
    const n = A.length;
    const nEigen = Math.min(k, n);
    const rand = createRandomGenerator(seed);
    const work = A.map(row => row.slice());
    const originalScale = matrixScale(A);
    const zeroThreshold = 1e-12 * (originalScale || 1);
    const maxRestarts = 5;

    const values: number[] = [];
    const vectors: number[][] = [];

    for (let c = 0; c < nEigen; c++) {
        let vector: number[] | null = null;

        if (matrixScale(work) > zeroThreshold) {
            for (let attempt = 0; attempt < maxRestarts && vector === null; attempt++) {
                const init = randomVector(n, rand);
                projectOutInPlace(init, vectors);
                let v = normalizeOrNull(init);
                if (v === null) {
                    continue;
                }
                for (let iter = 0; iter < maxIter; iter++) {
                    const Av = matVecMul(work, v);
                    projectOutInPlace(Av, vectors);
                    if (norm2(Av) <= zeroThreshold) {
                        // v is (numerically) in the null space of the deflated
                        // matrix: restart from a different random vector
                        v = null;
                        break;
                    }
                    const next = normalizeOrNull(Av);
                    if (next === null) {
                        v = null;
                        break;
                    }
                    let diffPlus = 0;
                    let diffMinus = 0;
                    for (let i = 0; i < n; i++) {
                        diffPlus += (next[i] - v[i]) ** 2;
                        diffMinus += (next[i] + v[i]) ** 2;
                    }
                    const diff = Math.sqrt(Math.min(diffPlus, diffMinus));
                    v = next;
                    if (diff < tol) {
                        break;
                    }
                }
                vector = v;
            }
        }

        if (vector === null) {
            // deflated matrix is zero on the remaining subspace: report a
            // zero eigenvalue with a deterministic orthonormal direction
            vectors.push(orthogonalComplementVector(n, vectors));
            values.push(0);
            continue;
        }

        const value = dot(vector, matVecMul(work, vector));
        values.push(value);
        vectors.push(vector);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                work[i][j] -= value * vector[i] * vector[j];
            }
        }
    }

    return { values, vectors };
}
