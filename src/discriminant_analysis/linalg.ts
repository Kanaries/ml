/**
 * Dense linear-algebra kernels used by the discriminant-analysis estimators.
 *
 * Both routines are Jacobi methods: slower than LAPACK-style algorithms but
 * unconditionally convergent, fully deterministic, and accurate to machine
 * precision for the small/medium matrices this library targets. Unlike the
 * top-k power iteration in `src/algebra/eigen.ts` they return the FULL
 * spectrum, which the sklearn-style tol-based rank cutting requires.
 */

/**
 * Thin singular value decomposition (values + right singular vectors) of an
 * n x p matrix via one-sided (Hestenes) Jacobi rotations.
 *
 * Returns `S` (length min(n, p), descending, all >= 0) and `Vt`
 * (min(n, p) x p, orthonormal rows) such that A ~= U diag(S) Vt. The left
 * singular vectors are not materialized because no caller needs them.
 */
export function jacobiSVD(A: number[][]): { S: number[]; Vt: number[][] } {
    const n = A.length;
    const p = n > 0 ? A[0].length : 0;
    if (n === 0 || p === 0) return { S: [], Vt: [] };

    // column-major working copy of A; rotations orthogonalize these columns
    const cols: number[][] = [];
    for (let j = 0; j < p; j++) {
        const c = new Array<number>(n);
        for (let i = 0; i < n; i++) c[i] = A[i][j];
        cols.push(c);
    }
    // V accumulated as columns: V[j] is the j-th right singular vector
    const V: number[][] = [];
    for (let j = 0; j < p; j++) {
        const v = new Array<number>(p).fill(0);
        v[j] = 1;
        V.push(v);
    }

    const eps = 1e-14;
    const maxSweeps = 100;
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
        let changed = false;
        for (let i = 0; i < p - 1; i++) {
            for (let j = i + 1; j < p; j++) {
                const ci = cols[i];
                const cj = cols[j];
                let alpha = 0;
                let beta = 0;
                let gamma = 0;
                for (let k = 0; k < n; k++) {
                    alpha += ci[k] * ci[k];
                    beta += cj[k] * cj[k];
                    gamma += ci[k] * cj[k];
                }
                if (gamma === 0 || Math.abs(gamma) <= eps * Math.sqrt(alpha * beta)) continue;
                changed = true;
                const zeta = (beta - alpha) / (2 * gamma);
                const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
                const c = 1 / Math.sqrt(1 + t * t);
                const s = c * t;
                for (let k = 0; k < n; k++) {
                    const tmp = ci[k];
                    ci[k] = c * tmp - s * cj[k];
                    cj[k] = s * tmp + c * cj[k];
                }
                const vi = V[i];
                const vj = V[j];
                for (let k = 0; k < p; k++) {
                    const tmp = vi[k];
                    vi[k] = c * tmp - s * vj[k];
                    vj[k] = s * tmp + c * vj[k];
                }
            }
        }
        if (!changed) break;
    }

    const norms = cols.map((c) => {
        let s = 0;
        for (const x of c) s += x * x;
        return Math.sqrt(s);
    });
    const order = norms.map((_, j) => j).sort((a, b) => norms[b] - norms[a]);
    const k = Math.min(n, p);
    const S: number[] = [];
    const Vt: number[][] = [];
    for (let r = 0; r < k; r++) {
        S.push(norms[order[r]]);
        Vt.push(V[order[r]].slice());
    }
    return { S, Vt };
}

/**
 * Full eigendecomposition of a symmetric matrix via the classical two-sided
 * Jacobi eigenvalue algorithm. Returns all eigenpairs sorted by eigenvalue
 * descending; `vectors[i]` is the (unit-norm) eigenvector for `values[i]`.
 */
export function symmetricEigDecomposition(A: number[][]): { values: number[]; vectors: number[][] } {
    const n = A.length;
    if (n === 0) return { values: [], vectors: [] };
    const B = A.map((row) => row.slice());
    const V: number[][] = [];
    for (let i = 0; i < n; i++) {
        const row = new Array<number>(n).fill(0);
        row[i] = 1;
        V.push(row);
    }

    let normF = 0;
    for (const row of A) for (const x of row) normF += x * x;
    normF = Math.sqrt(normF);
    const stop = 1e-14 * (normF || 1);
    const maxSweeps = 100;

    for (let sweep = 0; sweep < maxSweeps; sweep++) {
        let off = 0;
        for (let i = 0; i < n - 1; i++) {
            for (let j = i + 1; j < n; j++) off += B[i][j] * B[i][j];
        }
        if (Math.sqrt(off) <= stop) break;
        for (let i = 0; i < n - 1; i++) {
            for (let j = i + 1; j < n; j++) {
                const apq = B[i][j];
                if (Math.abs(apq) <= stop / n) continue;
                const tau = (B[j][j] - B[i][i]) / (2 * apq);
                const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
                const c = 1 / Math.sqrt(1 + t * t);
                const s = c * t;
                const bii = B[i][i];
                const bjj = B[j][j];
                B[i][i] = bii - t * apq;
                B[j][j] = bjj + t * apq;
                B[i][j] = 0;
                B[j][i] = 0;
                for (let k = 0; k < n; k++) {
                    if (k === i || k === j) continue;
                    const bki = B[k][i];
                    const bkj = B[k][j];
                    B[k][i] = c * bki - s * bkj;
                    B[i][k] = B[k][i];
                    B[k][j] = s * bki + c * bkj;
                    B[j][k] = B[k][j];
                }
                for (let k = 0; k < n; k++) {
                    const vki = V[k][i];
                    const vkj = V[k][j];
                    V[k][i] = c * vki - s * vkj;
                    V[k][j] = s * vki + c * vkj;
                }
            }
        }
    }

    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => B[b][b] - B[a][a]);
    const values: number[] = [];
    const vectors: number[][] = [];
    for (const idx of order) {
        values.push(B[idx][idx]);
        const v = new Array<number>(n);
        for (let k = 0; k < n; k++) v[k] = V[k][idx];
        vectors.push(v);
    }
    return { values, vectors };
}

/**
 * Cholesky factorization A = L L^T for a symmetric positive-definite matrix.
 * Returns the lower-triangular factor, or null when A is not (numerically)
 * positive definite so callers can raise a domain-specific error.
 */
export function cholesky(A: number[][]): number[][] | null {
    const n = A.length;
    let scale = 0;
    for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(A[i][i]));
    const tiny = Number.EPSILON * n * (scale || 1);
    const L: number[][] = [];
    for (let i = 0; i < n; i++) L.push(new Array<number>(n).fill(0));
    for (let j = 0; j < n; j++) {
        let d = A[j][j];
        for (let k = 0; k < j; k++) d -= L[j][k] * L[j][k];
        if (!(d > tiny)) return null;
        L[j][j] = Math.sqrt(d);
        for (let i = j + 1; i < n; i++) {
            let s = A[i][j];
            for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
            L[i][j] = s / L[j][j];
        }
    }
    return L;
}

/** Solve L x = b for lower-triangular L (forward substitution). */
export function solveLower(L: number[][], b: number[]): number[] {
    const n = L.length;
    const x = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
        let s = b[i];
        for (let k = 0; k < i; k++) s -= L[i][k] * x[k];
        x[i] = s / L[i][i];
    }
    return x;
}

/** Solve L^T x = b for lower-triangular L (backward substitution). */
export function solveLowerTransposed(L: number[][], b: number[]): number[] {
    const n = L.length;
    const x = new Array<number>(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
        let s = b[i];
        for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
        x[i] = s / L[i][i];
    }
    return x;
}
