/**
 * Numerical helpers shared by the mixture-model estimators.
 *
 * Everything here is dependency-free plain math so the fitted state of the
 * estimators stays serializable (no function-valued fields are ever stored).
 */

/** Machine epsilon for float64 (same constant sklearn uses as `np.finfo(float64).eps`). */
export const EPS = 2.220446049250313e-16;

/** Numerically stable log(sum(exp(values))). */
export function logsumexp(values: number[]): number {
    let max = -Infinity;
    for (let i = 0; i < values.length; i++) {
        if (values[i] > max) max = values[i];
    }
    if (max === -Infinity) return -Infinity;
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        sum += Math.exp(values[i] - max);
    }
    return max + Math.log(sum);
}

/**
 * Cholesky decomposition of a symmetric positive-definite matrix.
 * Returns the lower-triangular factor L with A = L·Lᵀ.
 */
export function cholesky(a: number[][]): number[][] {
    const n = a.length;
    const L: number[][] = [];
    for (let i = 0; i < n; i++) L.push(new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            let sum = a[i][j];
            for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
            if (i === j) {
                if (!(sum > 0)) {
                    throw new Error(
                        'Fitting the mixture model failed because some components have ill-defined empirical ' +
                            'covariance (a covariance matrix is not positive definite). Try to decrease the number ' +
                            'of components, or increase regCovar.'
                    );
                }
                L[i][j] = Math.sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }
    return L;
}

/** Solve L·x = b for lower-triangular L (forward substitution). */
export function solveLowerTriangular(L: number[][], b: number[]): number[] {
    const n = L.length;
    const x = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
        let sum = b[i];
        for (let j = 0; j < i; j++) sum -= L[i][j] * x[j];
        x[i] = sum / L[i][i];
    }
    return x;
}

/** Inverse of a symmetric positive-definite matrix via its Cholesky factor. */
export function spdInverse(a: number[][]): number[][] {
    const n = a.length;
    const L = cholesky(a);
    const inv: number[][] = [];
    for (let i = 0; i < n; i++) inv.push(new Array(n).fill(0));
    for (let col = 0; col < n; col++) {
        // forward solve L·y = e_col
        const y = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            let sum = i === col ? 1 : 0;
            for (let j = 0; j < i; j++) sum -= L[i][j] * y[j];
            y[i] = sum / L[i][i];
        }
        // backward solve Lᵀ·x = y
        const x = new Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            let sum = y[i];
            for (let j = i + 1; j < n; j++) sum -= L[j][i] * x[j];
            x[i] = sum / L[i][i];
        }
        for (let i = 0; i < n; i++) inv[i][col] = x[i];
    }
    return inv;
}

// Lanczos approximation coefficients (g = 7, n = 9); gives ~15 significant
// digits for log Γ(x) over the positive reals.
const LANCZOS_G = 7;
const LANCZOS_COEFFS = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];

/** log Γ(x) via the Lanczos approximation (reflection formula for x < 0.5). */
export function logGamma(x: number): number {
    if (Number.isNaN(x)) return NaN;
    if (x < 0.5) {
        // reflection: Γ(x)·Γ(1−x) = π / sin(πx)
        return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x);
    }
    const z = x - 1;
    let acc = LANCZOS_COEFFS[0];
    for (let i = 1; i < LANCZOS_COEFFS.length; i++) {
        acc += LANCZOS_COEFFS[i] / (z + i);
    }
    const t = z + LANCZOS_G + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(acc);
}

/**
 * Digamma function ψ(x) for x > 0: the recurrence ψ(x) = ψ(x+1) − 1/x lifts the
 * argument above 10, then the asymptotic series
 * ψ(x) ≈ ln x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶) + 1/(240x⁸) − 1/(132x¹⁰)
 * is accurate to well below 1e-13.
 */
export function digamma(x: number): number {
    if (!(x > 0)) {
        throw new Error(`digamma is only implemented for positive arguments (got ${x})`);
    }
    let result = 0;
    while (x < 10) {
        result -= 1 / x;
        x += 1;
    }
    const inv = 1 / x;
    const inv2 = inv * inv;
    result +=
        Math.log(x) -
        0.5 * inv -
        inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))));
    return result;
}

/** log Β(a, b) = log Γ(a) + log Γ(b) − log Γ(a + b). */
export function betaln(a: number, b: number): number {
    return logGamma(a) + logGamma(b) - logGamma(a + b);
}
