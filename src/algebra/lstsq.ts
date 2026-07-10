/**
 * Least squares via Householder QR: min ||Ax - b||_2.
 *
 * Works directly on A (n x p, n >= p) instead of the normal equations, so the
 * condition number is not squared — near-collinear designs stay solvable.
 * Returns false when A is (numerically) rank deficient or n < p; callers
 * should surface an informative error rather than produce garbage.
 */
export function lstsq(A: number[][], b: number[]): number[] | false {
    const n = A.length;
    if (n === 0) return false;
    const p = A[0].length;
    if (p === 0 || n < p) return false;

    const R = A.map(row => row.slice());
    const qtb = b.slice();

    for (let j = 0; j < p; j++) {
        // Householder reflection zeroing column j below the diagonal
        let norm = 0;
        for (let i = j; i < n; i++) norm += R[i][j] * R[i][j];
        norm = Math.sqrt(norm);
        if (norm === 0) {
            R[j][j] = 0;
            continue;
        }
        const alpha = R[j][j] > 0 ? -norm : norm;
        const v: number[] = new Array(n - j);
        v[0] = R[j][j] - alpha;
        for (let i = j + 1; i < n; i++) v[i - j] = R[i][j];
        let vnorm2 = 0;
        for (const x of v) vnorm2 += x * x;
        if (vnorm2 > 0) {
            for (let k = j + 1; k < p; k++) {
                let dot = 0;
                for (let i = j; i < n; i++) dot += v[i - j] * R[i][k];
                const f = (2 * dot) / vnorm2;
                for (let i = j; i < n; i++) R[i][k] -= f * v[i - j];
            }
            let dotb = 0;
            for (let i = j; i < n; i++) dotb += v[i - j] * qtb[i];
            const fb = (2 * dotb) / vnorm2;
            for (let i = j; i < n; i++) qtb[i] -= fb * v[i - j];
        }
        R[j][j] = alpha;
        for (let i = j + 1; i < n; i++) R[i][j] = 0;
    }

    // rank check against the largest diagonal magnitude
    let maxDiag = 0;
    for (let j = 0; j < p; j++) maxDiag = Math.max(maxDiag, Math.abs(R[j][j]));
    if (maxDiag === 0) return false;
    const tol = maxDiag * Math.max(n, p) * Number.EPSILON;
    for (let j = 0; j < p; j++) {
        if (Math.abs(R[j][j]) <= tol) return false;
    }

    // back substitution on R x = Q^T b
    const x = new Array(p).fill(0);
    for (let j = p - 1; j >= 0; j--) {
        let s = qtb[j];
        for (let k = j + 1; k < p; k++) s -= R[j][k] * x[k];
        x[j] = s / R[j][j];
    }
    return x;
}
