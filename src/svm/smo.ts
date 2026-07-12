/**
 * Sequential Minimal Optimization solvers for kernel SVMs, following
 * libsvm's Solver / Solver_NU (Platt's SMO with Keerthi's maximal
 * violating pair working-set selection).
 *
 * Both solvers work on a binary problem with labels y in {+1, -1} and
 * return unsigned dual coefficients alpha plus an intercept b so that the
 * decision function is f(x) = sum_i alpha_i * y_i * K(x_i, x) + b.
 */

export type KernelType = 'linear' | 'rbf' | 'poly' | 'sigmoid';

export interface KernelConfig {
    kernel: KernelType;
    /** resolved numeric gamma ('scale'/'auto' must be resolved by the caller) */
    gamma: number;
    degree: number;
    coef0: number;
}

const TAU = 1e-12;

/**
 * Resolve sklearn-style gamma: 'scale' = 1/(n_features * Var(X)) over ALL
 * entries of X, 'auto' = 1/n_features, numbers pass through.
 */
export function resolveGammaValue(gamma: number | 'scale' | 'auto', X: number[][]): number {
    if (typeof gamma === 'number') {
        return gamma;
    }
    const nFeatures = X[0].length;
    if (gamma === 'auto') {
        return 1 / nFeatures;
    }
    let sum = 0;
    let count = 0;
    for (const row of X) for (const v of row) { sum += v; count++; }
    const mean = sum / count;
    let varSum = 0;
    for (const row of X) for (const v of row) varSum += (v - mean) ** 2;
    const variance = varSum / count;
    return variance > 0 ? 1 / (nFeatures * variance) : 1;
}

export function kernelFunction(x1: number[], x2: number[], cfg: KernelConfig): number {
    if (cfg.kernel === 'rbf') {
        let sum = 0;
        for (let i = 0; i < x1.length; i++) {
            const d = x1[i] - x2[i];
            sum += d * d;
        }
        return Math.exp(-cfg.gamma * sum);
    }
    let dot = 0;
    for (let i = 0; i < x1.length; i++) {
        dot += x1[i] * x2[i];
    }
    switch (cfg.kernel) {
        case 'linear':
            return dot;
        case 'poly':
            return Math.pow(cfg.gamma * dot + cfg.coef0, cfg.degree);
        case 'sigmoid':
            return Math.tanh(cfg.gamma * dot + cfg.coef0);
        default:
            throw new Error(`unknown kernel: ${cfg.kernel}`);
    }
}

/**
 * Row cache for the kernel matrix. Problems up to ~2000 samples keep the
 * full n^2 matrix in memory; larger ones fall back to a bounded FIFO row
 * cache so memory stays around 256MB worst case.
 */
export class KernelMatrix {
    public readonly n: number;
    private X: number[][];
    private cfg: KernelConfig;
    private rows: Map<number, Float64Array>;
    private maxRows: number;
    private diagonal: Float64Array;

    constructor(X: number[][], cfg: KernelConfig) {
        this.X = X;
        this.cfg = cfg;
        this.n = X.length;
        this.rows = new Map();
        this.maxRows = this.n <= 2048 ? this.n : Math.max(2, Math.floor((256 << 20) / 8 / this.n));
        this.diagonal = new Float64Array(this.n);
        for (let i = 0; i < this.n; i++) {
            this.diagonal[i] = kernelFunction(X[i], X[i], cfg);
        }
    }

    public diag(i: number): number {
        return this.diagonal[i];
    }

    public getRow(i: number): Float64Array {
        const cached = this.rows.get(i);
        if (cached) {
            return cached;
        }
        const row = new Float64Array(this.n);
        for (let j = 0; j < this.n; j++) {
            row[j] = kernelFunction(this.X[i], this.X[j], this.cfg);
        }
        if (this.rows.size >= this.maxRows) {
            // FIFO eviction: Map iterates in insertion order
            const oldest = this.rows.keys().next().value as number;
            this.rows.delete(oldest);
        }
        this.rows.set(i, row);
        return row;
    }
}

export interface SMOSolution {
    /** unsigned dual coefficients (0 <= alpha_i); dual coef of sample i is alpha_i * y_i */
    alpha: number[];
    /** intercept: f(x) = sum_i alpha_i * y_i * K(x_i, x) + b */
    b: number;
    iterations: number;
    converged: boolean;
}

function resolveMaxIter(maxIter: number, n: number): number {
    // libsvm-style guard: -1 means "until convergence" but we still cap the
    // update count to avoid a numerically-stuck infinite loop
    return maxIter < 0 ? Math.max(10000000, 100 * n) : maxIter;
}

/**
 * Analytic two-variable update shared by both solvers (libsvm Solver::solve),
 * with per-sample upper bounds Ci, Cj. Mutates alpha and returns the deltas.
 */
function updatePair(
    alpha: Float64Array,
    y: number[],
    i: number,
    j: number,
    Ki: Float64Array,
    Kj: Float64Array,
    G: Float64Array,
    Ci: number,
    Cj: number
): { dAi: number; dAj: number } {
    const oldAi = alpha[i];
    const oldAj = alpha[j];
    // libsvm's quad_coef = QD[i] + QD[j] +/- 2*Q_i[j]; with Q_ij = y_i y_j K_ij
    // both label cases reduce to ||phi(x_i) - phi(x_j)||^2
    let quad = Ki[i] + Kj[j] - 2 * Ki[j];
    if (quad <= 0) {
        quad = TAU;
    }
    if (y[i] !== y[j]) {
        const delta = (-G[i] - G[j]) / quad;
        const diff = alpha[i] - alpha[j];
        alpha[i] += delta;
        alpha[j] += delta;
        if (diff > 0) {
            if (alpha[j] < 0) {
                alpha[j] = 0;
                alpha[i] = diff;
            }
        } else {
            if (alpha[i] < 0) {
                alpha[i] = 0;
                alpha[j] = -diff;
            }
        }
        if (diff > Ci - Cj) {
            if (alpha[i] > Ci) {
                alpha[i] = Ci;
                alpha[j] = Ci - diff;
            }
        } else {
            if (alpha[j] > Cj) {
                alpha[j] = Cj;
                alpha[i] = Cj + diff;
            }
        }
    } else {
        const delta = (G[i] - G[j]) / quad;
        const sum = alpha[i] + alpha[j];
        alpha[i] -= delta;
        alpha[j] += delta;
        if (sum > Ci) {
            if (alpha[i] > Ci) {
                alpha[i] = Ci;
                alpha[j] = sum - Ci;
            }
        } else {
            if (alpha[j] < 0) {
                alpha[j] = 0;
                alpha[i] = sum;
            }
        }
        if (sum > Cj) {
            if (alpha[j] > Cj) {
                alpha[j] = Cj;
                alpha[i] = sum - Cj;
            }
        } else {
            if (alpha[i] < 0) {
                alpha[i] = 0;
                alpha[j] = sum;
            }
        }
    }
    return { dAi: alpha[i] - oldAi, dAj: alpha[j] - oldAj };
}

function updateGradient(
    G: Float64Array,
    y: number[],
    i: number,
    j: number,
    Ki: Float64Array,
    Kj: Float64Array,
    dAi: number,
    dAj: number
): void {
    const yi = y[i];
    const yj = y[j];
    for (let t = 0; t < G.length; t++) {
        G[t] += y[t] * (yi * Ki[t] * dAi + yj * Kj[t] * dAj);
    }
}

/**
 * C-SVC dual:
 *   min 1/2 a'Qa - e'a  s.t.  0 <= a_i <= C,  y'a = 0,   Q_ij = y_i y_j K_ij
 */
export function solveCSVC(
    K: KernelMatrix,
    y: number[],
    C: number,
    tol: number,
    maxIter: number
): SMOSolution {
    const n = K.n;
    const alpha = new Float64Array(n);
    const G = new Float64Array(n).fill(-1);
    const limit = resolveMaxIter(maxIter, n);
    let iter = 0;
    let converged = false;
    while (iter < limit) {
        // maximal violating pair: i in I_up maximizing -y_t G_t,
        // j in I_low minimizing -y_t G_t
        let i = -1;
        let gMax = -Infinity;
        let j = -1;
        let gMin = Infinity;
        for (let t = 0; t < n; t++) {
            const v = -y[t] * G[t];
            if (y[t] === 1 ? alpha[t] < C : alpha[t] > 0) {
                if (v > gMax) {
                    gMax = v;
                    i = t;
                }
            }
            if (y[t] === 1 ? alpha[t] > 0 : alpha[t] < C) {
                if (v < gMin) {
                    gMin = v;
                    j = t;
                }
            }
        }
        if (i === -1 || j === -1 || gMax - gMin < tol) {
            converged = true;
            break;
        }
        const Ki = K.getRow(i);
        const Kj = K.getRow(j);
        const { dAi, dAj } = updatePair(alpha, y, i, j, Ki, Kj, G, C, C);
        updateGradient(G, y, i, j, Ki, Kj, dAi, dAj);
        iter++;
    }
    // rho as in libsvm calculate_rho; b = -rho
    let ub = Infinity;
    let lb = -Infinity;
    let sumFree = 0;
    let nFree = 0;
    for (let t = 0; t < n; t++) {
        const yG = y[t] * G[t];
        if (alpha[t] >= C) {
            if (y[t] === -1) {
                ub = Math.min(ub, yG);
            } else {
                lb = Math.max(lb, yG);
            }
        } else if (alpha[t] <= 0) {
            if (y[t] === 1) {
                ub = Math.min(ub, yG);
            } else {
                lb = Math.max(lb, yG);
            }
        } else {
            nFree++;
            sumFree += yG;
        }
    }
    const rho = nFree > 0 ? sumFree / nFree : (ub + lb) / 2;
    return { alpha: Array.from(alpha), b: -rho, iterations: iter, converged };
}

/**
 * nu-SVC dual (libsvm Solver_NU):
 *   min 1/2 a'Qa  s.t.  0 <= a_i <= 1,
 *   sum_{y_i=+1} a_i = sum_{y_i=-1} a_i = nu * n / 2.
 * The working pair is always selected within one class so both equality
 * constraints stay satisfied. The returned alphas and b are already scaled
 * by 1/r (libsvm's post-solve normalization), matching sklearn's
 * dual_coef_ / intercept_ / decision_function conventions.
 */
export function solveNuSVC(
    K: KernelMatrix,
    y: number[],
    nu: number,
    tol: number,
    maxIter: number
): SMOSolution {
    const n = K.n;
    const alpha = new Float64Array(n);
    let sumPos = (nu * n) / 2;
    let sumNeg = (nu * n) / 2;
    for (let t = 0; t < n; t++) {
        if (y[t] === 1) {
            alpha[t] = Math.min(1, sumPos);
            sumPos -= alpha[t];
        } else {
            alpha[t] = Math.min(1, sumNeg);
            sumNeg -= alpha[t];
        }
    }
    // G_t = sum_s Q_ts a_s (the linear term p is zero for nu-SVC)
    const G = new Float64Array(n);
    for (let s = 0; s < n; s++) {
        if (alpha[s] > 0) {
            const Ks = K.getRow(s);
            const coef = y[s] * alpha[s];
            for (let t = 0; t < n; t++) {
                G[t] += y[t] * coef * Ks[t];
            }
        }
    }
    const limit = resolveMaxIter(maxIter, n);
    let iter = 0;
    let converged = false;
    while (iter < limit) {
        // violating pair within each class, take the class with the larger gap
        let gMaxP = -Infinity;
        let gMaxP2 = -Infinity;
        let ip = -1;
        let jp = -1;
        let gMaxN = -Infinity;
        let gMaxN2 = -Infinity;
        let iN = -1;
        let jN = -1;
        for (let t = 0; t < n; t++) {
            if (y[t] === 1) {
                if (alpha[t] < 1 && -G[t] > gMaxP) {
                    gMaxP = -G[t];
                    ip = t;
                }
                if (alpha[t] > 0 && G[t] > gMaxP2) {
                    gMaxP2 = G[t];
                    jp = t;
                }
            } else {
                if (alpha[t] > 0 && G[t] > gMaxN) {
                    gMaxN = G[t];
                    iN = t;
                }
                if (alpha[t] < 1 && -G[t] > gMaxN2) {
                    gMaxN2 = -G[t];
                    jN = t;
                }
            }
        }
        const vioP = gMaxP + gMaxP2;
        const vioN = gMaxN + gMaxN2;
        if (Math.max(vioP, vioN) < tol) {
            converged = true;
            break;
        }
        const i = vioP > vioN ? ip : iN;
        const j = vioP > vioN ? jp : jN;
        const Ki = K.getRow(i);
        const Kj = K.getRow(j);
        const { dAi, dAj } = updatePair(alpha, y, i, j, Ki, Kj, G, 1, 1);
        updateGradient(G, y, i, j, Ki, Kj, dAi, dAj);
        iter++;
    }
    // libsvm Solver_NU::calculate_rho: r1/r2 are the average gradients of the
    // free vectors in each class; rho = (r1 - r2) / 2, r = (r1 + r2) / 2
    let nFree1 = 0;
    let sumFree1 = 0;
    let ub1 = Infinity;
    let lb1 = -Infinity;
    let nFree2 = 0;
    let sumFree2 = 0;
    let ub2 = Infinity;
    let lb2 = -Infinity;
    for (let t = 0; t < n; t++) {
        if (y[t] === 1) {
            if (alpha[t] >= 1) {
                lb1 = Math.max(lb1, G[t]);
            } else if (alpha[t] <= 0) {
                ub1 = Math.min(ub1, G[t]);
            } else {
                nFree1++;
                sumFree1 += G[t];
            }
        } else {
            if (alpha[t] >= 1) {
                lb2 = Math.max(lb2, G[t]);
            } else if (alpha[t] <= 0) {
                ub2 = Math.min(ub2, G[t]);
            } else {
                nFree2++;
                sumFree2 += G[t];
            }
        }
    }
    const r1 = nFree1 > 0 ? sumFree1 / nFree1 : (ub1 + lb1) / 2;
    const r2 = nFree2 > 0 ? sumFree2 / nFree2 : (ub2 + lb2) / 2;
    const r = (r1 + r2) / 2;
    const rho = (r1 - r2) / 2;
    // normalize by r so the margin is 1, like libsvm/sklearn expose it
    const scale = Math.abs(r) > TAU ? 1 / r : 1;
    const alphaOut = new Array<number>(n);
    for (let t = 0; t < n; t++) {
        alphaOut[t] = alpha[t] * scale;
    }
    return { alpha: alphaOut, b: -rho * scale, iterations: iter, converged };
}

// ---------------------------------------------------------------------------
// Generic solvers (libsvm Solver / Solver_NU with an arbitrary linear term,
// initial point and upper bound) backing the SVR / nu-SVR / one-class duals.
// solveCSVC / solveNuSVC above are intentionally left as-is.
// ---------------------------------------------------------------------------

interface GenericSolution {
    alpha: Float64Array;
    rho: number;
    iterations: number;
    converged: boolean;
}

/**
 * libsvm Solver: min 1/2 a'Qa + p'a  s.t. 0 <= a_i <= C, y'a = 0, with
 * Q_ij = y_i y_j K_ij. `getRow(i)` returns the i-th row of the *kernel*
 * matrix (unsigned, length l). `alphaInit` must be feasible; the equality
 * constraint value it fixes is preserved by every pairwise update.
 */
function solveGenericC(
    l: number,
    y: number[],
    p: number[],
    C: number,
    getRow: (i: number) => Float64Array,
    tol: number,
    maxIter: number,
    alphaInit?: Float64Array
): GenericSolution {
    const alpha = alphaInit ? alphaInit.slice() : new Float64Array(l);
    const G = Float64Array.from(p);
    if (alphaInit) {
        for (let s = 0; s < l; s++) {
            if (alpha[s] !== 0) {
                const Ks = getRow(s);
                const coef = y[s] * alpha[s];
                for (let t = 0; t < l; t++) {
                    G[t] += y[t] * coef * Ks[t];
                }
            }
        }
    }
    const limit = resolveMaxIter(maxIter, l);
    let iter = 0;
    let converged = false;
    while (iter < limit) {
        // maximal violating pair, exactly as in solveCSVC
        let i = -1;
        let gMax = -Infinity;
        let j = -1;
        let gMin = Infinity;
        for (let t = 0; t < l; t++) {
            const v = -y[t] * G[t];
            if (y[t] === 1 ? alpha[t] < C : alpha[t] > 0) {
                if (v > gMax) {
                    gMax = v;
                    i = t;
                }
            }
            if (y[t] === 1 ? alpha[t] > 0 : alpha[t] < C) {
                if (v < gMin) {
                    gMin = v;
                    j = t;
                }
            }
        }
        if (i === -1 || j === -1 || gMax - gMin < tol) {
            converged = true;
            break;
        }
        const Ki = getRow(i);
        const Kj = getRow(j);
        const { dAi, dAj } = updatePair(alpha, y, i, j, Ki, Kj, G, C, C);
        updateGradient(G, y, i, j, Ki, Kj, dAi, dAj);
        iter++;
    }
    // libsvm Solver::calculate_rho
    let ub = Infinity;
    let lb = -Infinity;
    let sumFree = 0;
    let nFree = 0;
    for (let t = 0; t < l; t++) {
        const yG = y[t] * G[t];
        if (alpha[t] >= C) {
            if (y[t] === -1) {
                ub = Math.min(ub, yG);
            } else {
                lb = Math.max(lb, yG);
            }
        } else if (alpha[t] <= 0) {
            if (y[t] === 1) {
                ub = Math.min(ub, yG);
            } else {
                lb = Math.max(lb, yG);
            }
        } else {
            nFree++;
            sumFree += yG;
        }
    }
    const rho = nFree > 0 ? sumFree / nFree : (ub + lb) / 2;
    return { alpha, rho, iterations: iter, converged };
}

/**
 * libsvm Solver_NU: same problem as solveGenericC but with the two separate
 * equality constraints sum_{y_i=+1} a_i = const and sum_{y_i=-1} a_i = const
 * fixed by `alphaInit`. The working pair is always chosen within one class.
 * Returns rho = (r1 - r2)/2 and r = (r1 + r2)/2 (libsvm's si->rho / si->r),
 * unscaled — nu-SVC's 1/r normalization does not apply to nu-SVR.
 */
function solveGenericNu(
    l: number,
    y: number[],
    p: number[],
    C: number,
    getRow: (i: number) => Float64Array,
    tol: number,
    maxIter: number,
    alphaInit: Float64Array
): GenericSolution & { r: number } {
    const alpha = alphaInit.slice();
    const G = Float64Array.from(p);
    for (let s = 0; s < l; s++) {
        if (alpha[s] !== 0) {
            const Ks = getRow(s);
            const coef = y[s] * alpha[s];
            for (let t = 0; t < l; t++) {
                G[t] += y[t] * coef * Ks[t];
            }
        }
    }
    const limit = resolveMaxIter(maxIter, l);
    let iter = 0;
    let converged = false;
    while (iter < limit) {
        // violating pair within each class, take the class with the larger gap
        let gMaxP = -Infinity;
        let gMaxP2 = -Infinity;
        let ip = -1;
        let jp = -1;
        let gMaxN = -Infinity;
        let gMaxN2 = -Infinity;
        let iN = -1;
        let jN = -1;
        for (let t = 0; t < l; t++) {
            if (y[t] === 1) {
                if (alpha[t] < C && -G[t] > gMaxP) {
                    gMaxP = -G[t];
                    ip = t;
                }
                if (alpha[t] > 0 && G[t] > gMaxP2) {
                    gMaxP2 = G[t];
                    jp = t;
                }
            } else {
                if (alpha[t] > 0 && G[t] > gMaxN) {
                    gMaxN = G[t];
                    iN = t;
                }
                if (alpha[t] < C && -G[t] > gMaxN2) {
                    gMaxN2 = -G[t];
                    jN = t;
                }
            }
        }
        const vioP = gMaxP + gMaxP2;
        const vioN = gMaxN + gMaxN2;
        if (Math.max(vioP, vioN) < tol) {
            converged = true;
            break;
        }
        const i = vioP > vioN ? ip : iN;
        const j = vioP > vioN ? jp : jN;
        const Ki = getRow(i);
        const Kj = getRow(j);
        const { dAi, dAj } = updatePair(alpha, y, i, j, Ki, Kj, G, C, C);
        updateGradient(G, y, i, j, Ki, Kj, dAi, dAj);
        iter++;
    }
    // libsvm Solver_NU::calculate_rho
    let nFree1 = 0;
    let sumFree1 = 0;
    let ub1 = Infinity;
    let lb1 = -Infinity;
    let nFree2 = 0;
    let sumFree2 = 0;
    let ub2 = Infinity;
    let lb2 = -Infinity;
    for (let t = 0; t < l; t++) {
        if (y[t] === 1) {
            if (alpha[t] >= C) {
                lb1 = Math.max(lb1, G[t]);
            } else if (alpha[t] <= 0) {
                ub1 = Math.min(ub1, G[t]);
            } else {
                nFree1++;
                sumFree1 += G[t];
            }
        } else {
            if (alpha[t] >= C) {
                lb2 = Math.max(lb2, G[t]);
            } else if (alpha[t] <= 0) {
                ub2 = Math.min(ub2, G[t]);
            } else {
                nFree2++;
                sumFree2 += G[t];
            }
        }
    }
    const r1 = nFree1 > 0 ? sumFree1 / nFree1 : (ub1 + lb1) / 2;
    const r2 = nFree2 > 0 ? sumFree2 / nFree2 : (ub2 + lb2) / 2;
    return { alpha, rho: (r1 - r2) / 2, r: (r1 + r2) / 2, iterations: iter, converged };
}

/**
 * Row provider for the doubled SVR problem: index i of the 2n-variable
 * problem maps to training sample i mod n, so row_i[t] = K(i mod n, t mod n).
 * Rows for i and i+n are identical, so the cache is keyed by i mod n.
 */
function makeDoubledRows(K: KernelMatrix): (i: number) => Float64Array {
    const n = K.n;
    const cache = new Map<number, Float64Array>();
    const maxRows = n <= 1024 ? n : Math.max(2, Math.floor((256 << 20) / 8 / (2 * n)));
    return (i: number) => {
        const base = i % n;
        const hit = cache.get(base);
        if (hit) {
            return hit;
        }
        const row = K.getRow(base);
        const ext = new Float64Array(2 * n);
        ext.set(row, 0);
        ext.set(row, n);
        if (cache.size >= maxRows) {
            const oldest = cache.keys().next().value as number;
            cache.delete(oldest);
        }
        cache.set(base, ext);
        return ext;
    };
}

export interface SVRSolution {
    /** signed dual coefficients beta_i = alpha_i - alpha*_i; f(x) = sum_i beta_i K(x_i, x) + b */
    coef: number[];
    b: number;
    iterations: number;
    converged: boolean;
}

/**
 * epsilon-SVR dual (libsvm solve_epsilon_svr), as a 2n-variable C-problem:
 * variables z = [alpha; alpha*], labels y2 = [+1...; -1...], linear term
 * p_i = epsilon - y_i (alpha part) / epsilon + y_i (alpha* part),
 * bounds 0 <= z <= C, constraint sum(alpha) - sum(alpha*) = 0.
 */
export function solveEpsilonSVR(
    K: KernelMatrix,
    y: number[],
    C: number,
    epsilon: number,
    tol: number,
    maxIter: number
): SVRSolution {
    const n = K.n;
    const l = 2 * n;
    const y2 = new Array<number>(l);
    const p = new Array<number>(l);
    for (let i = 0; i < n; i++) {
        y2[i] = 1;
        p[i] = epsilon - y[i];
        y2[i + n] = -1;
        p[i + n] = epsilon + y[i];
    }
    const sol = solveGenericC(l, y2, p, C, makeDoubledRows(K), tol, maxIter);
    const coef = new Array<number>(n);
    for (let i = 0; i < n; i++) {
        coef[i] = sol.alpha[i] - sol.alpha[i + n];
    }
    return { coef, b: -sol.rho, iterations: sol.iterations, converged: sol.converged };
}

/**
 * nu-SVR dual (libsvm solve_nu_svr): epsilon is replaced by the constraint
 * sum(alpha) + sum(alpha*) <= C * nu * n, handled by starting at
 * alpha_i = alpha*_i = min(remaining, C) with a total budget of C*nu*n/2 per
 * side and letting Solver_NU keep both per-class sums fixed. The linear term
 * is p_i = -y_i / +y_i. The effective tube width is epsilon = -r.
 */
export function solveNuSVR(
    K: KernelMatrix,
    y: number[],
    C: number,
    nu: number,
    tol: number,
    maxIter: number
): SVRSolution & { epsilon: number } {
    const n = K.n;
    const l = 2 * n;
    const y2 = new Array<number>(l);
    const p = new Array<number>(l);
    const alpha0 = new Float64Array(l);
    let budget = (C * nu * n) / 2;
    for (let i = 0; i < n; i++) {
        const a = Math.min(budget, C);
        alpha0[i] = a;
        alpha0[i + n] = a;
        budget -= a;
        y2[i] = 1;
        p[i] = -y[i];
        y2[i + n] = -1;
        p[i + n] = y[i];
    }
    const sol = solveGenericNu(l, y2, p, C, makeDoubledRows(K), tol, maxIter, alpha0);
    const coef = new Array<number>(n);
    for (let i = 0; i < n; i++) {
        coef[i] = sol.alpha[i] - sol.alpha[i + n];
    }
    return { coef, b: -sol.rho, epsilon: -sol.r, iterations: sol.iterations, converged: sol.converged };
}

export interface OneClassSolution {
    /** unsigned dual coefficients; decision f(x) = sum_i alpha_i K(x_i, x) - rho */
    alpha: number[];
    rho: number;
    iterations: number;
    converged: boolean;
}

/**
 * One-class dual (Schölkopf, libsvm solve_one_class):
 *   min 1/2 a'Ka  s.t.  0 <= a_i <= 1,  sum(a) = nu * n
 * (libsvm's scaling of the 0 <= a <= 1/(nu*n), sum(a) = 1 formulation by
 * nu*n — identical decision boundary, and matching sklearn's dual_coef_).
 * The equality constraint is fixed by the feasible starting point.
 */
export function solveOneClass(
    K: KernelMatrix,
    nu: number,
    tol: number,
    maxIter: number
): OneClassSolution {
    const n = K.n;
    const alpha0 = new Float64Array(n);
    const nBound = Math.floor(nu * n);
    for (let i = 0; i < nBound; i++) {
        alpha0[i] = 1;
    }
    if (nBound < n) {
        alpha0[nBound] = nu * n - nBound;
    }
    const y = new Array<number>(n).fill(1);
    const p = new Array<number>(n).fill(0);
    const sol = solveGenericC(n, y, p, 1, (i) => K.getRow(i), tol, maxIter, alpha0);
    return { alpha: Array.from(sol.alpha), rho: sol.rho, iterations: sol.iterations, converged: sol.converged };
}
