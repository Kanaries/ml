import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { symmetricEigDecomposition } from '../discriminant_analysis/linalg';
import { createRandomGenerator } from '../utils/random';

export type ICAFunction = 'logcosh' | 'exp' | 'cube';
export interface FastICAProps {
    nComponents?: number | null;
    algorithm?: 'parallel' | 'deflation';
    whiten?: 'unit-variance' | 'arbitrary-variance' | false;
    fun?: ICAFunction;
    funArgs?: { alpha?: number };
    maxIter?: number;
    tol?: number;
    randomState?: number;
}

function dot(a: number[], b: number[]): number { return a.reduce((sum, value, i) => sum + value * b[i], 0); }
function normalize(v: number[]): number[] { const n = Math.sqrt(dot(v, v)); return v.map(value => value / (n || 1)); }
function matMul(A: number[][], B: number[][]): number[][] {
    return A.map(row => Array.from({ length: B[0].length }, (_, j) => row.reduce((sum, value, k) => sum + value * B[k][j], 0)));
}
function symmetricDecorrelate(W: number[][]): number[][] {
    const gram = W.map(a => W.map(b => dot(a, b)));
    const eig = symmetricEigDecomposition(gram);
    const inverseRoot = Array.from({ length: W.length }, () => new Array(W.length).fill(0));
    for (let c = 0; c < eig.values.length; c++) {
        const scale = 1 / Math.sqrt(Math.max(eig.values[c], 1e-15));
        for (let i = 0; i < W.length; i++) for (let j = 0; j < W.length; j++) inverseRoot[i][j] += eig.vectors[c][i] * eig.vectors[c][j] * scale;
    }
    return matMul(inverseRoot, W);
}
function pseudoInverseRows(A: number[][]): number[][] {
    const gram = A.map(a => A.map(b => dot(a, b)));
    const eig = symmetricEigDecomposition(gram);
    const inverse = Array.from({ length: A.length }, () => new Array(A.length).fill(0));
    for (let c = 0; c < eig.values.length; c++) if (eig.values[c] > 1e-14) {
        for (let i = 0; i < A.length; i++) for (let j = 0; j < A.length; j++) inverse[i][j] += eig.vectors[c][i] * eig.vectors[c][j] / eig.values[c];
    }
    return Array.from({ length: A[0].length }, (_, feature) => Array.from({ length: A.length }, (_, c) => A.reduce((sum, row, r) => sum + row[feature] * inverse[r][c], 0)));
}

export class FastICA extends TransformerBase {
    private nComponents: number | null;
    private algorithm: 'parallel' | 'deflation';
    private whiten: 'unit-variance' | 'arbitrary-variance' | false;
    private fun: ICAFunction;
    private funArgs: { alpha?: number };
    private maxIter: number;
    private tol: number;
    private randomState?: number;
    private meanState: number[] = [];
    private componentsState: number[][] = [];
    private mixingState: number[][] = [];
    private nIterState = 0;

    constructor(props: FastICAProps = {}) {
        super();
        const { nComponents = null, algorithm = 'parallel', whiten = 'unit-variance', fun = 'logcosh', funArgs = {}, maxIter = 200, tol = 1e-4, randomState } = props;
        if (nComponents !== null && (!Number.isInteger(nComponents) || nComponents < 1)) throw new Error('nComponents must be null or positive');
        if ((algorithm !== 'parallel' && algorithm !== 'deflation') || (whiten !== false && whiten !== 'unit-variance' && whiten !== 'arbitrary-variance') || !['logcosh', 'exp', 'cube'].includes(fun) || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(tol) || tol <= 0) throw new Error('invalid FastICA parameters');
        this.nComponents = nComponents; this.algorithm = algorithm; this.whiten = whiten; this.fun = fun;
        this.funArgs = { ...funArgs }; this.maxIter = maxIter; this.tol = tol; this.randomState = randomState;
    }
    public getParams(): Params { return { nComponents: this.nComponents, algorithm: this.algorithm, whiten: this.whiten, fun: this.fun, funArgs: { ...this.funArgs }, maxIter: this.maxIter, tol: this.tol, randomState: this.randomState }; }
    private nonlinearity(value: number): [number, number] {
        if (this.fun === 'cube') return [value ** 3, 3 * value * value];
        if (this.fun === 'exp') { const e = Math.exp(-value * value / 2); return [value * e, (1 - value * value) * e]; }
        const alpha = this.funArgs.alpha ?? 1;
        if (!(alpha >= 1 && alpha <= 2)) throw new Error('logcosh alpha must be in [1, 2]');
        const g = Math.tanh(alpha * value);
        return [g, alpha * (1 - g * g)];
    }
    public fit(X: number[][]): void {
        if (X.length < 2 || X[0].length === 0 || X.some(row => row.length !== X[0].length || row.some(value => !Number.isFinite(value)))) throw new Error('X must be a finite rectangular matrix with at least two rows');
        const n = X.length, p = X[0].length, k = this.whiten === false ? p : Math.min(this.nComponents ?? p, n, p);
        this.nIterState = 0;
        this.meanState = this.whiten === false ? new Array(p).fill(0) : Array.from({ length: p }, (_, j) => X.reduce((sum, row) => sum + row[j], 0) / n);
        const centered = X.map(row => row.map((value, j) => value - this.meanState[j]));
        let whitening: number[][];
        if (this.whiten === false) whitening = Array.from({ length: k }, (_, i) => Array.from({ length: p }, (_, j) => i === j ? 1 : 0));
        else {
            const covariance = Array.from({ length: p }, (_, a) => Array.from({ length: p }, (_, b) => centered.reduce((sum, row) => sum + row[a] * row[b], 0) / n));
            const eig = symmetricEigDecomposition(covariance);
            whitening = eig.vectors.slice(0, k).map((vector, c) => vector.map(value => value / Math.sqrt(Math.max(eig.values[c], 1e-15))));
        }
        const Z = centered.map(row => whitening.map(vector => dot(row, vector)));
        const random = createRandomGenerator(this.randomState);
        let W = Array.from({ length: k }, () => Array.from({ length: k }, () => random() * 2 - 1));
        if (this.algorithm === 'parallel') {
            W = symmetricDecorrelate(W);
            for (this.nIterState = 1; this.nIterState <= this.maxIter; this.nIterState++) {
                const next = W.map(w => {
                    const result = new Array(k).fill(0); let derivative = 0;
                    for (const sample of Z) { const [g, gp] = this.nonlinearity(dot(sample, w)); derivative += gp; for (let j = 0; j < k; j++) result[j] += sample[j] * g; }
                    return result.map((value, j) => value / n - derivative / n * w[j]);
                });
                const decorrelated = symmetricDecorrelate(next);
                const limit = Math.max(...decorrelated.map((row, i) => Math.abs(Math.abs(dot(row, W[i])) - 1)));
                W = decorrelated;
                if (limit < this.tol) break;
            }
            this.nIterState = Math.min(this.nIterState, this.maxIter);
        } else {
            const found: number[][] = [];
            let maxUsed = 0;
            for (let c = 0; c < k; c++) {
                let w = normalize(W[c]); let used = 0;
                for (used = 1; used <= this.maxIter; used++) {
                    const next = new Array(k).fill(0); let derivative = 0;
                    for (const sample of Z) { const [g, gp] = this.nonlinearity(dot(sample, w)); derivative += gp; for (let j = 0; j < k; j++) next[j] += sample[j] * g; }
                    for (let j = 0; j < k; j++) next[j] = next[j] / n - derivative / n * w[j];
                    for (const previous of found) { const projection = dot(next, previous); for (let j = 0; j < k; j++) next[j] -= projection * previous[j]; }
                    const normalized = normalize(next), limit = Math.abs(Math.abs(dot(normalized, w)) - 1); w = normalized;
                    if (limit < this.tol) break;
                }
                maxUsed = Math.max(maxUsed, Math.min(used, this.maxIter)); found.push(w);
            }
            W = found; this.nIterState = maxUsed;
        }
        this.componentsState = matMul(W, whitening);
        if (this.whiten === 'arbitrary-variance') {
            const scale = Math.sqrt(n);
            this.componentsState = this.componentsState.map(component => component.map(value => value / scale));
        } else if (this.whiten === 'unit-variance') {
            const sources = centered.map(row => this.componentsState.map(component => dot(row, component)));
            for (let c = 0; c < this.componentsState.length; c++) {
                const mean = sources.reduce((sum, row) => sum + row[c], 0) / n;
                const std = Math.sqrt(sources.reduce((sum, row) => sum + (row[c] - mean) ** 2, 0) / n) || 1;
                this.componentsState[c] = this.componentsState[c].map(value => value / std);
            }
        }
        this.mixingState = pseudoInverseRows(this.componentsState);
    }
    public transform(X: number[][]): number[][] {
        if (this.componentsState.length === 0) throw new Error('FastICA is not fitted');
        if (X.some(row => row.length !== this.meanState.length || row.some(value => !Number.isFinite(value)))) throw new Error('input must be finite and match the fitted feature count');
        return X.map(row => this.componentsState.map(component => component.reduce((sum, value, j) => sum + value * (row[j] - this.meanState[j]), 0)));
    }
    public inverseTransform(X: number[][]): number[][] {
        if (this.mixingState.length === 0) throw new Error('FastICA is not fitted');
        if (X.some(row => row.length !== this.componentsState.length || row.some(value => !Number.isFinite(value)))) throw new Error('input must be finite and match the fitted component count');
        return X.map(row => this.meanState.map((mean, feature) => mean + row.reduce((sum, value, c) => sum + value * this.mixingState[feature][c], 0)));
    }
    public get components(): number[][] { return this.componentsState.map(row => row.slice()); }
    public get mixing(): number[][] { return this.mixingState.map(row => row.slice()); }
    public get mean(): number[] { return this.meanState.slice(); }
    public get nIter(): number { return this.nIterState; }
}
registerEstimator('FastICA', FastICA);
