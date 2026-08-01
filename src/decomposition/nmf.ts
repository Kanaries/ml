import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { jacobiSVD } from '../discriminant_analysis/linalg';
import { createRandomGenerator } from '../utils/random';

export type NMFInit = 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar';
export interface NMFProps {
    nComponents?: number;
    init?: NMFInit;
    maxIter?: number;
    tol?: number;
    alphaW?: number;
    alphaH?: number;
    l1Ratio?: number;
    randomState?: number;
}

function multiply(A: number[][], B: number[][]): number[][] {
    return A.map(row => Array.from({ length: B[0].length }, (_, j) => row.reduce((sum, value, k) => sum + value * B[k][j], 0)));
}
function transpose(A: number[][]): number[][] { return Array.from({ length: A[0].length }, (_, j) => A.map(row => row[j])); }
function frobenius(A: number[][], B?: number[][]): number {
    let sum = 0;
    for (let i = 0; i < A.length; i++) for (let j = 0; j < A[i].length; j++) sum += (A[i][j] - (B?.[i][j] ?? 0)) ** 2;
    return Math.sqrt(sum);
}

export class NMF extends TransformerBase {
    private nComponents: number;
    private init: NMFInit;
    private maxIter: number;
    private tol: number;
    private alphaW: number;
    private alphaH: number;
    private l1Ratio: number;
    private randomState?: number;
    private componentsState: number[][] = [];
    private WState: number[][] = [];
    private reconstructionErrState = 0;
    private nIterState = 0;

    constructor(props: NMFProps = {}) {
        super();
        const { nComponents = 2, init = 'nndsvda', maxIter = 200, tol = 1e-4, alphaW = 0, alphaH = 0, l1Ratio = 0, randomState } = props;
        if (!Number.isInteger(nComponents) || nComponents < 1) throw new Error('nComponents must be positive');
        if (!['random', 'nndsvd', 'nndsvda', 'nndsvdar'].includes(init) || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(tol) || tol < 0 || !Number.isFinite(alphaW) || alphaW < 0 || !Number.isFinite(alphaH) || alphaH < 0 || !Number.isFinite(l1Ratio) || l1Ratio < 0 || l1Ratio > 1) throw new Error('invalid NMF parameters');
        this.nComponents = nComponents; this.init = init; this.maxIter = maxIter; this.tol = tol;
        this.alphaW = alphaW; this.alphaH = alphaH; this.l1Ratio = l1Ratio; this.randomState = randomState;
    }
    public getParams(): Params { return { nComponents: this.nComponents, init: this.init, maxIter: this.maxIter, tol: this.tol, alphaW: this.alphaW, alphaH: this.alphaH, l1Ratio: this.l1Ratio, randomState: this.randomState }; }
    private initialize(X: number[][]): [number[][], number[][]] {
        const n = X.length, p = X[0].length, k = this.nComponents, random = createRandomGenerator(this.randomState);
        let total = 0; for (const row of X) for (const value of row) total += value;
        const average = total / (n * p);
        if (this.init === 'random') {
            const scale = Math.sqrt(average / k);
            return [Array.from({ length: n }, () => Array.from({ length: k }, () => Math.max(1e-8, random() * scale))), Array.from({ length: k }, () => Array.from({ length: p }, () => Math.max(1e-8, random() * scale)))];
        }
        const { S, Vt } = jacobiSVD(X), usable = Math.min(k, S.length);
        const W = Array.from({ length: n }, () => new Array(k).fill(0));
        const H = Array.from({ length: k }, () => new Array(p).fill(0));
        const left = (component: number) => X.map(row => row.reduce((sum, value, j) => sum + value * Vt[component][j], 0) / Math.max(S[component], 1e-15));
        if (usable > 0) {
            const u = left(0).map(Math.abs), v = Vt[0].map(Math.abs), scale = Math.sqrt(S[0]);
            for (let i = 0; i < n; i++) W[i][0] = scale * u[i];
            for (let j = 0; j < p; j++) H[0][j] = scale * v[j];
        }
        for (let c = 1; c < usable; c++) {
            const u = left(c), v = Vt[c];
            const up = u.map(x => Math.max(x, 0)), un = u.map(x => Math.max(-x, 0));
            const vp = v.map(x => Math.max(x, 0)), vn = v.map(x => Math.max(-x, 0));
            const norm = (values: number[]) => Math.sqrt(values.reduce((sum, x) => sum + x * x, 0));
            const positive = norm(up) * norm(vp), negative = norm(un) * norm(vn);
            const uu = positive > negative ? up : un, vv = positive > negative ? vp : vn;
            const sigma = Math.sqrt(S[c] * Math.max(positive, negative));
            const nu = norm(uu) || 1, nv = norm(vv) || 1;
            for (let i = 0; i < n; i++) W[i][c] = sigma * uu[i] / nu;
            for (let j = 0; j < p; j++) H[c][j] = sigma * vv[j] / nv;
        }
        for (const matrix of [W, H]) for (const row of matrix) for (let j = 0; j < row.length; j++) {
            if (row[j] < 1e-6) row[j] = 0;
            if (row[j] === 0) row[j] = this.init === 'nndsvda' ? average : this.init === 'nndsvdar' ? average * random() / 100 : 0;
        }
        return [W, H];
    }
    private updateW(X: number[][], W: number[][], H: number[][]): void {
        const numerator = multiply(X, transpose(H)), denominator = multiply(W, multiply(H, transpose(H)));
        const l1 = this.alphaW * this.l1Ratio * X[0].length, l2 = this.alphaW * (1 - this.l1Ratio) * X[0].length;
        for (let i = 0; i < W.length; i++) for (let c = 0; c < W[i].length; c++) W[i][c] *= numerator[i][c] / Math.max(denominator[i][c] + l1 + l2 * W[i][c], 1e-15);
    }
    private updateH(X: number[][], W: number[][], H: number[][]): void {
        const numerator = multiply(transpose(W), X), denominator = multiply(multiply(transpose(W), W), H);
        const l1 = this.alphaH * this.l1Ratio * X.length, l2 = this.alphaH * (1 - this.l1Ratio) * X.length;
        for (let c = 0; c < H.length; c++) for (let j = 0; j < H[c].length; j++) H[c][j] *= numerator[c][j] / Math.max(denominator[c][j] + l1 + l2 * H[c][j], 1e-15);
    }
    public fit(X: number[][]): void { this.fitTransform(X); }
    public fitTransform(X: number[][]): number[][] {
        if (X.length === 0 || X[0].length === 0 || X.some(row => row.length !== X[0].length || row.some(value => value < 0 || !Number.isFinite(value)))) throw new Error('NMF requires a non-empty finite rectangular non-negative matrix');
        if (this.nComponents > Math.min(X.length, X[0].length) && this.init !== 'random') throw new Error('NNDSVD initialization requires nComponents <= min(X.shape)');
        let [W, H] = this.initialize(X); const initial = frobenius(X, multiply(W, H)); let previous = initial;
        for (this.nIterState = 1; this.nIterState <= this.maxIter; this.nIterState++) {
            this.updateW(X, W, H); this.updateH(X, W, H);
            if (this.tol > 0 && this.nIterState % 10 === 0) {
                const error = frobenius(X, multiply(W, H));
                if ((previous - error) / Math.max(initial, 1e-15) < this.tol) { previous = error; break; }
                previous = error;
            }
        }
        this.nIterState = Math.min(this.nIterState, this.maxIter);
        this.WState = W; this.componentsState = H; this.reconstructionErrState = frobenius(X, multiply(W, H));
        return W.map(row => row.slice());
    }
    public transform(X: number[][]): number[][] {
        if (this.componentsState.length === 0) throw new Error('NMF is not fitted');
        if (X.some(row => row.length !== this.componentsState[0].length || row.some(value => value < 0 || !Number.isFinite(value)))) throw new Error('NMF requires finite non-negative input with the fitted feature count');
        let total = 0; for (const row of X) for (const value of row) total += value;
        const initialValue = Math.sqrt((total / Math.max(1, X.length * this.componentsState[0].length)) / this.nComponents);
        const W = Array.from({ length: X.length }, () => new Array(this.nComponents).fill(initialValue));
        const initialError = frobenius(X, multiply(W, this.componentsState)); let previous = initialError;
        for (let iteration = 1; iteration <= this.maxIter; iteration++) {
            this.updateW(X, W, this.componentsState);
            if (this.tol > 0 && iteration % 10 === 0) {
                const error = frobenius(X, multiply(W, this.componentsState));
                if ((previous - error) / Math.max(initialError, 1e-15) < this.tol) break;
                previous = error;
            }
        }
        return W;
    }
    public inverseTransform(X: number[][]): number[][] { if (this.componentsState.length === 0) throw new Error('NMF is not fitted'); if (X.some(row => row.length !== this.nComponents || row.some(value => value < 0 || !Number.isFinite(value)))) throw new Error('input must be finite, non-negative, and match the fitted component count'); return multiply(X, this.componentsState); }
    public get components(): number[][] { return this.componentsState.map(row => row.slice()); }
    public get reconstructionErr(): number { return this.reconstructionErrState; }
    public get nIter(): number { return this.nIterState; }
}
registerEstimator('NMF', NMF);
