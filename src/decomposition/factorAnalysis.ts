import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { jacobiSVD } from '../discriminant_analysis/linalg';
import { identity, inverseMatrix, matMul, pseudoInverseSymmetric, transpose, validateMatrix } from '../utils/numerics';

export interface FactorAnalysisProps { nComponents?: number | null; tol?: number; maxIter?: number; noiseVarianceInit?: number[] | null; rotation?: 'varimax' | 'quartimax' | null; }

function rotate(loadings: number[][], method: 'varimax' | 'quartimax'): number[][] {
    const p = loadings[0].length, k = loadings.length, gamma = method === 'varimax' ? 1 : 0, original = transpose(loadings);
    let R = identity(k), previous = 0;
    for (let iteration = 0; iteration < 50; iteration++) {
        const L = matMul(original, R), columnNorm = Array.from({ length: k }, (_, j) => L.reduce((sum, row) => sum + row[j] ** 2, 0));
        const target = L.map(row => row.map((value, j) => value ** 3 - gamma / p * value * columnNorm[j]));
        const B = matMul(transpose(original), target), svd = jacobiSVD(B), V = transpose(svd.Vt);
        const U = B.map(row => svd.Vt.map((vector, c) => row.reduce((sum, value, j) => sum + value * vector[j], 0) / Math.max(svd.S[c], 1e-15)));
        R = matMul(U, svd.Vt); const objective = svd.S.reduce((a, b) => a + b, 0); if (objective - previous < 1e-7) break; previous = objective;
    }
    return transpose(matMul(original, R));
}

export class FactorAnalysis extends TransformerBase {
    private nComponents: number | null; private tol: number; private maxIter: number; private noiseVarianceInit: number[] | null; private rotation: 'varimax' | 'quartimax' | null;
    private meanState: number[] = []; private componentsState: number[][] = []; private noiseVarianceState: number[] = []; private loglikeState: number[] = []; private nIterState = 0;
    constructor(props: FactorAnalysisProps = {}) { super(); const { nComponents = null, tol = .01, maxIter = 1000, noiseVarianceInit = null, rotation = null } = props; if (nComponents !== null && (!Number.isInteger(nComponents) || nComponents < 0) || !Number.isFinite(tol) || tol <= 0 || !Number.isInteger(maxIter) || maxIter < 1 || rotation !== null && !['varimax', 'quartimax'].includes(rotation)) throw new Error('invalid FactorAnalysis parameters'); this.nComponents = nComponents; this.tol = tol; this.maxIter = maxIter; this.noiseVarianceInit = noiseVarianceInit?.slice() ?? null; this.rotation = rotation; }
    public getParams(): Params { return { nComponents: this.nComponents, tol: this.tol, maxIter: this.maxIter, noiseVarianceInit: this.noiseVarianceInit?.slice() ?? null, rotation: this.rotation }; }
    public fit(X: number[][]): void {
        const p = validateMatrix(X, 2), n = X.length, k = Math.min(this.nComponents ?? p, p, n); this.meanState = Array.from({ length: p }, (_, j) => X.reduce((sum, row) => sum + row[j], 0) / n);
        const centered = X.map(row => row.map((value, j) => value - this.meanState[j])), variance = Array.from({ length: p }, (_, j) => centered.reduce((sum, row) => sum + row[j] ** 2, 0) / n);
        if (this.noiseVarianceInit && this.noiseVarianceInit.length !== p) throw new Error('noiseVarianceInit must match feature count');
        let psi = this.noiseVarianceInit?.slice() ?? new Array(p).fill(1), old = -Infinity, W = Array.from({ length: k }, () => new Array(p).fill(0)); this.loglikeState = [];
        for (this.nIterState = 1; this.nIterState <= this.maxIter; this.nIterState++) {
            const sqrtPsi = psi.map(value => Math.sqrt(value) + 1e-12), scaled = centered.map(row => row.map((value, j) => value / (sqrtPsi[j] * Math.sqrt(n))));
            const svd = jacobiSVD(scaled), squared = svd.S.map(value => value * value), unexplained = squared.slice(k).reduce((a, b) => a + b, 0);
            W = Array.from({ length: k }, (_, c) => svd.Vt[c].map((value, j) => Math.sqrt(Math.max(squared[c] - 1, 0)) * value * sqrtPsi[j]));
            const ll = -.5 * n * (p * Math.log(2 * Math.PI) + k + squared.slice(0, k).reduce((sum, value) => sum + Math.log(value), 0) + unexplained + psi.reduce((sum, value) => sum + Math.log(value), 0));
            this.loglikeState.push(ll); if (ll - old < this.tol) break; old = ll;
            psi = variance.map((value, j) => Math.max(value - W.reduce((sum, row) => sum + row[j] ** 2, 0), 1e-12));
        }
        this.nIterState = Math.min(this.nIterState, this.maxIter); this.componentsState = this.rotation && W.length > 0 ? rotate(W, this.rotation) : W; this.noiseVarianceState = psi;
    }
    public transform(X: number[][]): number[][] {
        if (this.componentsState.length === 0 && (this.nComponents ?? 1) !== 0) throw new Error('FactorAnalysis is not fitted');
        const features = validateMatrix(X); if (features !== this.meanState.length) throw new Error('feature count does not match fitted data');
        if (this.componentsState.length === 0) return X.map(() => []);
        const Wpsi = this.componentsState.map(row => row.map((value, j) => value / this.noiseVarianceState[j])), covariance = inverseMatrix(identity(this.componentsState.length).map((row, i) => row.map((value, j) => value + Wpsi[i].reduce((sum, x, feature) => sum + x * this.componentsState[j][feature], 0))));
        return matMul(X.map(row => row.map((value, j) => value - this.meanState[j])), transpose(Wpsi)).map(row => covariance.map(column => row.reduce((sum, value, j) => sum + value * column[j], 0)));
    }
    public getCovariance(): number[][] { if (this.componentsState.length === 0) return this.noiseVarianceState.map((value, i) => this.noiseVarianceState.map((_, j) => i === j ? value : 0)); const covariance = matMul(transpose(this.componentsState), this.componentsState); return covariance.map((row, i) => row.map((value, j) => value + (i === j ? this.noiseVarianceState[i] : 0))); }
    public getPrecision(): number[][] { return pseudoInverseSymmetric(this.getCovariance()); }
    public get components(): number[][] { return this.componentsState.map(row => row.slice()); }
    public get mean(): number[] { return this.meanState.slice(); }
    public get noiseVariance(): number[] { return this.noiseVarianceState.slice(); }
    public get loglike(): number[] { return this.loglikeState.slice(); }
    public get nIter(): number { return this.nIterState; }
}
registerEstimator('FactorAnalysis', FactorAnalysis);
