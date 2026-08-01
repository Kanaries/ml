import { BaseEstimator } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { dot, matMul, pseudoInverseSymmetric, transpose, validateMatrix } from '../utils/numerics';
import { r2Score } from '../metrics';

interface PLSProps { nComponents?: number; scale?: boolean; maxIter?: number; tol?: number; }
type Targets = number[] | number[][];

abstract class BasePLS extends BaseEstimator {
    protected nComponents: number; protected scale: boolean; protected maxIter: number; protected tol: number; protected mode: 'A' | 'B'; protected canonical: boolean;
    protected xMean: number[] = []; protected yMean: number[] = []; protected xStd: number[] = []; protected yStd: number[] = [];
    protected xWeightsState: number[][] = []; protected yWeightsState: number[][] = []; protected xLoadingsState: number[][] = []; protected yLoadingsState: number[][] = []; protected xRotationsState: number[][] = []; protected yRotationsState: number[][] = []; protected nIterState: number[] = []; protected yWas1d = false;
    constructor(props: PLSProps, mode: 'A' | 'B', canonical: boolean) { super(); const { nComponents = 2, scale = true, maxIter = 500, tol = 1e-6 } = props; if (!Number.isInteger(nComponents) || nComponents < 1 || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(tol) || tol <= 0) throw new Error('invalid cross-decomposition parameters'); this.nComponents = nComponents; this.scale = scale; this.maxIter = maxIter; this.tol = tol; this.mode = mode; this.canonical = canonical; }
    public getParams(): Params { return { nComponents: this.nComponents, scale: this.scale, maxIter: this.maxIter, tol: this.tol }; }
    private normalize(X: number[][]): { centered: number[][]; mean: number[]; std: number[] } { const n = X.length, p = X[0].length, mean = Array.from({ length: p }, (_, j) => X.reduce((sum, row) => sum + row[j], 0) / n), std = Array.from({ length: p }, (_, j) => this.scale ? Math.sqrt(X.reduce((sum, row) => sum + (row[j] - mean[j]) ** 2, 0) / Math.max(1, n - 1)) || 1 : 1); return { centered: X.map(row => row.map((value, j) => (value - mean[j]) / std[j])), mean, std }; }
    private pseudoInverse(X: number[][]): number[][] { return matMul(pseudoInverseSymmetric(matMul(transpose(X), X)), transpose(X)); }
    public fit(X: number[][], y: Targets): void {
        const p = validateMatrix(X, 2), Y = Array.isArray(y[0]) ? (y as number[][]).map(row => row.slice()) : (y as number[]).map(value => [value]); this.yWas1d = !Array.isArray(y[0]);
        if (Y.length !== X.length || Y.some(row => row.some(value => !Number.isFinite(value)))) throw new Error('X and y must have matching finite rows'); const q = Y[0].length;
        if (this.nComponents > (this.canonical ? Math.min(X.length, p, q) : Math.min(X.length, p))) throw new Error('nComponents exceeds the rank upper bound');
        const xn = this.normalize(X), yn = this.normalize(Y); this.xMean = xn.mean; this.xStd = xn.std; this.yMean = yn.mean; this.yStd = yn.std;
        let Xk = xn.centered.map(row => row.slice()), Yk = yn.centered.map(row => row.slice());
        const xWeightsColumns: number[][] = [], yWeightsColumns: number[][] = [], xLoadingColumns: number[][] = [], yLoadingColumns: number[][] = []; this.nIterState = [];
        for (let component = 0; component < this.nComponents; component++) {
            let yScore = transpose(Yk).find(column => column.some(value => Math.abs(value) > Number.EPSILON)); if (!yScore) break;
            let xWeights = new Array(p).fill(100), yWeights = new Array(q).fill(0), used = 0;
            const Xpinv = this.mode === 'B' ? this.pseudoInverse(Xk) : [], Ypinv = this.mode === 'B' ? this.pseudoInverse(Yk) : [];
            for (used = 1; used <= this.maxIter; used++) {
                const previous = xWeights.slice();
                xWeights = this.mode === 'B' ? Xpinv.map(row => dot(row, yScore!)) : transpose(Xk).map(column => dot(column, yScore!) / Math.max(dot(yScore!, yScore!), Number.EPSILON));
                const xNorm = Math.sqrt(dot(xWeights, xWeights)) + Number.EPSILON; xWeights = xWeights.map(value => value / xNorm); const xScore = Xk.map(row => dot(row, xWeights));
                yWeights = this.mode === 'B' ? Ypinv.map(row => dot(row, xScore)) : transpose(Yk).map(column => dot(column, xScore) / Math.max(dot(xScore, xScore), Number.EPSILON));
                if (this.canonical) { const norm = Math.sqrt(dot(yWeights, yWeights)) + Number.EPSILON; yWeights = yWeights.map(value => value / norm); }
                yScore = Yk.map(row => dot(row, yWeights) / (dot(yWeights, yWeights) + Number.EPSILON));
                if (xWeights.reduce((sum, value, i) => sum + (value - previous[i]) ** 2, 0) < this.tol || q === 1) break;
            }
            let pivot = 0; for (let i = 1; i < p; i++) if (Math.abs(xWeights[i]) > Math.abs(xWeights[pivot])) pivot = i; if (xWeights[pivot] < 0) { xWeights = xWeights.map(value => -value); yWeights = yWeights.map(value => -value); }
            const xScore = Xk.map(row => dot(row, xWeights)), yDen = this.canonical ? 1 : dot(yWeights, yWeights), finalYScore = Yk.map(row => dot(row, yWeights) / Math.max(yDen, Number.EPSILON));
            const xDen = dot(xScore, xScore), xLoading = transpose(Xk).map(column => dot(xScore, column) / xDen), yBasis = this.canonical ? finalYScore : xScore, yLoading = transpose(Yk).map(column => dot(yBasis, column) / dot(yBasis, yBasis));
            Xk = Xk.map((row, i) => row.map((value, j) => value - xScore[i] * xLoading[j])); Yk = Yk.map((row, i) => row.map((value, j) => value - yBasis[i] * yLoading[j]));
            xWeightsColumns.push(xWeights); yWeightsColumns.push(yWeights); xLoadingColumns.push(xLoading); yLoadingColumns.push(yLoading); this.nIterState.push(Math.min(used, this.maxIter));
        }
        this.xWeightsState = transpose(xWeightsColumns); this.yWeightsState = transpose(yWeightsColumns); this.xLoadingsState = transpose(xLoadingColumns); this.yLoadingsState = transpose(yLoadingColumns);
        const xMiddle = matMul(transpose(this.xLoadingsState), this.xWeightsState), yMiddle = matMul(transpose(this.yLoadingsState), this.yWeightsState);
        this.xRotationsState = matMul(this.xWeightsState, this.pseudoInverse(xMiddle)); this.yRotationsState = matMul(this.yWeightsState, this.pseudoInverse(yMiddle));
    }
    public transform(X: number[][]): number[][] { if (this.xRotationsState.length === 0) throw new Error('cross-decomposition estimator is not fitted'); return matMul(X.map(row => row.map((value, j) => (value - this.xMean[j]) / this.xStd[j])), this.xRotationsState); }
    public predict(X: number[][]): number[] | number[][] { const scores = this.transform(X), scaled = matMul(scores, transpose(this.yLoadingsState)), prediction = scaled.map(row => row.map((value, j) => value * this.yStd[j] + this.yMean[j])); return this.yWas1d ? prediction.map(row => row[0]) : prediction; }
    public score(X: number[][], y: Targets): number {
        const prediction = this.predict(X);
        if (!Array.isArray(y[0])) return r2Score(prediction as number[], y as number[]);
        const expected = y as number[][], actual = prediction as number[][];
        return expected[0].reduce((sum, _, output) => sum + r2Score(actual.map(row => row[output]), expected.map(row => row[output])), 0) / expected[0].length;
    }
    public get xWeights(): number[][] { return this.xWeightsState.map(row => row.slice()); } public get yWeights(): number[][] { return this.yWeightsState.map(row => row.slice()); }
    public get xLoadings(): number[][] { return this.xLoadingsState.map(row => row.slice()); } public get yLoadings(): number[][] { return this.yLoadingsState.map(row => row.slice()); }
    public get xRotations(): number[][] { return this.xRotationsState.map(row => row.slice()); } public get nIter(): number[] { return this.nIterState.slice(); }
}

export interface PLSRegressionProps extends PLSProps {}
export class PLSRegression extends BasePLS { constructor(props: PLSRegressionProps = {}) { super(props, 'A', false); } }
registerEstimator('PLSRegression', PLSRegression);

export interface CCAProps extends PLSProps {}
export class CCA extends BasePLS { constructor(props: CCAProps = {}) { super(props, 'B', true); } }
registerEstimator('CCA', CCA);
