import { Params, registerEstimator } from '../base/estimator';
import { logDetSymmetricPositive, pseudoInverseSymmetric } from '../utils/numerics';
import { EmpiricalCovariance, EmpiricalCovarianceProps, empiricalMoments } from './empiricalCovariance';

export interface GraphicalLassoProps extends EmpiricalCovarianceProps { alpha?: number; maxIter?: number; tol?: number; enetTol?: number; }

function softThreshold(value: number, alpha: number): number { return Math.sign(value) * Math.max(Math.abs(value) - alpha, 0); }

/** Sparse inverse covariance using the Friedman et al. block coordinate-descent algorithm. */
export class GraphicalLasso extends EmpiricalCovariance {
    private alpha: number; private maxIter: number; private tol: number; private enetTol: number; private nIterState = 0; private costsState: number[] = [];
    constructor(props: GraphicalLassoProps = {}) {
        super(props); const { alpha = .01, maxIter = 100, tol = 1e-4, enetTol = 1e-4 } = props;
        if (!Number.isFinite(alpha) || alpha < 0 || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(tol) || tol <= 0 || !Number.isFinite(enetTol) || enetTol <= 0) throw new Error('invalid GraphicalLasso parameters');
        this.alpha = alpha; this.maxIter = maxIter; this.tol = tol; this.enetTol = enetTol;
    }
    public getParams(): Params { return { alpha: this.alpha, maxIter: this.maxIter, tol: this.tol, enetTol: this.enetTol, assumeCentered: this.assumeCentered }; }
    private dualGap(empirical: number[][], precision: number[][]): number {
        const p = empirical.length; let trace = 0, l1OffDiagonal = 0;
        for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) { trace += empirical[i][j] * precision[j][i]; if (i !== j) l1OffDiagonal += Math.abs(precision[i][j]); }
        return trace - p + this.alpha * l1OffDiagonal;
    }
    private objective(empirical: number[][], precision: number[][]): number {
        const p = empirical.length, logDet = logDetSymmetricPositive(precision); let trace = 0, l1OffDiagonal = 0;
        for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) { trace += empirical[i][j] * precision[j][i]; if (i !== j) l1OffDiagonal += Math.abs(precision[i][j]); }
        return -logDet + trace + this.alpha * l1OffDiagonal;
    }
    public fit(X: number[][]): void {
        const result = empiricalMoments(X, this.assumeCentered), empirical = result.covariance, p = empirical.length;
        if (this.alpha === 0) { this.setEstimate(result.location, empirical); this.nIterState = 0; this.costsState = []; return; }
        const covariance = empirical.map((row, i) => row.map((value, j) => i === j ? value : .95 * value));
        let precision = pseudoInverseSymmetric(covariance); this.costsState = [];
        for (this.nIterState = 1; this.nIterState <= this.maxIter; this.nIterState++) {
            for (let feature = 0; feature < p; feature++) {
                const indices = Array.from({ length: p }, (_, i) => i).filter(i => i !== feature);
                const gram = indices.map(i => indices.map(j => covariance[i][j])), target = indices.map(i => empirical[i][feature]);
                const beta = indices.map(index => -precision[index][feature] / Math.max(precision[feature][feature], 1000 * Number.EPSILON));
                for (let inner = 0; inner < 1000; inner++) {
                    let largestChange = 0;
                    for (let j = 0; j < beta.length; j++) {
                        let residual = target[j]; for (let k = 0; k < beta.length; k++) if (k !== j) residual -= gram[j][k] * beta[k];
                        const next = softThreshold(residual, this.alpha) / Math.max(gram[j][j], Number.EPSILON); largestChange = Math.max(largestChange, Math.abs(next - beta[j])); beta[j] = next;
                    }
                    if (largestChange < this.enetTol) break;
                }
                const oldColumn = indices.map(index => covariance[index][feature]), covarianceRow = gram.map(row => row.reduce((sum, value, j) => sum + value * beta[j], 0));
                const diagonal = 1 / Math.max(covariance[feature][feature] - oldColumn.reduce((sum, value, j) => sum + value * beta[j], 0), Number.EPSILON);
                precision[feature][feature] = diagonal;
                indices.forEach((index, j) => { precision[feature][index] = -diagonal * beta[j]; precision[index][feature] = -diagonal * beta[j]; });
                indices.forEach((index, j) => { covariance[feature][index] = covarianceRow[j]; covariance[index][feature] = covarianceRow[j]; });
            }
            const gap = this.dualGap(empirical, precision); this.costsState.push(this.objective(empirical, precision));
            if (Math.abs(gap) < this.tol) break;
        }
        this.nIterState = Math.min(this.nIterState, this.maxIter); this.locationState = result.location; this.precisionState = precision; this.covarianceState = covariance;
    }
    public get nIter(): number { return this.nIterState; }
    public get costs(): number[] { return this.costsState.slice(); }
}
registerEstimator('GraphicalLasso', GraphicalLasso);
