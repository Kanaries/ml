import { BaseEstimator } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { dot, logDetSymmetricPositive, pseudoInverseSymmetric, validateMatrix } from '../utils/numerics';

export interface EmpiricalCovarianceProps { assumeCentered?: boolean; }

export function empiricalMoments(X: number[][], assumeCentered: boolean): { location: number[]; covariance: number[][]; centered: number[][] } {
    const p = validateMatrix(X);
    const location = assumeCentered ? new Array(p).fill(0) : Array.from({ length: p }, (_, j) => X.reduce((sum, row) => sum + row[j], 0) / X.length);
    const centered = X.map(row => row.map((value, j) => value - location[j]));
    const covariance = Array.from({ length: p }, (_, a) => Array.from({ length: p }, (_, b) => centered.reduce((sum, row) => sum + row[a] * row[b], 0) / X.length));
    return { location, covariance, centered };
}

export class EmpiricalCovariance extends BaseEstimator {
    protected assumeCentered: boolean; protected locationState: number[] = []; protected covarianceState: number[][] = []; protected precisionState: number[][] = [];
    constructor(props: EmpiricalCovarianceProps = {}) { super(); this.assumeCentered = props.assumeCentered ?? false; }
    public getParams(): Params { return { assumeCentered: this.assumeCentered }; }
    protected setEstimate(location: number[], covariance: number[][]): void {
        this.locationState = location.slice(); this.covarianceState = covariance.map(row => row.slice()); this.precisionState = pseudoInverseSymmetric(covariance);
    }
    public fit(X: number[][]): void { const result = empiricalMoments(X, this.assumeCentered); this.setEstimate(result.location, result.covariance); }
    public mahalanobis(X: number[][]): number[] {
        if (this.precisionState.length === 0) throw new Error('covariance estimator is not fitted');
        return X.map(row => { if (row.length !== this.locationState.length) throw new Error('feature count differs from fitted covariance'); const delta = row.map((value, j) => value - this.locationState[j]); return dot(delta, this.precisionState.map(matrixRow => dot(matrixRow, delta))); });
    }
    public score(X: number[][]): number {
        validateMatrix(X); const p = this.locationState.length, precisionLogDet = logDetSymmetricPositive(this.precisionState);
        return -.5 * (p * Math.log(2 * Math.PI) - precisionLogDet + this.mahalanobis(X).reduce((a, b) => a + b, 0) / X.length);
    }
    public errorNorm(comparison: number[][], scaling = true, squared = true): number {
        if (comparison.length !== this.covarianceState.length || comparison.some(row => row.length !== comparison.length)) throw new Error('comparison must match covariance shape');
        let error = 0; for (let i = 0; i < comparison.length; i++) for (let j = 0; j < comparison.length; j++) error += (comparison[i][j] - this.covarianceState[i][j]) ** 2;
        if (scaling) error /= comparison.length; return squared ? error : Math.sqrt(error);
    }
    public get location(): number[] { return this.locationState.slice(); }
    public get covariance(): number[][] { return this.covarianceState.map(row => row.slice()); }
    public get precision(): number[][] { return this.precisionState.map(row => row.slice()); }
}
registerEstimator('EmpiricalCovariance', EmpiricalCovariance);
