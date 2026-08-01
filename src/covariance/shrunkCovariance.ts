import { Params, registerEstimator } from '../base/estimator';
import { EmpiricalCovariance, EmpiricalCovarianceProps, empiricalMoments } from './empiricalCovariance';

export interface ShrunkCovarianceProps extends EmpiricalCovarianceProps { shrinkage?: number; }

function shrink(covariance: number[][], amount: number): number[][] {
    const p = covariance.length, mu = covariance.reduce((sum, row, i) => sum + row[i], 0) / p;
    return covariance.map((row, i) => row.map((value, j) => (1 - amount) * value + (i === j ? amount * mu : 0)));
}

export class ShrunkCovariance extends EmpiricalCovariance {
    protected shrinkage: number;
    constructor(props: ShrunkCovarianceProps = {}) { super(props); const { shrinkage = .1 } = props; if (!Number.isFinite(shrinkage) || shrinkage < 0 || shrinkage > 1) throw new Error('shrinkage must be in [0, 1]'); this.shrinkage = shrinkage; }
    public getParams(): Params { return { assumeCentered: this.assumeCentered, shrinkage: this.shrinkage }; }
    public fit(X: number[][]): void { const result = empiricalMoments(X, this.assumeCentered); this.setEstimate(result.location, shrink(result.covariance, this.shrinkage)); }
    public get shrinkageValue(): number { return this.shrinkage; }
}
registerEstimator('ShrunkCovariance', ShrunkCovariance);

export class LedoitWolf extends EmpiricalCovariance {
    private shrinkageState = 0;
    public getParams(): Params { return { assumeCentered: this.assumeCentered }; }
    public fit(X: number[][]): void {
        const result = empiricalMoments(X, this.assumeCentered), n = X.length, p = result.covariance.length;
        if (p === 1) this.shrinkageState = 0;
        else {
            const X2 = result.centered.map(row => row.map(value => value * value));
            const trace = Array.from({ length: p }, (_, j) => X2.reduce((sum, row) => sum + row[j], 0) / n);
            const mu = trace.reduce((a, b) => a + b, 0) / p;
            let betaRaw = 0, deltaRaw = 0;
            for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) {
                let squaredDot = 0, productDot = 0;
                for (let s = 0; s < n; s++) { squaredDot += X2[s][i] * X2[s][j]; productDot += result.centered[s][i] * result.centered[s][j]; }
                betaRaw += squaredDot; deltaRaw += productDot * productDot;
            }
            const deltaBase = deltaRaw / (n * n);
            let beta = (betaRaw / n - deltaBase) / (p * n);
            let delta = (deltaBase - 2 * mu * trace.reduce((a, b) => a + b, 0) + p * mu * mu) / p;
            beta = Math.min(beta, delta); this.shrinkageState = beta === 0 || delta === 0 ? 0 : beta / delta;
        }
        this.setEstimate(result.location, shrink(result.covariance, this.shrinkageState));
    }
    public get shrinkageValue(): number { return this.shrinkageState; }
}
registerEstimator('LedoitWolf', LedoitWolf);

export class OAS extends EmpiricalCovariance {
    private shrinkageState = 0;
    public getParams(): Params { return { assumeCentered: this.assumeCentered }; }
    public fit(X: number[][]): void {
        const result = empiricalMoments(X, this.assumeCentered), n = X.length, p = result.covariance.length;
        if (p === 1) this.shrinkageState = 0;
        else {
            let alpha = 0, trace = 0;
            for (let i = 0; i < p; i++) { trace += result.covariance[i][i]; for (let j = 0; j < p; j++) alpha += result.covariance[i][j] ** 2; }
            alpha /= p * p; const mu2 = (trace / p) ** 2, denominator = (n + 1) * (alpha - mu2 / p);
            this.shrinkageState = denominator === 0 ? 1 : Math.min((alpha + mu2) / denominator, 1);
        }
        this.setEstimate(result.location, shrink(result.covariance, this.shrinkageState));
    }
    public get shrinkageValue(): number { return this.shrinkageState; }
}
registerEstimator('OAS', OAS);
