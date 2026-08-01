import { BaseEstimator } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { logGamma, unitBallVolume, validateMatrix } from '../utils/numerics';

export type DensityKernel = 'gaussian' | 'tophat' | 'epanechnikov' | 'exponential' | 'linear' | 'cosine';
export interface KernelDensityProps { bandwidth?: number; kernel?: DensityKernel; atol?: number; rtol?: number; }

function radialIntegral(kernel: DensityKernel, dimension: number): number {
    if (kernel === 'gaussian') return Math.exp(dimension / 2 * Math.log(2 * Math.PI));
    const surface = dimension * unitBallVolume(dimension);
    if (kernel === 'tophat') return unitBallVolume(dimension);
    if (kernel === 'epanechnikov') return 2 * unitBallVolume(dimension) / (dimension + 2);
    if (kernel === 'linear') return unitBallVolume(dimension) / (dimension + 1);
    if (kernel === 'exponential') return surface * Math.exp(logGamma(dimension));
    const power = dimension - 1, profile = (radius: number) => radius ** power * Math.cos(Math.PI * radius / 2);
    const simpson = (left: number, right: number) => { const middle = (left + right) / 2; return (right - left) * (profile(left) + 4 * profile(middle) + profile(right)) / 6; };
    const integrate = (left: number, right: number, whole: number, tolerance: number, depth: number): number => {
        const middle = (left + right) / 2, lower = simpson(left, middle), upper = simpson(middle, right), delta = lower + upper - whole;
        if (depth === 0 || Math.abs(delta) <= 15 * tolerance) return lower + upper + delta / 15;
        return integrate(left, middle, lower, tolerance / 2, depth - 1) + integrate(middle, right, upper, tolerance / 2, depth - 1);
    };
    const integral = integrate(0, 1, simpson(0, 1), 1e-13, 24);
    return surface * integral;
}

export class KernelDensity extends BaseEstimator {
    private bandwidth: number; private kernel: DensityKernel; private atol: number; private rtol: number;
    private trainingState: number[][] = []; private nFeaturesState = 0;
    constructor(props: KernelDensityProps = {}) {
        super(); const { bandwidth = 1, kernel = 'gaussian', atol = 0, rtol = 0 } = props;
        if (!Number.isFinite(bandwidth) || bandwidth <= 0 || !['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'].includes(kernel) || !Number.isFinite(atol) || atol < 0 || !Number.isFinite(rtol) || rtol < 0) throw new Error('invalid KernelDensity parameters');
        this.bandwidth = bandwidth; this.kernel = kernel; this.atol = atol; this.rtol = rtol;
    }
    public getParams(): Params { return { bandwidth: this.bandwidth, kernel: this.kernel, atol: this.atol, rtol: this.rtol }; }
    public fit(X: number[][]): void { this.nFeaturesState = validateMatrix(X); this.trainingState = X.map(row => row.slice()); }
    private profile(radius: number): number {
        if (this.kernel === 'gaussian') return Math.exp(-radius * radius / 2);
        if (this.kernel === 'exponential') return Math.exp(-radius);
        if (this.kernel === 'tophat') return radius <= 1 ? 1 : 0;
        if (radius >= 1) return 0;
        if (this.kernel === 'epanechnikov') return 1 - radius * radius;
        if (this.kernel === 'linear') return 1 - radius;
        return Math.cos(Math.PI * radius / 2);
    }
    public scoreSamples(X: number[][]): number[] {
        if (this.trainingState.length === 0) throw new Error('KernelDensity is not fitted');
        if (X.some(row => row.length !== this.nFeaturesState || row.some(value => !Number.isFinite(value)))) throw new Error('input must be finite and match fitted dimensionality');
        const normalizer = radialIntegral(this.kernel, this.nFeaturesState) * this.bandwidth ** this.nFeaturesState * this.trainingState.length;
        return X.map(row => {
            const density = this.trainingState.reduce((sum, sample) => {
                const radius = Math.sqrt(row.reduce((distance, value, j) => distance + (value - sample[j]) ** 2, 0)) / this.bandwidth;
                return sum + this.profile(radius);
            }, 0) / normalizer;
            return Math.log(density);
        });
    }
    public score(X: number[][]): number { return this.scoreSamples(X).reduce((sum, value) => sum + value, 0); }
}
registerEstimator('KernelDensity', KernelDensity);
