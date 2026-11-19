import { Params, RegressorBase, registerEstimator } from '../base';
import { solveLower, solveLowerTransposed } from '../discriminant_analysis/linalg';
import { cholesky } from '../mixture/math';
import { dot, validateMatrix, validateRegressionData } from '../utils/numerics';
import { kernelMatrix, rbfKernel, RBFKernelOptions, resolveKernel, validateLengthScale, validateQuery } from './common';

export interface GaussianProcessOptimizerOptions {
    lengthScaleBounds?: [number, number];
    signalVarianceBounds?: [number, number];
    maxIter?: number;
}

export interface GaussianProcessRegressorProps extends RBFKernelOptions {
    /** Observation noise variance added to the training diagonal, in normalized units when normalizeY is enabled. */
    alpha?: number | number[];
    normalizeY?: boolean;
    /** Opt into bounded coordinate search over log kernel parameters. Null keeps the supplied kernel fixed. */
    optimizer?: GaussianProcessOptimizerOptions | null;
}

interface Posterior {
    kernel: Required<RBFKernelOptions>;
    lower: number[][];
    dual: number[];
    likelihood: number;
}

function bounds(pair: [number, number], name: string): [number, number] {
    if (!Array.isArray(pair) || pair.length !== 2 || pair.some(v => !Number.isFinite(v) || v <= 0) || pair[0] > pair[1]) {
        throw new Error(name + ' requires ordered, finite, positive bounds');
    }
    return [pair[0], pair[1]];
}

function optimizerOptions(options: GaussianProcessOptimizerOptions): Required<GaussianProcessOptimizerOptions> {
    if (typeof options !== 'object' || Array.isArray(options)) throw new Error('optimizer requires an options object or null');
    const { maxIter = 50, lengthScaleBounds = [1e-5, 1e5], signalVarianceBounds = [1e-5, 1e5] } = options;
    if (!Number.isInteger(maxIter) || maxIter < 1) throw new Error('optimizer.maxIter must be a positive integer');
    return {
        maxIter,
        lengthScaleBounds: bounds(lengthScaleBounds, 'lengthScaleBounds'),
        signalVarianceBounds: bounds(signalVarianceBounds, 'signalVarianceBounds'),
    };
}

function posterior(X: number[][], y: number[], alpha: number | number[], kernel: Required<RBFKernelOptions>): Posterior | null {
    const gram = kernelMatrix(X, kernel);
    for (let i = 0; i < X.length; i++) {
        gram[i][i] += Array.isArray(alpha) ? alpha[i] : alpha;
        if (!Number.isFinite(gram[i][i])) return null;
    }
    let lower: number[][];
    try {
        lower = cholesky(gram);
    } catch {
        // A candidate on a singular boundary has no usable posterior.
        return null;
    }
    const projected = solveLower(lower, y);
    const dual = solveLowerTransposed(lower, projected);
    let likelihood = -0.5 * (dot(projected, projected) + y.length * Math.log(2 * Math.PI));
    for (let i = 0; i < y.length; i++) likelihood -= Math.log(lower[i][i]);
    if (!Number.isFinite(likelihood) || dual.some(value => !Number.isFinite(value))) return null;
    return { kernel, lower, dual, likelihood };
}

function maximizeLikelihood(
    initial: Posterior, X: number[][], y: number[], alpha: number | number[], options: Required<GaussianProcessOptimizerOptions>
): Posterior {
    const { lengthScale, signalVariance } = initial.kernel;
    const scales = Array.isArray(lengthScale) ? lengthScale : [lengthScale];
    const parameters = [signalVariance, ...scales];
    const ranges = [options.signalVarianceBounds, ...scales.map(() => options.lengthScaleBounds)];
    if (parameters.some((value, i) => value < ranges[i][0] || value > ranges[i][1])) {
        throw new Error('Initial kernel parameters must lie inside optimizer bounds');
    }
    let best = initial;
    let step = 1;
    for (let iteration = 0; iteration < options.maxIter && step >= 1e-4; iteration++) {
        const before = best.likelihood;
        for (let i = 0; i < parameters.length; i++) {
            const start = parameters[i];
            for (const move of [-step, step]) {
                const candidate = Math.max(ranges[i][0], Math.min(ranges[i][1], start * Math.exp(move)));
                if (candidate === start) continue;
                const proposed = parameters.slice();
                proposed[i] = candidate;
                const kernel = {
                    signalVariance: proposed[0],
                    lengthScale: Array.isArray(lengthScale) ? proposed.slice(1) : proposed[1],
                };
                const trial = posterior(X, y, alpha, kernel);
                if (trial && trial.likelihood > best.likelihood) {
                    best = trial;
                    parameters[i] = candidate;
                }
            }
        }
        if (best.likelihood === before) step *= 0.5;
    }
    return best;
}

/** Scale the observations before summation/squaring so large finite targets can be normalized. */
function normalizeTargets(y: number[]): { mean: number; scale: number; targets: number[] } {
    let magnitude = 0;
    for (const value of y) magnitude = Math.max(magnitude, Math.abs(value));
    if (magnitude === 0) return { mean: 0, scale: 1, targets: y };
    const scaled = y.map(value => value / magnitude);
    // Accumulate displacements from the first observation to preserve constant targets exactly.
    const meanScaled = scaled[0] + scaled.reduce((sum, value) => sum + (value - scaled[0]), 0) / y.length;
    const centered = scaled.map(value => value - meanScaled);
    const sdScaled = Math.sqrt(dot(centered, centered) / y.length);
    if (sdScaled === 0) return { mean: y[0], scale: 1, targets: centered };
    return { mean: meanScaled * magnitude, scale: sdScaled * magnitude, targets: centered.map(value => value / sdScaled) };
}

/** Exact, single-target Gaussian process regression with a scaled RBF kernel. */
export class GaussianProcessRegressor extends RegressorBase {
    private readonly kernel: Required<RBFKernelOptions>;
    private readonly alpha: number | number[];
    private readonly normalizeY: boolean;
    private readonly optimizer: Required<GaussianProcessOptimizerOptions> | null;
    private fitted?: Posterior;
    private training: number[][] = [];
    private targetMean = 0;
    private targetScale = 1;

    constructor(props: GaussianProcessRegressorProps = {}) {
        super();
        const { alpha = 1e-10, normalizeY = false, optimizer = null } = props;
        this.kernel = resolveKernel(props);
        const noise = Array.isArray(alpha) ? alpha : [alpha];
        if (noise.length === 0 || noise.some(value => !Number.isFinite(value) || value < 0)) {
            throw new Error('alpha must contain finite nonnegative noise variances');
        }
        if (typeof normalizeY !== 'boolean') throw new Error('normalizeY must be boolean');
        this.alpha = Array.isArray(alpha) ? alpha.slice() : alpha;
        this.normalizeY = normalizeY;
        this.optimizer = optimizer === null ? null : optimizerOptions(optimizer);
    }

    public getParams(): Params {
        return {
            ...resolveKernel(this.kernel),
            alpha: Array.isArray(this.alpha) ? this.alpha.slice() : this.alpha,
            normalizeY: this.normalizeY,
            optimizer: this.optimizer && optimizerOptions(this.optimizer),
        };
    }

    public fit(X: number[][], y: number[]): void {
        const p = validateRegressionData(X, y);
        validateLengthScale(this.kernel, p);
        if (Array.isArray(this.alpha) && this.alpha.length !== y.length) {
            throw new Error('alpha must have one noise variance per training sample');
        }
        const { mean, scale, targets } = this.normalizeY ? normalizeTargets(y) : { mean: 0, scale: 1, targets: y };
        const initial = posterior(X, targets, this.alpha, this.kernel);
        if (!initial) throw new Error('Unstable Gaussian process covariance or target scale. Check the data scale or increase alpha.');
        const fitted = this.optimizer ? maximizeLikelihood(initial, X, targets, this.alpha, this.optimizer) : initial;
        this.fitted = fitted;
        this.training = X.map(row => row.slice());
        this.targetMean = mean;
        this.targetScale = scale;
    }

    public get fittedKernel(): Required<RBFKernelOptions> {
        if (!this.fitted) throw new Error('GaussianProcessRegressor is not fitted');
        return resolveKernel(this.fitted.kernel);
    }

    public logMarginalLikelihood(): number {
        if (!this.fitted) throw new Error('GaussianProcessRegressor is not fitted');
        return this.fitted.likelihood;
    }

    public predict(X: number[][]): number[];
    public predict(X: number[][], options: { returnStd: true; returnCov?: false }): { mean: number[]; std: number[] };
    public predict(X: number[][], options: { returnCov: true; returnStd?: false }): { mean: number[]; cov: number[][] };
    public predict(X: number[][], options: { returnStd?: false; returnCov?: false }): number[];
    public predict(X: number[][], options: { returnStd?: boolean; returnCov?: boolean } = {}):
        number[] | { mean: number[]; std: number[] } | { mean: number[]; cov: number[][] } {
        if (options.returnStd && options.returnCov) throw new Error('Request either standard deviations or covariance');
        const kernel = this.fitted?.kernel ?? this.kernel;
        if (this.fitted) {
            validateQuery(X, this.training[0].length);
        } else if (X.length > 0) {
            validateLengthScale(kernel, validateMatrix(X));
        }

        const projections: number[][] = [];
        const deviations: number[] = [];
        const mean = X.map(row => {
            const cross = this.training.map(train => rbfKernel(row, train, kernel));
            if (options.returnStd || options.returnCov) {
                const projected = this.fitted ? solveLower(this.fitted.lower, cross) : [];
                if (options.returnCov) projections.push(projected);
                if (options.returnStd) {
                    deviations.push(this.targetScale * Math.sqrt(Math.max(0, kernel.signalVariance - dot(projected, projected))));
                }
            }
            return this.fitted ? this.targetMean + this.targetScale * dot(cross, this.fitted.dual) : 0;
        });
        if (options.returnStd) return { mean, std: deviations };
        if (options.returnCov) {
            const cov = kernelMatrix(X, kernel);
            for (let i = 0; i < X.length; i++) {
                for (let j = 0; j <= i; j++) {
                    const value = cov[i][j] - dot(projections[i], projections[j]);
                    cov[i][j] = cov[j][i] = this.targetScale * (i === j ? Math.max(0, value) : value) * this.targetScale;
                }
            }
            return { mean, cov };
        }
        return mean;
    }
}
registerEstimator('GaussianProcessRegressor', GaussianProcessRegressor);
