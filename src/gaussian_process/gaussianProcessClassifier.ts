import { ClassifierBase, Params, registerEstimator } from '../base';
import { cholesky, solveLower, solveLowerTransposed } from '../discriminant_analysis/linalg';
import { dot, validateRegressionData } from '../utils/numerics';
import { kernelMatrix, rbfKernel, RBFKernelOptions, resolveKernel, validateLengthScale, validateQuery } from './common';

export interface GaussianProcessClassifierProps extends RBFKernelOptions {
    /** Maximum number of Newton updates used to find each binary posterior mode. */
    maxIterPredict?: number;
    /** Relative tolerance on the change in latent function values. */
    tol?: number;
}

interface BinaryPosterior {
    dual: number[];
    sqrtWeight: number[];
    lower: number[][];
}

/** Negative log posterior at the latent values f = K * dual (up to an additive constant). */
function objective(f: number[], dual: number[], labels: number[]): number {
    let value = 0.5 * dot(dual, f);
    for (let i = 0; i < f.length; i++) {
        const margin = labels[i] ? -f[i] : f[i];
        value += Math.max(0, margin) + Math.log1p(Math.exp(-Math.abs(margin)));
    }
    return value;
}

function binaryPosterior(K: number[][], labels: number[], maxIter: number, tol: number): BinaryPosterior {
    const n = labels.length;
    let f = new Array<number>(n).fill(0);
    let dual = new Array<number>(n).fill(0);
    let lastObjective = n * Math.LN2;

    // Newton's method in latent space (GPML algorithm 3.1). I + sqrt(W) K sqrt(W)
    // stays positive definite even for repeated samples and singular Gram matrices.
    for (let iter = 0; iter <= maxIter; iter++) {
        const residual = new Array<number>(n);
        const sqrtWeight = new Array<number>(n);
        for (let i = 0; i < n; i++) {
            const exp = Math.exp(-Math.abs(f[i]));
            sqrtWeight[i] = Math.sqrt(exp) / (1 + exp);
            // Compute y - sigmoid(f) without subtracting two numbers near one.
            residual[i] = labels[i]
                ? (f[i] >= 0 ? exp : 1) / (1 + exp)
                : -(f[i] >= 0 ? 1 : exp) / (1 + exp);
        }
        const lower = cholesky(K.map((row, i) => row.map((value, j) =>
            sqrtWeight[i] * value * sqrtWeight[j] + (i === j ? 1 : 0))));
        if (!lower) throw new Error('GaussianProcessClassifier posterior factorization failed; check kernel scaling');
        const b = residual.map((value, i) => value + sqrtWeight[i] ** 2 * f[i]);
        const rhs = K.map((row, i) => sqrtWeight[i] * dot(row, b));
        const correction = solveLowerTransposed(lower, solveLower(lower, rhs));
        const nextDual = b.map((value, i) => value - sqrtWeight[i] * correction[i]);
        const nextF = K.map(row => dot(row, nextDual));
        let change = 0, scale = 1;
        for (let i = 0; i < n; i++) {
            change = Math.max(change, Math.abs(nextF[i] - f[i]));
            scale = Math.max(scale, Math.abs(f[i]));
        }
        if (change <= tol * scale) return { dual: residual, sqrtWeight, lower };
        if (iter === maxIter) break;

        // Backtrack if the full Newton step increases the objective. Interpolating
        // both representations preserves f = K * dual along the entire step.
        let fraction = 1, accepted = false;
        while (fraction >= 2 ** -30) {
            const candidateF = fraction === 1 ? nextF : f.map((v, i) => v + fraction * (nextF[i] - v));
            const candidateDual = fraction === 1 ? nextDual : dual.map((v, i) => v + fraction * (nextDual[i] - v));
            const value = objective(candidateF, candidateDual, labels);
            if (Number.isFinite(value) && value <= lastObjective + 16 * Number.EPSILON * (1 + Math.abs(lastObjective))) {
                f = candidateF;
                dual = candidateDual;
                lastObjective = value;
                accepted = true;
                break;
            }
            fraction /= 2;
        }
        if (!accepted) throw new Error('GaussianProcessClassifier posterior line search failed; check kernel scaling');
    }
    throw new Error('GaussianProcessClassifier posterior did not converge; increase maxIterPredict or tol');
}

// Williams & Barber (1998), Appendix A: integrate the logistic link against the
// posterior Gaussian via a weighted sum of error functions. The terms cancel
// strongly, so evaluate erf to double precision.
const SCALES = [0.41, 0.4, 0.37, 0.44, 0.39];
const WEIGHTS = [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654];

function erf(x: number): number {
    if (Number.isNaN(x)) throw new Error('GaussianProcessClassifier probability calculation was not finite');
    if (Math.abs(x) >= 6) return Math.sign(x);
    const twiceSquared = 2 * x * x;
    let term = 1, sum = 1;
    for (let k = 1; ; k++) {
        term *= twiceSquared / (2 * k + 1);
        sum += term;
        if (term <= Number.EPSILON * sum) break;
    }
    return (2 / Math.sqrt(Math.PI)) * x * Math.exp(-x * x) * sum;
}

function positiveProbability(covariances: number[], priorVariance: number, model: BinaryPosterior): number {
    const mean = dot(covariances, model.dual);
    const v = solveLower(model.lower, covariances.map((value, i) => value * model.sqrtWeight[i]));
    const variance = Math.max(0, priorVariance - dot(v, v));
    let p = 0.5;
    for (let k = 0; k < SCALES.length; k++) {
        const scale = SCALES[k];
        p += 0.5 * WEIGHTS[k] * erf(scale * mean / Math.sqrt(1 + (2 * scale * scale) * variance));
    }
    return Math.max(0, Math.min(1, p));
}

export class GaussianProcessClassifier extends ClassifierBase {
    private kernel: Required<RBFKernelOptions>;
    private maxIterPredict: number;
    private tol: number;
    private trainX: number[][] = [];
    private classes: number[] = [];
    private posteriors: BinaryPosterior[] = [];

    constructor(props: GaussianProcessClassifierProps = {}) {
        super();
        this.kernel = resolveKernel(props);
        this.maxIterPredict = props.maxIterPredict ?? 100;
        this.tol = props.tol ?? 1e-8;
        if (!Number.isInteger(this.maxIterPredict) || this.maxIterPredict < 1) {
            throw new Error('maxIterPredict must be a positive integer');
        }
        if (!Number.isFinite(this.tol) || this.tol <= 0) throw new Error('tol must be positive and finite');
    }

    public getParams(): Params {
        return { ...resolveKernel(this.kernel), maxIterPredict: this.maxIterPredict, tol: this.tol };
    }

    public getClasses(): number[] {
        return this.classes.slice();
    }

    public fit(X: number[][], y: number[]): void {
        const nFeatures = validateRegressionData(X, y);
        validateLengthScale(this.kernel, nFeatures);
        const classes = Array.from(new Set(y)).sort((a, b) => a - b);
        if (classes.length < 2) throw new Error('GaussianProcessClassifier requires at least two classes');
        const K = kernelMatrix(X, this.kernel);
        const positives = classes.length === 2 ? [classes[1]] : classes;
        const posteriors = positives.map(label =>
            binaryPosterior(K, y.map(v => Number(v === label)), this.maxIterPredict, this.tol));
        this.trainX = X.map(row => row.slice());
        this.classes = classes;
        this.posteriors = posteriors;
    }

    private checkQueries(X: number[][]): void {
        if (!this.posteriors.length) throw new Error('GaussianProcessClassifier must be fitted before prediction');
        validateQuery(X, this.trainX[0].length);
    }

    private covariances(row: number[]): number[] {
        return this.trainX.map(train => rbfKernel(row, train, this.kernel));
    }

    public predictProba(X: number[][]): number[][] {
        this.checkQueries(X);
        return X.map(row => {
            const k = this.covariances(row);
            const positive = this.posteriors.map(model => positiveProbability(k, this.kernel.signalVariance, model));
            if (positive.length === 1) return [1 - positive[0], positive[0]];
            const sum = positive.reduce((s, p) => s + p, 0);
            return positive.map(p => sum === 0 ? 1 / positive.length : p / sum);
        });
    }

    public predict(X: number[][]): number[] {
        if (this.classes.length === 2) {
            this.checkQueries(X);
            return X.map(row => this.classes[Number(dot(this.covariances(row), this.posteriors[0].dual) > 0)]);
        }
        return this.predictProba(X).map(row => {
            let best = 0;
            for (let i = 1; i < row.length; i++) if (row[i] >= row[best]) best = i;
            return this.classes[best];
        });
    }
}

registerEstimator('GaussianProcessClassifier', GaussianProcessClassifier);
