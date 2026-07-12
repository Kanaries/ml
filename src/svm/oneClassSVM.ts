import { OutlierBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import {
    KernelConfig,
    KernelMatrix,
    KernelType,
    kernelFunction,
    resolveGammaValue,
    solveOneClass,
} from './smo';

export interface OneClassSVMProps {
    kernel?: KernelType;
    degree?: number;
    /** 'scale' = 1/(n_features * Var(X)) like sklearn; 'auto' = 1/n_features */
    gamma?: number | 'scale' | 'auto';
    coef0?: number;
    tol?: number;
    /**
     * upper bound on the fraction of training errors (points classified as
     * outliers), lower bound on the fraction of support vectors
     */
    nu?: number;
    /** hard limit on SMO pair updates; -1 (default) = run until convergence */
    maxIter?: number;
}

const SV_EPS = 1e-12;

/**
 * One-class SVM for unsupervised outlier/novelty detection (Schölkopf's
 * formulation, solved with SMO on the libsvm one-class dual).
 *
 * `decisionFunction(X)` returns f(x) = sum_i alpha_i K(sv_i, x) - rho;
 * `predict(X)` follows the sklearn label convention: **+1 = inlier,
 * -1 = outlier** (positive decision value = inlier).
 *
 * NOTE: this differs from this library's `IsolationForest`, whose `predict`
 * returns 1 for outliers and 0 for inliers.
 */
export class OneClassSVM extends OutlierBase {
    private kernel: KernelType;
    private degree: number;
    private gamma: number | 'scale' | 'auto';
    private coef0: number;
    private tol: number;
    private nu: number;
    private maxIter: number;
    private gammaValue: number;
    /** support-vector rows (copies of training samples with alpha_i > 0) */
    private svX: number[][];
    /** training-set indices of the support vectors */
    private supportIndices: number[];
    /** unsigned dual coefficients alpha_i, aligned with svX */
    private dualCoef: number[];
    /** decision offset: f(x) = sum alpha_i K(sv_i, x) - rho */
    private rho: number;
    private fitted: boolean;

    constructor(props: OneClassSVMProps = {}) {
        super();
        const {
            kernel = 'rbf',
            degree = 3,
            gamma = 'scale',
            coef0 = 0,
            tol = 1e-3,
            nu = 0.5,
            maxIter = -1,
        } = props;
        if (!Number.isFinite(nu) || nu <= 0 || nu > 1) {
            throw new Error('nu <= 0 or nu > 1');
        }
        this.kernel = kernel;
        this.degree = degree;
        this.gamma = gamma;
        this.coef0 = coef0;
        this.tol = tol;
        this.nu = nu;
        this.maxIter = maxIter;
        this.gammaValue = 1;
        this.svX = [];
        this.supportIndices = [];
        this.dualCoef = [];
        this.rho = 0;
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            kernel: this.kernel,
            degree: this.degree,
            gamma: this.gamma,
            coef0: this.coef0,
            tol: this.tol,
            nu: this.nu,
            maxIter: this.maxIter,
        };
    }

    private kernelConfig(): KernelConfig {
        return { kernel: this.kernel, gamma: this.gammaValue, degree: this.degree, coef0: this.coef0 };
    }

    public fit(samplesX: number[][]): void {
        if (samplesX.length === 0) {
            throw new Error('X must be non-empty');
        }
        // deep copy: predictions must not silently move when the caller
        // mutates the array it passed to fit
        const X = samplesX.map(row => row.slice());
        this.gammaValue = resolveGammaValue(this.gamma, X);
        const K = new KernelMatrix(X, this.kernelConfig());
        const sol = solveOneClass(K, this.nu, this.tol, this.maxIter);
        this.svX = [];
        this.supportIndices = [];
        this.dualCoef = [];
        for (let i = 0; i < X.length; i++) {
            if (sol.alpha[i] > SV_EPS) {
                this.svX.push(X[i]);
                this.supportIndices.push(i);
                this.dualCoef.push(sol.alpha[i]);
            }
        }
        this.rho = sol.rho;
        this.fitted = true;
    }

    private checkFitted(): void {
        if (!this.fitted) {
            throw new Error('OneClassSVM must be fitted before calling predict');
        }
    }

    /** signed distance-like score; positive = inlier, negative = outlier */
    public decisionFunction(samplesX: number[][]): number[] {
        this.checkFitted();
        const cfg = this.kernelConfig();
        const results: number[] = [];
        for (const x of samplesX) {
            let sum = -this.rho;
            for (let k = 0; k < this.svX.length; k++) {
                sum += this.dualCoef[k] * kernelFunction(this.svX[k], x, cfg);
            }
            results.push(sum);
        }
        return results;
    }

    /** sklearn convention: +1 = inlier, -1 = outlier */
    public predict(samplesX: number[][]): number[] {
        return this.decisionFunction(samplesX).map(d => (d > 0 ? 1 : -1));
    }

    /** sorted training-set indices of the support vectors (sklearn's support_) */
    public getSupportVectors(): number[] {
        this.checkFitted();
        return this.supportIndices.slice();
    }
}
registerEstimator('OneClassSVM', OneClassSVM);
