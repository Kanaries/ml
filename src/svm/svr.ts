import { RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import {
    KernelConfig,
    KernelMatrix,
    KernelType,
    SVRSolution,
    kernelFunction,
    resolveGammaValue,
    solveEpsilonSVR,
} from './smo';

export interface SVRProps {
    kernel?: KernelType;
    degree?: number;
    /** 'scale' = 1/(n_features * Var(X)) like sklearn; 'auto' = 1/n_features */
    gamma?: number | 'scale' | 'auto';
    coef0?: number;
    tol?: number;
    C?: number;
    /** half-width of the epsilon-insensitive tube (no penalty for |f(x)-y| <= epsilon) */
    epsilon?: number;
    /** hard limit on SMO pair updates; -1 (default) = run until convergence */
    maxIter?: number;
}

const SV_EPS = 1e-12;

/**
 * epsilon-Support Vector Regression solved in the dual with SMO on the
 * libsvm 2n-variable formulation (alpha and alpha* pairs, maximal-violating-
 * pair working-set selection). The fitted model is
 * f(x) = sum_k dualCoef_k K(sv_k, x) + intercept with dualCoef = alpha - alpha*.
 */
export class SVR extends RegressorBase {
    protected kernel: KernelType;
    protected degree: number;
    protected gamma: number | 'scale' | 'auto';
    protected coef0: number;
    protected tol: number;
    protected C: number;
    protected epsilon: number;
    protected maxIter: number;
    protected gammaValue: number;
    /** support-vector rows (copies of the training samples with nonzero dual coef) */
    protected svX: number[][];
    /** training-set indices of the support vectors */
    protected supportIndices: number[];
    /** signed dual coefficients alpha_k - alpha*_k, aligned with svX */
    protected dualCoef: number[];
    protected intercept: number;
    protected fitted: boolean;

    constructor(props: SVRProps = {}) {
        super();
        const {
            kernel = 'rbf',
            degree = 3,
            gamma = 'scale',
            coef0 = 0,
            tol = 1e-3,
            C = 1,
            epsilon = 0.1,
            maxIter = -1,
        } = props;
        if (!Number.isFinite(C) || C <= 0) {
            throw new Error('C must be a finite number > 0');
        }
        if (!Number.isFinite(epsilon) || epsilon < 0) {
            throw new Error('epsilon must be a finite number >= 0');
        }
        this.kernel = kernel;
        this.degree = degree;
        this.gamma = gamma;
        this.coef0 = coef0;
        this.tol = tol;
        this.C = C;
        this.epsilon = epsilon;
        this.maxIter = maxIter;
        this.gammaValue = 1;
        this.svX = [];
        this.supportIndices = [];
        this.dualCoef = [];
        this.intercept = 0;
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            kernel: this.kernel,
            degree: this.degree,
            gamma: this.gamma,
            coef0: this.coef0,
            tol: this.tol,
            C: this.C,
            epsilon: this.epsilon,
            maxIter: this.maxIter,
        };
    }

    protected kernelConfig(): KernelConfig {
        return { kernel: this.kernel, gamma: this.gammaValue, degree: this.degree, coef0: this.coef0 };
    }

    /** solve the regression dual; overridden by NuSVR */
    protected solveDual(K: KernelMatrix, y: number[]): SVRSolution {
        return solveEpsilonSVR(K, y, this.C, this.epsilon, this.tol, this.maxIter);
    }

    public fit(trainX: number[][], trainY: number[]): void {
        if (trainX.length === 0 || trainY.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (trainX.length !== trainY.length) {
            throw new Error('X and y must have the same length');
        }
        // deep copy: predictions must not silently move when the caller
        // mutates the array it passed to fit
        const X = trainX.map(row => row.slice());
        this.gammaValue = resolveGammaValue(this.gamma, X);
        const K = new KernelMatrix(X, this.kernelConfig());
        const sol = this.solveDual(K, trainY);
        this.svX = [];
        this.supportIndices = [];
        this.dualCoef = [];
        for (let i = 0; i < X.length; i++) {
            if (Math.abs(sol.coef[i]) > SV_EPS) {
                this.svX.push(X[i]);
                this.supportIndices.push(i);
                this.dualCoef.push(sol.coef[i]);
            }
        }
        this.intercept = sol.b;
        this.fitted = true;
    }

    protected checkFitted(): void {
        if (!this.fitted) {
            throw new Error(`${this.constructor.name} must be fitted before calling predict`);
        }
    }

    public predict(testX: number[][]): number[] {
        this.checkFitted();
        const cfg = this.kernelConfig();
        const results: number[] = [];
        for (const x of testX) {
            let sum = this.intercept;
            for (let k = 0; k < this.svX.length; k++) {
                sum += this.dualCoef[k] * kernelFunction(this.svX[k], x, cfg);
            }
            results.push(sum);
        }
        return results;
    }

    /** sorted training-set indices of the support vectors (sklearn's support_) */
    public getSupportVectors(): number[] {
        this.checkFitted();
        return this.supportIndices.slice();
    }
}
registerEstimator('SVR', SVR);
