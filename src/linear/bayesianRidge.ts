import { RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { symmetricEigDecomposition } from '../discriminant_analysis/linalg';
import { dot, inverseMatrix, validateRegressionData } from '../utils/numerics';

interface BayesianProps { maxIter?: number; tol?: number; alpha1?: number; alpha2?: number; lambda1?: number; lambda2?: number; fitIntercept?: boolean; }

export class BayesianRidge extends RegressorBase {
    private maxIter: number; private tol: number; private alpha1: number; private alpha2: number; private lambda1: number; private lambda2: number; private fitIntercept: boolean;
    private coefState: number[] = []; private interceptState = 0; private alphaState = 1; private lambdaState = 1; private sigmaState: number[][] = []; private nIterState = 0; private xOffsetState: number[] = [];
    constructor(props: BayesianProps = {}) { super(); const { maxIter = 300, tol = 1e-3, alpha1 = 1e-6, alpha2 = 1e-6, lambda1 = 1e-6, lambda2 = 1e-6, fitIntercept = true } = props; if (!Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(tol) || tol <= 0 || [alpha1, alpha2, lambda1, lambda2].some(value => !Number.isFinite(value) || value < 0)) throw new Error('invalid BayesianRidge parameters'); this.maxIter = maxIter; this.tol = tol; this.alpha1 = alpha1; this.alpha2 = alpha2; this.lambda1 = lambda1; this.lambda2 = lambda2; this.fitIntercept = fitIntercept; }
    public getParams(): Params { return { maxIter: this.maxIter, tol: this.tol, alpha1: this.alpha1, alpha2: this.alpha2, lambda1: this.lambda1, lambda2: this.lambda2, fitIntercept: this.fitIntercept }; }
    public fit(X: number[][], y: number[]): void {
        const p = validateRegressionData(X, y), n = X.length; this.xOffsetState = this.fitIntercept ? Array.from({ length: p }, (_, j) => X.reduce((sum, row) => sum + row[j], 0) / n) : new Array(p).fill(0); const yMean = this.fitIntercept ? y.reduce((a, b) => a + b, 0) / n : 0;
        const centered = X.map(row => row.map((value, j) => value - this.xOffsetState[j])), target = y.map(value => value - yMean), gram = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => centered.reduce((sum, row) => sum + row[i] * row[j], 0))), xty = Array.from({ length: p }, (_, j) => centered.reduce((sum, row, i) => sum + row[j] * target[i], 0));
        const eigen = symmetricEigDecomposition(gram).values, variance = target.reduce((sum, value) => sum + value * value, 0) / n; let alpha = 1 / (variance + Number.EPSILON), lambda = 1, previous: number[] | null = null, coef = new Array(p).fill(0);
        for (this.nIterState = 1; this.nIterState <= this.maxIter; this.nIterState++) {
            const sigma = inverseMatrix(gram.map((row, i) => row.map((value, j) => alpha * value + (i === j ? lambda : 0)))); coef = sigma.map(row => alpha * dot(row, xty));
            const residual = target.map((value, i) => value - dot(centered[i], coef)), sse = dot(residual, residual), gamma = eigen.reduce((sum, value) => sum + alpha * value / (lambda + alpha * value), 0);
            lambda = (gamma + 2 * this.lambda1) / (dot(coef, coef) + 2 * this.lambda2); alpha = (n - gamma + 2 * this.alpha1) / (sse + 2 * this.alpha2);
            if (previous && coef.reduce((sum, value, i) => sum + Math.abs(value - previous![i]), 0) < this.tol) break; previous = coef.slice();
        }
        this.nIterState = Math.min(this.nIterState, this.maxIter); this.alphaState = alpha; this.lambdaState = lambda; this.sigmaState = inverseMatrix(gram.map((row, i) => row.map((value, j) => alpha * value + (i === j ? lambda : 0)))); this.coefState = this.sigmaState.map(row => alpha * dot(row, xty)); this.interceptState = yMean - dot(this.xOffsetState, this.coefState);
    }
    public predict(X: number[][]): number[] { if (this.coefState.length === 0) throw new Error('BayesianRidge is not fitted'); return X.map(row => this.interceptState + dot(row, this.coefState)); }
    public predictStd(X: number[][]): number[] { return X.map(row => { const centered = row.map((value, j) => value - this.xOffsetState[j]); return Math.sqrt(dot(centered, this.sigmaState.map(matrixRow => dot(matrixRow, centered))) + 1 / this.alphaState); }); }
    public get coef(): number[] { return this.coefState.slice(); } public get intercept(): number { return this.interceptState; } public get alpha(): number { return this.alphaState; } public get lambda(): number { return this.lambdaState; } public get sigma(): number[][] { return this.sigmaState.map(row => row.slice()); } public get nIter(): number { return this.nIterState; }
}
registerEstimator('BayesianRidge', BayesianRidge);

export interface ARDRegressionProps extends BayesianProps { thresholdLambda?: number; }
export class ARDRegression extends RegressorBase {
    private maxIter: number; private tol: number; private alpha1: number; private alpha2: number; private lambda1: number; private lambda2: number; private thresholdLambda: number; private fitIntercept: boolean;
    private coefState: number[] = []; private interceptState = 0; private alphaState = 1; private lambdaState: number[] = []; private sigmaState: number[][] = []; private nIterState = 0; private xOffsetState: number[] = [];
    constructor(props: ARDRegressionProps = {}) { super(); const { maxIter = 300, tol = 1e-3, alpha1 = 1e-6, alpha2 = 1e-6, lambda1 = 1e-6, lambda2 = 1e-6, thresholdLambda = 1e4, fitIntercept = true } = props; if (!Number.isInteger(maxIter) || maxIter < 1 || !Number.isFinite(tol) || tol <= 0 || !Number.isFinite(thresholdLambda) || thresholdLambda <= 0) throw new Error('invalid ARDRegression parameters'); this.maxIter = maxIter; this.tol = tol; this.alpha1 = alpha1; this.alpha2 = alpha2; this.lambda1 = lambda1; this.lambda2 = lambda2; this.thresholdLambda = thresholdLambda; this.fitIntercept = fitIntercept; }
    public getParams(): Params { return { maxIter: this.maxIter, tol: this.tol, alpha1: this.alpha1, alpha2: this.alpha2, lambda1: this.lambda1, lambda2: this.lambda2, thresholdLambda: this.thresholdLambda, fitIntercept: this.fitIntercept }; }
    public fit(X: number[][], y: number[]): void {
        const p = validateRegressionData(X, y), n = X.length; this.xOffsetState = this.fitIntercept ? Array.from({ length: p }, (_, j) => X.reduce((sum, row) => sum + row[j], 0) / n) : new Array(p).fill(0); const yMean = this.fitIntercept ? y.reduce((a, b) => a + b, 0) / n : 0, centered = X.map(row => row.map((value, j) => value - this.xOffsetState[j])), target = y.map(value => value - yMean);
        let alpha = 1 / (target.reduce((sum, value) => sum + value * value, 0) / n + Number.EPSILON), lambda = new Array(p).fill(1), coef = new Array(p).fill(0), previous: number[] | null = null, keep = new Array(p).fill(true), sigma: number[][] = [];
        for (this.nIterState = 1; this.nIterState <= this.maxIter; this.nIterState++) {
            const active = keep.map((value, i) => value ? i : -1).filter(i => i >= 0), gram = active.map(a => active.map(b => centered.reduce((sum, row) => sum + row[a] * row[b], 0))), precision = gram.map((row, i) => row.map((value, j) => alpha * value + (i === j ? lambda[active[i]] : 0))); sigma = inverseMatrix(precision);
            const xty = active.map(feature => centered.reduce((sum, row, i) => sum + row[feature] * target[i], 0)); active.forEach((feature, i) => coef[feature] = alpha * dot(sigma[i], xty));
            const residual = target.map((value, i) => value - dot(centered[i], coef)), gamma = active.map((feature, i) => 1 - lambda[feature] * sigma[i][i]); active.forEach((feature, i) => lambda[feature] = (gamma[i] + 2 * this.lambda1) / (coef[feature] ** 2 + 2 * this.lambda2)); alpha = (n - gamma.reduce((a, b) => a + b, 0) + 2 * this.alpha1) / (dot(residual, residual) + 2 * this.alpha2);
            keep = lambda.map(value => value < this.thresholdLambda); keep.forEach((value, i) => { if (!value) coef[i] = 0; }); if (previous && coef.reduce((sum, value, i) => sum + Math.abs(value - previous![i]), 0) < this.tol) break; previous = coef.slice(); if (!keep.some(Boolean)) break;
        }
        this.nIterState = Math.min(this.nIterState, this.maxIter); this.coefState = coef; this.alphaState = alpha; this.lambdaState = lambda; this.sigmaState = sigma; this.interceptState = yMean - dot(this.xOffsetState, coef);
    }
    public predict(X: number[][]): number[] { if (this.coefState.length === 0) throw new Error('ARDRegression is not fitted'); return X.map(row => this.interceptState + dot(row, this.coefState)); }
    public get coef(): number[] { return this.coefState.slice(); } public get intercept(): number { return this.interceptState; } public get alpha(): number { return this.alphaState; } public get lambda(): number[] { return this.lambdaState.slice(); } public get sigma(): number[][] { return this.sigmaState.map(row => row.slice()); } public get nIter(): number { return this.nIterState; }
}
registerEstimator('ARDRegression', ARDRegression);
