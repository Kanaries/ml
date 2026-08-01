import { RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { dot, pseudoInverseSymmetric, solveLinear, validateRegressionData } from '../utils/numerics';

export type KernelRidgeKernel = 'linear' | 'poly' | 'rbf' | 'sigmoid' | 'cosine';
export interface KernelRidgeProps { alpha?: number; kernel?: KernelRidgeKernel; gamma?: number | null; degree?: number; coef0?: number; }

export class KernelRidge extends RegressorBase {
    private alpha: number; private kernel: KernelRidgeKernel; private gamma: number | null; private degree: number; private coef0: number;
    private trainingState: number[][] = []; private dualCoefState: number[] = []; private nFeaturesState = 0;
    constructor(props: KernelRidgeProps = {}) {
        super();
        const { alpha = 1, kernel = 'linear', gamma = null, degree = 3, coef0 = 1 } = props;
        if (!Number.isFinite(alpha) || alpha < 0 || !['linear', 'poly', 'rbf', 'sigmoid', 'cosine'].includes(kernel) || gamma !== null && (!Number.isFinite(gamma) || gamma <= 0) || !Number.isInteger(degree) || degree < 1 || !Number.isFinite(coef0)) throw new Error('invalid KernelRidge parameters');
        this.alpha = alpha; this.kernel = kernel; this.gamma = gamma; this.degree = degree; this.coef0 = coef0;
    }
    public getParams(): Params { return { alpha: this.alpha, kernel: this.kernel, gamma: this.gamma, degree: this.degree, coef0: this.coef0 }; }
    private similarity(a: number[], b: number[]): number {
        const gamma = this.gamma ?? 1 / a.length, product = dot(a, b);
        if (this.kernel === 'linear') return product;
        if (this.kernel === 'poly') return (gamma * product + this.coef0) ** this.degree;
        if (this.kernel === 'sigmoid') return Math.tanh(gamma * product + this.coef0);
        if (this.kernel === 'cosine') return product / ((Math.sqrt(dot(a, a)) * Math.sqrt(dot(b, b))) || 1);
        return Math.exp(-gamma * a.reduce((sum, value, j) => sum + (value - b[j]) ** 2, 0));
    }
    public fit(X: number[][], y: number[]): void {
        this.nFeaturesState = validateRegressionData(X, y); this.trainingState = X.map(row => row.slice());
        const K = X.map((row, i) => X.map((other, j) => this.similarity(row, other) + (i === j ? this.alpha : 0)));
        try { this.dualCoefState = solveLinear(K, y); }
        catch { this.dualCoefState = pseudoInverseSymmetric(K).map(row => dot(row, y)); }
    }
    public predict(X: number[][]): number[] {
        if (this.trainingState.length === 0) throw new Error('KernelRidge is not fitted');
        if (X.some(row => row.length !== this.nFeaturesState || row.some(value => !Number.isFinite(value)))) throw new Error('input must be finite and match the fitted feature count');
        return X.map(row => this.trainingState.reduce((sum, training, i) => sum + this.dualCoefState[i] * this.similarity(row, training), 0));
    }
    public get dualCoef(): number[] { return this.dualCoefState.slice(); }
    public get XFit(): number[][] { return this.trainingState.map(row => row.slice()); }
}
registerEstimator('KernelRidge', KernelRidge);
