import { ClassifierBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { KernelConfig, KernelMatrix, KernelType, SMOSolution, kernelFunction, solveCSVC } from './smo';

export interface SVCProps {
    C?: number;
    kernel?: KernelType;
    /** 'scale' = 1/(n_features * Var(X)) like sklearn; 'auto' = 1/n_features */
    gamma?: number | 'scale' | 'auto';
    degree?: number;
    coef0?: number;
    tol?: number;
    /** hard limit on SMO pair updates; -1 (default) = run until convergence */
    maxIter?: number;
}

interface BinaryModel {
    /** training-set indices of this classifier's support vectors */
    indices: number[];
    /** alpha_i * y_i for each support vector */
    dualCoef: number[];
    b: number;
    /** class label voted for when the decision value is positive */
    positiveClass: number;
    negativeClass: number;
}

const SV_EPS = 1e-12;

/**
 * C-Support Vector Classification solved in the dual with SMO (libsvm-style
 * maximal-violating-pair working-set selection). Multiclass problems are
 * handled one-vs-one like sklearn/libsvm.
 */
export class SVC extends ClassifierBase {
    protected C: number;
    protected kernel: KernelType;
    protected gamma: number | 'scale' | 'auto';
    protected degree: number;
    protected coef0: number;
    protected tol: number;
    protected maxIter: number;
    protected gammaValue: number;
    protected trainX: number[][];
    protected trainY: number[];
    protected classes: number[];
    protected models: BinaryModel[];

    constructor(props: SVCProps = {}) {
        super();
        const {
            C = 1,
            kernel = 'rbf',
            gamma = 'scale',
            degree = 3,
            coef0 = 0,
            tol = 1e-3,
            maxIter = -1,
        } = props;
        if (!Number.isFinite(C) || C <= 0) {
            throw new Error('C must be a finite number > 0');
        }
        this.C = C;
        this.kernel = kernel;
        this.gamma = gamma;
        this.degree = degree;
        this.coef0 = coef0;
        this.tol = tol;
        this.maxIter = maxIter;
        this.gammaValue = 1;
        this.trainX = [];
        this.trainY = [];
        this.classes = [];
        this.models = [];
    }

    public getParams(): Params {
        return {
            C: this.C,
            kernel: this.kernel,
            gamma: this.gamma,
            degree: this.degree,
            coef0: this.coef0,
            tol: this.tol,
            maxIter: this.maxIter,
        };
    }

    protected kernelConfig(): KernelConfig {
        return { kernel: this.kernel, gamma: this.gammaValue, degree: this.degree, coef0: this.coef0 };
    }

    protected resolveGamma(X: number[][]): number {
        if (typeof this.gamma === 'number') {
            return this.gamma;
        }
        const nFeatures = X[0].length;
        if (this.gamma === 'auto') {
            return 1 / nFeatures;
        }
        // 'scale': 1 / (n_features * Var(X)) over ALL entries, like sklearn
        let sum = 0;
        let count = 0;
        for (const row of X) for (const v of row) { sum += v; count++; }
        const mean = sum / count;
        let varSum = 0;
        for (const row of X) for (const v of row) varSum += (v - mean) ** 2;
        const variance = varSum / count;
        return variance > 0 ? 1 / (nFeatures * variance) : 1;
    }

    /** solve one binary subproblem; y entries are +1/-1. Overridden by NuSVC. */
    protected solveBinary(X: number[][], y: number[]): SMOSolution {
        const K = new KernelMatrix(X, this.kernelConfig());
        return solveCSVC(K, y, this.C, this.tol, this.maxIter);
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
        this.trainX = trainX.map(row => row.slice());
        this.trainY = trainY.slice();
        this.classes = Array.from(new Set(this.trainY)).sort((a, b) => a - b);
        if (this.classes.length < 2) {
            throw new Error('SVC requires at least 2 classes');
        }
        this.gammaValue = this.resolveGamma(this.trainX);
        this.models = [];
        // one-vs-one: one binary classifier per unordered class pair
        for (let a = 0; a < this.classes.length; a++) {
            for (let b = a + 1; b < this.classes.length; b++) {
                const indices: number[] = [];
                for (let t = 0; t < this.trainY.length; t++) {
                    if (this.trainY[t] === this.classes[a] || this.trainY[t] === this.classes[b]) {
                        indices.push(t);
                    }
                }
                const subX = indices.map(t => this.trainX[t]);
                const subY = indices.map(t => (this.trainY[t] === this.classes[a] ? 1 : -1));
                const sol = this.solveBinary(subX, subY);
                const svIndices: number[] = [];
                const dualCoef: number[] = [];
                for (let t = 0; t < indices.length; t++) {
                    if (sol.alpha[t] > SV_EPS) {
                        svIndices.push(indices[t]);
                        dualCoef.push(sol.alpha[t] * subY[t]);
                    }
                }
                this.models.push({
                    indices: svIndices,
                    dualCoef,
                    b: sol.b,
                    positiveClass: this.classes[a],
                    negativeClass: this.classes[b],
                });
            }
        }
    }

    protected checkFitted(): void {
        if (this.models.length === 0) {
            throw new Error(`${this.constructor.name} must be fitted before calling predict`);
        }
    }

    protected decisionValue(model: BinaryModel, x: number[]): number {
        const cfg = this.kernelConfig();
        let sum = model.b;
        for (let k = 0; k < model.indices.length; k++) {
            sum += model.dualCoef[k] * kernelFunction(this.trainX[model.indices[k]], x, cfg);
        }
        return sum;
    }

    public predict(testX: number[][]): number[] {
        this.checkFitted();
        const results: number[] = [];
        for (const x of testX) {
            const votes = new Map<number, number>();
            for (const model of this.models) {
                const winner = this.decisionValue(model, x) > 0 ? model.positiveClass : model.negativeClass;
                votes.set(winner, (votes.get(winner) || 0) + 1);
            }
            let best = this.classes[0];
            let bestVotes = -1;
            for (const cls of this.classes) {
                const v = votes.get(cls) || 0;
                if (v > bestVotes) {
                    bestVotes = v;
                    best = cls;
                }
            }
            results.push(best);
        }
        return results;
    }

    /**
     * Binary-only decision function, sklearn convention: positive values
     * mean classes[1]. (The internal one-vs-one model is trained with
     * classes[0] as +1, hence the sign flip.)
     */
    public decisionFunction(testX: number[][]): number[] {
        this.checkFitted();
        if (this.classes.length !== 2) {
            throw new Error('decisionFunction is only implemented for binary classification');
        }
        return testX.map(x => -this.decisionValue(this.models[0], x));
    }

    /** sorted training-set indices of the support vectors (union over one-vs-one models) */
    public getSupportVectors(): number[] {
        this.checkFitted();
        const set = new Set<number>();
        for (const model of this.models) {
            for (const idx of model.indices) {
                set.add(idx);
            }
        }
        return Array.from(set).sort((a, b) => a - b);
    }

    /** number of support vectors per class, ordered like sklearn's n_support_ */
    public getNSupport(): number[] {
        const counts = new Map<number, number>(this.classes.map(c => [c, 0]));
        for (const idx of this.getSupportVectors()) {
            const cls = this.trainY[idx];
            counts.set(cls, (counts.get(cls) || 0) + 1);
        }
        return this.classes.map(c => counts.get(c) || 0);
    }
}
registerEstimator('SVC', SVC);
