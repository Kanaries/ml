import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { argMax, knnKernel, multiply, rbfKernel, rowNormalize, transpose } from './utils';

export type KernelType = 'rbf' | 'knn' | ((X: number[][], Y: number[][]) => number[][]);

export interface LabelSpreadingOptions {
    kernel?: KernelType;
    gamma?: number;
    nNeighbors?: number;
    alpha?: number;
    maxIter?: number;
    tol?: number;
}

export class LabelSpreading extends ClassifierBase {
    private kernel: KernelType;
    private gamma: number;
    private nNeighbors: number;
    private alpha: number;
    private maxIter: number;
    private tol: number;

    private X: number[][] = [];
    private classes: number[] = [];
    private labelDistributions: number[][] = [];
    private transduction: number[] = [];
    private nIter: number = 0;

    constructor(options: LabelSpreadingOptions = {}) {
        super();
        const { kernel = 'rbf', gamma = 20, nNeighbors = 7, alpha = 0.2, maxIter = 30, tol = 1e-3 } = options;
        this.kernel = kernel;
        this.gamma = gamma;
        this.nNeighbors = nNeighbors;
        this.alpha = alpha;
        this.maxIter = maxIter;
        this.tol = tol;
    }

    public getParams(): Params {
        return {
            kernel: this.kernel,
            gamma: this.gamma,
            nNeighbors: this.nNeighbors,
            alpha: this.alpha,
            maxIter: this.maxIter,
            tol: this.tol,
        };
    }

    private getKernel(): (X: number[][], Y: number[][]) => number[][] {
        if (typeof this.kernel === 'function') return this.kernel;
        if (this.kernel === 'rbf') return (X, Y) => rbfKernel(X, Y, this.gamma);
        return (X, Y) => knnKernel(X, Y, this.nNeighbors);
    }

    public fit(X: number[][], y: number[]): void {
        this.X = X;
        const kernelFunc = this.getKernel();
        // sklearn LabelSpreading._build_graph: -laplacian(affinity, normed=True)
        // with the diagonal set to zero. scipy's normed laplacian zeroes the
        // diagonal of the affinity before computing degrees, so the graph is
        // D^{-1/2} W D^{-1/2} with diag(W) = 0 and D from the zero-diagonal W.
        const affinity = kernelFunc(X, X);
        const n = affinity.length;
        const degree: number[] = [];
        for (let i = 0; i < n; i++) {
            let d = 0;
            for (let j = 0; j < n; j++) {
                if (j !== i) d += affinity[i][j];
            }
            degree.push(d);
        }
        const dInvSqrt = degree.map(d => (d === 0 ? 0 : 1 / Math.sqrt(d)));
        const graph: number[][] = [];
        for (let i = 0; i < n; i++) {
            const row: number[] = [];
            for (let j = 0; j < n; j++) {
                row.push(i === j ? 0 : dInvSqrt[i] * affinity[i][j] * dInvSqrt[j]);
            }
            graph.push(row);
        }
        const labeledMask = y.map(v => v !== -1);
        this.classes = Array.from(new Set(y.filter(v => v !== -1))).sort((a, b) => a - b);
        const classIndex = new Map<number, number>();
        this.classes.forEach((c, idx) => classIndex.set(c, idx));
        const Y: number[][] = [];
        for (let i = 0; i < X.length; i++) {
            const row = new Array(this.classes.length).fill(0);
            if (labeledMask[i]) {
                row[classIndex.get(y[i])!] = 1;
            }
            Y.push(row);
        }
        // soft clamping: y_static = (1 - alpha) * Y
        const yStatic = Y.map(row => row.map(v => (1 - this.alpha) * v));
        let F = Y.map(r => r.slice());
        let Fprevious = F.map(r => r.map(() => 0));
        for (this.nIter = 0; this.nIter < this.maxIter; this.nIter++) {
            let diff = 0;
            for (let i = 0; i < F.length; i++) {
                for (let j = 0; j < F[i].length; j++) {
                    diff += Math.abs(F[i][j] - Fprevious[i][j]);
                }
            }
            if (diff < this.tol) break;
            Fprevious = F;
            F = multiply(graph, F);
            for (let i = 0; i < F.length; i++) {
                for (let j = 0; j < F[i].length; j++) {
                    F[i][j] = this.alpha * F[i][j] + yStatic[i][j];
                }
            }
        }
        this.labelDistributions = rowNormalize(F);
        this.transduction = this.labelDistributions.map(row => this.classes[argMax(row)]);
    }

    public predict(testX: number[][]): number[] {
        return this.predictProba(testX).map(row => this.classes[argMax(row)]);
    }

    public predictProba(testX: number[][]): number[][] {
        let W: number[][];
        if (this.kernel === 'knn') {
            W = knnKernel(testX, this.X, this.nNeighbors);
        } else {
            // sklearn computes kernel(train, test) and transposes it.
            W = transpose(this.getKernel()(this.X, testX));
        }
        return rowNormalize(multiply(W, this.labelDistributions));
    }

    public getTransduction(): number[] {
        return this.transduction;
    }

    public getNIter(): number {
        return this.nIter;
    }
}
registerEstimator('LabelSpreading', LabelSpreading);
