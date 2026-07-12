import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { argMax, knnKernel, multiply, rbfKernel, rowNormalize, transpose } from './utils';

export type KernelType = 'rbf' | 'knn' | ((X: number[][], Y: number[][]) => number[][]);

export interface LabelPropagationOptions {
    kernel?: KernelType;
    gamma?: number;
    nNeighbors?: number;
    maxIter?: number;
    tol?: number;
}

export class LabelPropagation extends ClassifierBase {
    private kernel: KernelType;
    private gamma: number;
    private nNeighbors: number;
    private maxIter: number;
    private tol: number;

    private X: number[][] = [];
    private classes: number[] = [];
    private labelDistributions: number[][] = [];
    private transduction: number[] = [];
    private nIter: number = 0;

    constructor(options: LabelPropagationOptions = {}) {
        super();
        const { kernel = 'rbf', gamma = 20, nNeighbors = 7, maxIter = 1000, tol = 1e-3 } = options;
        this.kernel = kernel;
        this.gamma = gamma;
        this.nNeighbors = nNeighbors;
        this.maxIter = maxIter;
        this.tol = tol;
    }

    public getParams(): Params {
        return {
            kernel: this.kernel,
            gamma: this.gamma,
            nNeighbors: this.nNeighbors,
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
        // sklearn LabelPropagation._build_graph: row-normalized affinity
        // matrix with the diagonal kept (rbf k(x, x) = 1 counts in the sum).
        const affinity = kernelFunc(X, X);
        const graph = rowNormalize(affinity);
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
            // propagate, row-normalize (zero-sum rows stay zero), then
            // hard-clamp labeled rows back to their one-hot targets.
            F = rowNormalize(multiply(graph, F));
            for (let i = 0; i < X.length; i++) {
                if (labeledMask[i]) {
                    F[i] = Y[i].slice();
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
registerEstimator('LabelPropagation', LabelPropagation);
