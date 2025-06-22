import { ClassifierBase } from '../base';

export type KernelType = 'rbf' | 'knn' | ((X: number[][], Y: number[][]) => number[][]);

export interface LabelSpreadingOptions {
    kernel?: KernelType;
    gamma?: number;
    nNeighbors?: number;
    alpha?: number;
    maxIter?: number;
    tol?: number;
}

function argMax(arr: number[]): number {
    let m = -Infinity;
    let idx = 0;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > m) {
            m = arr[i];
            idx = i;
        }
    }
    return idx;
}

function rowNormalize(mat: number[][]): number[][] {
    return mat.map(row => {
        const s = row.reduce((a, b) => a + b, 0);
        if (s === 0) return row.map(() => 0);
        return row.map(v => v / s);
    });
}

function multiply(A: number[][], B: number[][]): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < A.length; i++) {
        const row: number[] = new Array(B[0].length).fill(0);
        for (let k = 0; k < B.length; k++) {
            for (let j = 0; j < B[0].length; j++) {
                row[j] += A[i][k] * B[k][j];
            }
        }
        result.push(row);
    }
    return result;
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

    private rbfKernel(X: number[][], Y: number[][]): number[][] {
        const result: number[][] = [];
        for (const x of X) {
            const row: number[] = [];
            for (const y of Y) {
                let d = 0;
                for (let i = 0; i < x.length; i++) {
                    const diff = x[i] - y[i];
                    d += diff * diff;
                }
                row.push(Math.exp(-this.gamma * d));
            }
            result.push(row);
        }
        return result;
    }

    private knnKernel(X: number[][], Y: number[][]): number[][] {
        const result: number[][] = [];
        for (const x of X) {
            const dists: { d: number; i: number }[] = [];
            for (let i = 0; i < Y.length; i++) {
                const y = Y[i];
                let d = 0;
                for (let j = 0; j < x.length; j++) {
                    const diff = x[j] - y[j];
                    d += diff * diff;
                }
                dists.push({ d: Math.sqrt(d), i });
            }
            dists.sort((a, b) => a.d - b.d);
            const row = new Array(Y.length).fill(0);
            for (let k = 0; k < Math.min(this.nNeighbors, dists.length); k++) {
                row[dists[k].i] = 1;
            }
            result.push(row);
        }
        return result;
    }

    private getKernel(): (X: number[][], Y: number[][]) => number[][] {
        if (typeof this.kernel === 'function') return this.kernel;
        if (this.kernel === 'rbf') return this.rbfKernel.bind(this);
        return this.knnKernel.bind(this);
    }

    public fit(X: number[][], y: number[]): void {
        this.X = X;
        const kernelFunc = this.getKernel();
        let W = kernelFunc(X, X);
        // make symmetric
        for (let i = 0; i < W.length; i++) {
            for (let j = i + 1; j < W.length; j++) {
                const val = Math.max(W[i][j], W[j][i]);
                W[i][j] = val;
                W[j][i] = val;
            }
        }
        const degree = W.map(row => row.reduce((a, b) => a + b, 0));
        const DinvSqrt = degree.map(d => d === 0 ? 0 : 1 / Math.sqrt(d));
        const T: number[][] = [];
        for (let i = 0; i < W.length; i++) {
            const row: number[] = [];
            for (let j = 0; j < W.length; j++) {
                row.push(DinvSqrt[i] * W[i][j] * DinvSqrt[j]);
            }
            T.push(row);
        }
        const S = rowNormalize(T);
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
        for (let iter = 0; iter < this.maxIter; iter++) {
            let Fnew = multiply(S, F);
            for (let i = 0; i < Fnew.length; i++) {
                for (let j = 0; j < Fnew[i].length; j++) {
                    Fnew[i][j] = this.alpha * Fnew[i][j] + (1 - this.alpha) * Y[i][j];
                }
            }
            let diff = 0;
            for (let i = 0; i < F.length; i++) {
                for (let j = 0; j < F[i].length; j++) {
                    diff += Math.abs(Fnew[i][j] - F[i][j]);
                }
            }
            F = Fnew;
            if (diff < this.tol) {
                this.nIter = iter + 1;
                break;
            }
            if (iter === this.maxIter - 1) {
                this.nIter = this.maxIter;
            }
        }
        this.labelDistributions = F;
        this.transduction = F.map(row => this.classes[argMax(row)]);
    }

    public predict(testX: number[][]): number[] {
        return this.predictProba(testX).map(row => this.classes[argMax(row)]);
    }

    public predictProba(testX: number[][]): number[][] {
        const kernelFunc = this.getKernel();
        const W = rowNormalize(kernelFunc(testX, this.X));
        const probs = multiply(W, this.labelDistributions);
        return probs;
    }
}
