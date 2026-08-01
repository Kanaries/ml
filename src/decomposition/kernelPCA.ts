import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { symmetricEigDecomposition } from '../discriminant_analysis/linalg';

export type KernelPCAKernel = 'linear' | 'poly' | 'rbf' | 'sigmoid' | 'cosine';
export interface KernelPCAProps {
    nComponents?: number | null;
    kernel?: KernelPCAKernel;
    gamma?: number;
    degree?: number;
    coef0?: number;
    alpha?: number;
    fitInverseTransform?: boolean;
    randomState?: number;
}

function solve(A: number[][], B: number[][]): number[][] {
    const n = A.length, outputs = B[0].length;
    const aug = A.map((row, i) => [...row, ...B[i]]);
    for (let col = 0; col < n; col++) {
        let pivot = col;
        for (let row = col + 1; row < n; row++) if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
        if (Math.abs(aug[pivot][col]) < 1e-14) throw new Error('kernel system is singular; increase alpha');
        [aug[pivot], aug[col]] = [aug[col], aug[pivot]];
        const scale = aug[col][col];
        for (let j = col; j < n + outputs; j++) aug[col][j] /= scale;
        for (let row = 0; row < n; row++) if (row !== col) {
            const factor = aug[row][col];
            for (let j = col; j < n + outputs; j++) aug[row][j] -= factor * aug[col][j];
        }
    }
    return aug.map(row => row.slice(n));
}

export class KernelPCA extends TransformerBase {
    private nComponents: number | null;
    private kernel: KernelPCAKernel;
    private gamma?: number;
    private degree: number;
    private coef0: number;
    private alpha: number;
    private fitInverseTransform: boolean;
    private randomState?: number;
    private resolvedGamma = 1;
    private fitX: number[][] = [];
    private eigenvaluesState: number[] = [];
    private eigenvectorsState: number[][] = [];
    private kernelColumnMeans: number[] = [];
    private kernelTotalMean = 0;
    private transformedFit: number[][] = [];
    private inverseDual: number[][] = [];

    constructor(props: KernelPCAProps = {}) {
        super();
        const { nComponents = null, kernel = 'linear', gamma, degree = 3, coef0 = 1, alpha = 1, fitInverseTransform = false, randomState } = props;
        if (nComponents !== null && (!Number.isInteger(nComponents) || nComponents < 1)) throw new Error('nComponents must be null or a positive integer');
        if (!['linear', 'poly', 'rbf', 'sigmoid', 'cosine'].includes(kernel) || (gamma !== undefined && (!Number.isFinite(gamma) || gamma <= 0)) || !Number.isInteger(degree) || degree < 1 || !Number.isFinite(coef0) || !Number.isFinite(alpha) || alpha < 0) throw new Error('invalid KernelPCA parameters');
        this.nComponents = nComponents; this.kernel = kernel; this.gamma = gamma; this.degree = degree;
        this.coef0 = coef0; this.alpha = alpha; this.fitInverseTransform = fitInverseTransform; this.randomState = randomState;
    }
    public getParams(): Params { return { nComponents: this.nComponents, kernel: this.kernel, gamma: this.gamma, degree: this.degree, coef0: this.coef0, alpha: this.alpha, fitInverseTransform: this.fitInverseTransform, randomState: this.randomState }; }
    private pair(a: number[], b: number[]): number {
        let dot = 0, d2 = 0, na = 0, nb = 0;
        for (let j = 0; j < a.length; j++) { dot += a[j] * b[j]; const d = a[j] - b[j]; d2 += d * d; na += a[j] ** 2; nb += b[j] ** 2; }
        if (this.kernel === 'linear') return dot;
        if (this.kernel === 'poly') return (this.resolvedGamma * dot + this.coef0) ** this.degree;
        if (this.kernel === 'rbf') return Math.exp(-this.resolvedGamma * d2);
        if (this.kernel === 'sigmoid') return Math.tanh(this.resolvedGamma * dot + this.coef0);
        return na === 0 || nb === 0 ? 0 : dot / Math.sqrt(na * nb);
    }
    private matrix(X: number[][], Y: number[][]): number[][] { return X.map(x => Y.map(y => this.pair(x, y))); }
    public fit(X: number[][]): void {
        if (X.length < 2 || X[0].length === 0 || X.some(row => row.length !== X[0].length || row.some(value => !Number.isFinite(value)))) throw new Error('X must be a finite rectangular matrix with at least two samples');
        this.fitX = X.map(row => row.slice());
        this.resolvedGamma = this.gamma ?? 1 / X[0].length;
        const K = this.matrix(X, X), n = X.length;
        const rowMeans = K.map(row => row.reduce((a, b) => a + b, 0) / n);
        this.kernelColumnMeans = Array.from({ length: n }, (_, j) => K.reduce((sum, row) => sum + row[j], 0) / n);
        this.kernelTotalMean = rowMeans.reduce((a, b) => a + b, 0) / n;
        const centered = K.map((row, i) => row.map((value, j) => value - rowMeans[i] - this.kernelColumnMeans[j] + this.kernelTotalMean));
        const requested = Math.min(this.nComponents ?? n, n);
        const eigen = symmetricEigDecomposition(centered);
        let values = eigen.values.slice(0, requested);
        let vectors = eigen.vectors.slice(0, requested).map(vector => {
            const oriented = vector.slice(); let pivot = 0;
            for (let i = 1; i < oriented.length; i++) if (Math.abs(oriented[i]) > Math.abs(oriented[pivot])) pivot = i;
            if (oriented[pivot] < 0) for (let i = 0; i < oriented.length; i++) oriented[i] *= -1;
            return oriented;
        });
        const maxEigenvalue = values[0] ?? 0; let minEigenvalue = 0;
        for (const value of values) minEigenvalue = Math.min(minEigenvalue, value);
        if (maxEigenvalue < 0 || (minEigenvalue < -1e-10 && minEigenvalue < -1e-5 * maxEigenvalue)) {
            throw new Error('kernel matrix has significant negative eigenvalues and is not positive semidefinite');
        }
        values = values.map(value => value <= 0 || value < maxEigenvalue * 1e-12 ? 0 : value);
        if (this.nComponents === null) {
            const keep = values.map((value, i) => ({ value, vector: vectors[i] })).filter(entry => entry.value > 0);
            values = keep.map(entry => entry.value); vectors = keep.map(entry => entry.vector);
        }
        this.eigenvaluesState = values;
        this.eigenvectorsState = vectors;
        this.transformedFit = Array.from({ length: n }, (_, i) => this.eigenvectorsState.map((vector, c) => vector[i] * Math.sqrt(this.eigenvaluesState[c])));
        if (this.fitInverseTransform) {
            const inverseKernel = this.matrix(this.transformedFit, this.transformedFit);
            for (let i = 0; i < n; i++) inverseKernel[i][i] += this.alpha;
            this.inverseDual = solve(inverseKernel, X);
        } else this.inverseDual = [];
    }
    public transform(X: number[][]): number[][] {
        if (this.fitX.length === 0) throw new Error('KernelPCA is not fitted');
        if (X.some(row => row.length !== this.fitX[0].length || row.some(value => !Number.isFinite(value)))) throw new Error('input feature count differs from fitted KernelPCA');
        const K = this.matrix(X, this.fitX);
        return K.map(row => {
            const mean = row.reduce((a, b) => a + b, 0) / row.length;
            const centered = row.map((value, j) => value - mean - this.kernelColumnMeans[j] + this.kernelTotalMean);
            return this.eigenvectorsState.map((vector, c) => this.eigenvaluesState[c] <= 1e-12 ? 0
                : vector.reduce((sum, value, j) => sum + value * centered[j], 0) / Math.sqrt(this.eigenvaluesState[c]));
        });
    }
    public fitTransform(X: number[][]): number[][] { this.fit(X); return this.transformedFit.map(row => row.slice()); }
    public inverseTransform(X: number[][]): number[][] {
        if (!this.fitInverseTransform || this.inverseDual.length === 0) throw new Error('inverseTransform requires fitInverseTransform=true');
        if (X.some(row => row.length !== this.transformedFit[0].length || row.some(value => !Number.isFinite(value)))) throw new Error('input must be finite and match the fitted component count');
        const K = this.matrix(X, this.transformedFit);
        return K.map(row => Array.from({ length: this.inverseDual[0].length }, (_, j) => row.reduce((sum, value, i) => sum + value * this.inverseDual[i][j], 0)));
    }
    public get eigenvalues(): number[] { return this.eigenvaluesState.slice(); }
    public get eigenvectors(): number[][] {
        if (this.eigenvectorsState.length === 0) return this.fitX.map(() => []);
        return Array.from({ length: this.eigenvectorsState[0].length }, (_, sample) => this.eigenvectorsState.map(vector => vector[sample]));
    }
}
registerEstimator('KernelPCA', KernelPCA);
