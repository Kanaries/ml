/** Squared exponential covariance with either one length scale or one per feature. */
export interface RBFKernelOptions {
    lengthScale?: number | number[];
    signalVariance?: number;
}

export function resolveKernel(options: RBFKernelOptions): Required<RBFKernelOptions> {
    const { lengthScale = 1, signalVariance = 1 } = options;
    const scales = Array.isArray(lengthScale) ? lengthScale : [lengthScale];
    if (scales.length === 0 || scales.some(value => !Number.isFinite(value) || value <= 0)) {
        throw new Error('lengthScale must contain finite positive values');
    }
    if (!Number.isFinite(signalVariance) || signalVariance <= 0) {
        throw new Error('signalVariance must be finite and positive');
    }
    return { lengthScale: Array.isArray(lengthScale) ? lengthScale.slice() : lengthScale, signalVariance };
}

export function validateLengthScale(kernel: Required<RBFKernelOptions>, nFeatures: number): void {
    if (Array.isArray(kernel.lengthScale) && kernel.lengthScale.length !== nFeatures) {
        throw new Error('lengthScale must have one entry per feature');
    }
}

export function validateQuery(X: number[][], nFeatures: number): void {
    if (X.some(row => row.length !== nFeatures || row.some(value => !Number.isFinite(value)))) {
        throw new Error('Query features must be finite and match the expected feature count');
    }
}

export function rbfKernel(a: number[], b: number[], kernel: Required<RBFKernelOptions>): number {
    let squaredDistance = 0;
    for (let j = 0; j < a.length; j++) {
        const scale = Array.isArray(kernel.lengthScale) ? kernel.lengthScale[j] : kernel.lengthScale;
        const diff = a[j] - b[j];
        const scaled = Number.isFinite(diff) ? diff / scale : a[j] / scale - b[j] / scale;
        squaredDistance += scaled * scaled;
    }
    return kernel.signalVariance * Math.exp(-0.5 * squaredDistance);
}

/** Symmetric Gram matrix; off-diagonal entries need only one kernel evaluation each. */
export function kernelMatrix(X: number[][], kernel: Required<RBFKernelOptions>): number[][] {
    const matrix = X.map(() => new Array<number>(X.length));
    for (let i = 0; i < X.length; i++) {
        matrix[i][i] = kernel.signalVariance;
        for (let j = 0; j < i; j++) {
            matrix[i][j] = matrix[j][i] = rbfKernel(X[i], X[j], kernel);
        }
    }
    return matrix;
}
