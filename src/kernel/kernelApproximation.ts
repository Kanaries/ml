import { Params, TransformerBase, registerEstimator } from '../base';
import { NumericMatrix, forEachNonZeroInRow, isCSRMatrix, matrixShape } from '../data';
import { symmetricEigDecomposition } from '../discriminant_analysis/linalg';
import { kernelFunction } from '../svm/smo';
import { mt19937 } from '../utils/mt19937';
import { dot, validateMatrix } from '../utils/numerics';

interface SamplingProps {
    nComponents?: number;
    randomState?: number;
}

function validateSampling(nComponents: number, randomState?: number): void {
    if (!Number.isSafeInteger(nComponents) || nComponents < 1) {
        throw new Error('nComponents must be a positive integer');
    }
    if (randomState !== undefined && (!Number.isInteger(randomState) || randomState < 0 || randomState > 0xffffffff)) {
        throw new Error('randomState must be an unsigned 32-bit integer');
    }
}

function validateNumericInput(X: NumericMatrix): void {
    if (isCSRMatrix(X)) {
        if (X.nRows === 0 || X.nCols === 0) throw new Error('X must be a non-empty matrix');
        for (let i = 0; i < X.nRows; i++) {
            X.forEachNonZeroInRow(i, (_, value) => {
                if (!Number.isFinite(value)) throw new Error('X must contain finite values');
            });
        }
    } else {
        validateMatrix(X);
    }
}

/** Global variance with a single implicit group for the zeros in sparse input. */
function scaledWeightDeviation(X: NumericMatrix): number {
    const [rows, columns] = matrixShape(X);
    let magnitude = 0;
    for (let row = 0; row < rows; row++) {
        forEachNonZeroInRow(X, row, (_, value) => { magnitude = Math.max(magnitude, Math.abs(value)); });
    }
    if (magnitude === 0) return Math.SQRT2;
    let count = 0, mean = 0, squaredDeviations = 0;
    for (let row = 0; row < rows; row++) {
        forEachNonZeroInRow(X, row, (_, value) => {
            count++;
            const scaled = value / magnitude;
            const delta = scaled - mean;
            mean += delta / count;
            squaredDeviations += delta * (scaled - mean);
        });
    }
    const total = rows * columns;
    if (count < total) squaredDeviations += mean * mean * (count / total) * (total - count);
    // Compute sqrt(2 * gamma) directly; squaring the original data can
    // overflow even when the final projected coordinates are well in range.
    return squaredDeviations > 0 ? Math.sqrt(2 * rows) / Math.sqrt(squaredDeviations) / magnitude : Math.SQRT2;
}

export interface RBFSamplerProps extends SamplingProps {
    gamma?: number | 'scale';
}

/** Random Fourier features for the Gaussian radial basis function kernel. */
export class RBFSampler extends TransformerBase<NumericMatrix> {
    public readonly acceptedInputKinds = ['dense', 'csr'] as const;
    private readonly nComponents: number;
    private readonly gamma: number | 'scale';
    private readonly randomState?: number;
    private weights: number[][] = [];
    private offsets: number[] = [];

    constructor({ nComponents = 100, gamma = 1, randomState }: RBFSamplerProps = {}) {
        super();
        validateSampling(nComponents, randomState);
        if (gamma !== 'scale' && (!Number.isFinite(gamma) || gamma < 0)) {
            throw new Error("gamma must be a non-negative finite number or 'scale'");
        }
        this.nComponents = nComponents;
        this.gamma = gamma;
        this.randomState = randomState;
    }

    public getParams(): Params {
        return { nComponents: this.nComponents, gamma: this.gamma, randomState: this.randomState };
    }

    public fit(X: NumericMatrix): void {
        validateNumericInput(X);
        const [, columns] = matrixShape(X);
        const std = this.gamma === 'scale' ? scaledWeightDeviation(X) : Math.SQRT2 * Math.sqrt(this.gamma);
        const random = mt19937(this.randomState);
        const weights = Array.from({ length: columns }, () =>
            Array.from({ length: this.nComponents }, () => {
                const weight = std * random.normal();
                if (!Number.isFinite(weight)) throw new Error('random weights exceed the numeric range');
                return weight;
            }));
        const offsets = Array.from({ length: this.nComponents }, () => 2 * Math.PI * random.uniform());
        this.weights = weights;
        this.offsets = offsets;
    }

    public transform(X: NumericMatrix): number[][] {
        if (this.weights.length === 0) throw new Error('RBFSampler is not fitted');
        validateNumericInput(X);
        const [rows, columns] = matrixShape(X);
        if (columns !== this.weights.length) throw new Error('X must match the fitted feature count');
        const scale = Math.sqrt(2 / this.nComponents);
        return Array.from({ length: rows }, (_, row) => {
            const projection = new Array<number>(this.nComponents).fill(0);
            forEachNonZeroInRow(X, row, (column, value) => {
                const weights = this.weights[column];
                for (let j = 0; j < weights.length; j++) projection[j] += value * weights[j];
            });
            return projection.map((value, j) => {
                if (!Number.isFinite(value)) throw new Error('random projection exceeds the numeric range');
                return scale * Math.cos(value + this.offsets[j]);
            });
        });
    }

    public get randomWeights(): number[][] { return this.weights.map(row => row.slice()); }
    public get randomOffset(): number[] { return this.offsets.slice(); }
}
registerEstimator('RBFSampler', RBFSampler);

export type NystroemKernel = 'linear' | 'rbf' | 'poly' | 'cosine';

export interface NystroemProps extends SamplingProps {
    kernel?: NystroemKernel;
    gamma?: number | null;
    degree?: number;
    coef0?: number;
}

/** Inverse square root from the eigensystem of a symmetric landmark kernel.
 * For an indefinite polynomial kernel the sign reproduces U diag(1/sqrt(S)) Vt.
 */
function normalizeLandmarks(gram: number[][]): number[][] {
    let magnitude = 0;
    for (const row of gram) {
        for (const value of row) magnitude = Math.max(magnitude, Math.abs(value));
    }
    if (!Number.isFinite(magnitude)) throw new Error('landmark kernel values exceed the numeric range');
    // Rescale before Jacobi computes its squared Frobenius norm.
    const scaled = gram.map(row => row.map(value => value / (magnitude || 1)));
    const { values, vectors } = symmetricEigDecomposition(scaled);
    const factors = values.map(value => {
        const sign = value < 0 ? -1 : 1;
        return sign / Math.max(1e-6, Math.sqrt(Math.abs(value)) * Math.sqrt(magnitude));
    });
    const size = gram.length;
    const result = Array.from({ length: size }, () => new Array<number>(size).fill(0));
    for (let k = 0; k < size; k++) {
        const basis = vectors[k];
        for (let i = 0; i < size; i++) {
            const factor = factors[k] * basis[i];
            for (let j = 0; j <= i; j++) result[i][j] += factor * basis[j];
        }
    }
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < i; j++) result[j][i] = result[i][j];
    }
    return result;
}

/** Nyström kernel feature map fitted on a uniform sample of landmarks. */
export class Nystroem extends TransformerBase {
    private readonly nComponents: number;
    private readonly randomState?: number;
    private readonly kernel: NystroemKernel;
    private readonly gamma: number | null;
    private readonly degree: number;
    private readonly coef0: number;
    private landmarks: number[][] = [];
    private indices: number[] = [];
    private basisNormalization: number[][] = [];

    constructor({ nComponents = 100, randomState, kernel = 'rbf', gamma = null, degree = 3, coef0 = 1 }: NystroemProps = {}) {
        super();
        validateSampling(nComponents, randomState);
        if (!['linear', 'rbf', 'poly', 'cosine'].includes(kernel)) throw new Error('unsupported Nystroem kernel');
        if (gamma !== null && (!Number.isFinite(gamma) || gamma < 0)) throw new Error('gamma must be null or non-negative and finite');
        if (!Number.isInteger(degree) || degree < 1) throw new Error('degree must be a positive integer');
        if (!Number.isFinite(coef0)) throw new Error('coef0 must be finite');
        this.nComponents = nComponents;
        this.randomState = randomState;
        this.kernel = kernel;
        this.gamma = gamma;
        this.degree = degree;
        this.coef0 = coef0;
    }

    public getParams(): Params {
        return { nComponents: this.nComponents, randomState: this.randomState, kernel: this.kernel,
            gamma: this.gamma, degree: this.degree, coef0: this.coef0 };
    }

    private similarity(a: number[], b: number[]): number {
        if (this.kernel === 'cosine') {
            let scaleA = 0, scaleB = 0;
            for (let j = 0; j < a.length; j++) {
                scaleA = Math.max(scaleA, Math.abs(a[j]));
                scaleB = Math.max(scaleB, Math.abs(b[j]));
            }
            if (scaleA === 0 || scaleB === 0) return 0;
            let squaredA = 0, squaredB = 0, product = 0;
            for (let j = 0; j < a.length; j++) {
                const x = a[j] / scaleA, y = b[j] / scaleB;
                squaredA += x * x;
                squaredB += y * y;
                product += x * y;
            }
            return product / Math.sqrt(squaredA * squaredB);
        }
        if (this.kernel === 'rbf' && this.gamma === 0) return 1;
        return kernelFunction(a, b, { kernel: this.kernel, gamma: this.gamma ?? 1 / a.length,
            degree: this.degree, coef0: this.coef0 });
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const random = mt19937(this.randomState);
        const permutation = Array.from({ length: X.length }, (_, i) => i);
        for (let i = permutation.length - 1; i > 0; i--) {
            const j = random.interval(i);
            [permutation[i], permutation[j]] = [permutation[j], permutation[i]];
        }
        const indices = permutation.slice(0, this.nComponents);
        const landmarks = indices.map(i => X[i].slice());
        const gram = landmarks.map(() => new Array<number>(indices.length));
        for (let i = 0; i < gram.length; i++) {
            for (let j = 0; j <= i; j++) gram[j][i] = gram[i][j] = this.similarity(landmarks[i], landmarks[j]);
        }
        const normalization = normalizeLandmarks(gram);
        this.indices = indices;
        this.landmarks = landmarks;
        this.basisNormalization = normalization;
    }

    public transform(X: number[][]): number[][] {
        if (this.landmarks.length === 0) throw new Error('Nystroem is not fitted');
        const columns = validateMatrix(X);
        if (columns !== this.landmarks[0].length) throw new Error('X must match the fitted feature count');
        return X.map(row => {
            const embedded = this.landmarks.map(landmark => this.similarity(row, landmark));
            return this.basisNormalization.map(weights => {
                const feature = dot(embedded, weights);
                if (!Number.isFinite(feature)) throw new Error('kernel features exceed the numeric range');
                return feature;
            });
        });
    }

    public get componentIndices(): number[] { return this.indices.slice(); }
    public get components(): number[][] { return this.landmarks.map(row => row.slice()); }
    public get normalization(): number[][] { return this.basisNormalization.map(row => row.slice()); }
}
registerEstimator('Nystroem', Nystroem);
