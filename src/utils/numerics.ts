import { symmetricEigDecomposition } from '../discriminant_analysis/linalg';

export function dot(a: number[], b: number[]): number {
    return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

export function transpose(A: number[][]): number[][] {
    if (A.length === 0) return [];
    return Array.from({ length: A[0].length }, (_, j) => A.map(row => row[j]));
}

export function matMul(A: number[][], B: number[][]): number[][] {
    if (A.length === 0 || B.length === 0) return [];
    return A.map(row => Array.from({ length: B[0].length }, (_, j) => row.reduce((sum, value, k) => sum + value * B[k][j], 0)));
}

export function identity(n: number): number[][] {
    return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
}

export function solveLinear(A: number[][], b: number[], tolerance = 1e-14): number[] {
    const n = A.length;
    if (n === 0 || b.length !== n || A.some(row => row.length !== n)) throw new Error('solveLinear requires a square matrix');
    const augmented = A.map((row, i) => [...row, b[i]]);
    let scale = 0; for (const row of A) for (const value of row) scale = Math.max(scale, Math.abs(value));
    for (let col = 0; col < n; col++) {
        let pivot = col;
        for (let row = col + 1; row < n; row++) if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivot][col])) pivot = row;
        if (Math.abs(augmented[pivot][col]) <= tolerance * Math.max(1, scale)) throw new Error('matrix is singular');
        [augmented[col], augmented[pivot]] = [augmented[pivot], augmented[col]];
        const divisor = augmented[col][col];
        for (let j = col; j <= n; j++) augmented[col][j] /= divisor;
        for (let row = 0; row < n; row++) if (row !== col) {
            const factor = augmented[row][col];
            if (factor === 0) continue;
            for (let j = col; j <= n; j++) augmented[row][j] -= factor * augmented[col][j];
        }
    }
    return augmented.map(row => row[n]);
}

export function inverseMatrix(A: number[][]): number[][] {
    const n = A.length;
    const columns = identity(n).map(column => solveLinear(A, column));
    return transpose(columns);
}

export function pseudoInverseSymmetric(A: number[][], rcond = 1e-12): number[][] {
    const eig = symmetricEigDecomposition(A), n = A.length;
    const largest = Math.max(0, ...eig.values.map(Math.abs));
    const out = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let c = 0; c < eig.values.length; c++) if (Math.abs(eig.values[c]) > rcond * largest) {
        const scale = 1 / eig.values[c], vector = eig.vectors[c];
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i][j] += scale * vector[i] * vector[j];
    }
    return out;
}

export function logDetSymmetricPositive(A: number[][]): number {
    const values = symmetricEigDecomposition(A).values;
    if (values.some(value => value <= 0)) return -Infinity;
    return values.reduce((sum, value) => sum + Math.log(value), 0);
}

export function validateMatrix(X: number[][], minimumRows = 1): number {
    if (X.length < minimumRows || X[0]?.length === 0 || X.some(row => row.length !== X[0].length || row.some(value => !Number.isFinite(value)))) {
        throw new Error('X must be a non-empty finite rectangular matrix');
    }
    return X[0].length;
}

export function validateRegressionData(X: number[][], y: number[]): number {
    const p = validateMatrix(X);
    if (y.length !== X.length || y.some(value => !Number.isFinite(value))) throw new Error('X and y must have the same length and finite values');
    return p;
}

export function normalRandom(random: () => number): number {
    const u = Math.max(Number.MIN_VALUE, random());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

export function logGamma(z: number): number {
    const coefficients = [676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406, 12.507343278686905, -0.13857109526572012, 9.984369578019571e-6, 1.5056327351493116e-7];
    if (z < .5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
    z -= 1; let x = .9999999999998099;
    for (let i = 0; i < coefficients.length; i++) x += coefficients[i] / (z + i + 1);
    const t = z + coefficients.length - .5;
    return .5 * Math.log(2 * Math.PI) + (z + .5) * Math.log(t) - t + Math.log(x);
}

export function unitBallVolume(dimension: number): number {
    return Math.exp(dimension / 2 * Math.log(Math.PI) - logGamma(dimension / 2 + 1));
}
