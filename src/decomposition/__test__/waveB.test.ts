import { loadModel } from '../../base';
import fs from 'fs';
import path from 'path';
import { FastICA } from '../fastICA';
import { IncrementalPCA } from '../incrementalPCA';
import { KernelPCA } from '../kernelPCA';
import { NMF } from '../nmf';
import { PCA } from '../pca';

const waveB = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_b.json'), 'utf8'));

function distance(a: number[], b: number[]): number {
    return Math.sqrt(a.reduce((sum, value, j) => sum + (value - b[j]) ** 2, 0));
}

test('linear KernelPCA preserves the PCA geometry and training transform parity', () => {
    const X = [[1, 2], [2, 4.1], [3, 5.9], [4, 8.2], [5, 9.8]];
    const kernel = new KernelPCA({ nComponents: 2, kernel: 'linear' });
    const embedded = kernel.fitTransform(X);
    const replay = kernel.transform(X);
    replay.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(embedded[i][j], 7)));
    const pca = new PCA({ nComponents: 2 });
    const reference = pca.fitTransform(X);
    for (let i = 0; i < X.length; i++) for (let j = 0; j < X.length; j++) {
        expect(distance(embedded[i], embedded[j])).toBeCloseTo(distance(reference[i], reference[j]), 6);
    }
});

test('KernelPCA matches the pinned sklearn linear-kernel fixture', () => {
    const { X, transform: expected, eigenvalues } = waveB.kernel_pca;
    const model = new KernelPCA({ nComponents: 2, kernel: 'linear', randomState: 0 });
    model.fitTransform(X).forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(expected[i][j], 10)));
    model.eigenvalues.forEach((value, i) => expect(value).toBeCloseTo(eigenvalues[i], 10));
    expect(model.eigenvectors).toHaveLength(X.length);
    expect(model.eigenvectors[0]).toHaveLength(2);
});

test('KernelPCA retains positive sigmoid eigenpairs and removes zero modes when nComponents is null', () => {
    const sigmoid = new KernelPCA({ nComponents: 3, kernel: 'sigmoid', gamma: 1, coef0: 1 });
    sigmoid.fit([[-2], [-1], [0], [1], [2]]);
    sigmoid.eigenvalues.forEach((value, i) => expect(value).toBeCloseTo(waveB.kernel_pca.sigmoid_eigenvalues[i], 10));

    const linear = new KernelPCA({ nComponents: null, kernel: 'linear' });
    linear.fit([[1, 2], [2, 4.1], [3, 5.9], [4, 8.2], [5, 9.8]]);
    expect(linear.eigenvalues).toHaveLength(2);
});

test('RBF KernelPCA supports out-of-sample and inverse transforms', () => {
    const X = Array.from({ length: 16 }, (_, i) => [Math.cos(i * Math.PI / 8), Math.sin(i * Math.PI / 8)]);
    const model = new KernelPCA({ nComponents: 4, kernel: 'rbf', gamma: 2, fitInverseTransform: true, alpha: .01 });
    const embedded = model.fitTransform(X);
    expect(model.transform([[1, 0]])[0]).toHaveLength(4);
    const reconstructed = model.inverseTransform(embedded);
    expect(reconstructed.flat().every(Number.isFinite)).toBe(true);
    const revived = loadModel(JSON.stringify(model)) as KernelPCA;
    expect(revived.transform([[1, 0]])).toEqual(model.transform([[1, 0]]));
});

test('FastICA unmixes full-rank signals and reconstructs the observations', () => {
    const sources = Array.from({ length: 200 }, (_, i) => {
        const t = i / 20;
        return [Math.sin(t), (i % 40 < 20 ? 1 : -1)];
    });
    const X = sources.map(([a, b]) => [a + .5 * b, .3 * a + b]);
    const model = new FastICA({ nComponents: 2, randomState: 0, maxIter: 1000, tol: 1e-6 });
    const transformed = model.fitTransform(X);
    const reconstructed = model.inverseTransform(transformed);
    reconstructed.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(X[i][j], 6)));
    const mean0 = transformed.reduce((sum, row) => sum + row[0], 0) / transformed.length;
    const mean1 = transformed.reduce((sum, row) => sum + row[1], 0) / transformed.length;
    const correlation = transformed.reduce((sum, row) => sum + (row[0] - mean0) * (row[1] - mean1), 0) / transformed.length;
    expect(Math.abs(correlation)).toBeLessThan(1e-6);
});

test('FastICA whitening modes match the pinned sklearn one-component fixture', () => {
    const X = waveB.fast_ica.X;
    const unit = new FastICA({ nComponents: 1, whiten: 'unit-variance', randomState: 0, tol: 1e-8 });
    unit.fitTransform(X).forEach((row, i) => expect(row[0]).toBeCloseTo(waveB.fast_ica.unit[i][0], 12));
    const arbitrary = new FastICA({ nComponents: 1, whiten: 'arbitrary-variance', randomState: 0, tol: 1e-8 });
    arbitrary.fitTransform(X).forEach((row, i) => expect(row[0]).toBeCloseTo(waveB.fast_ica.arbitrary[i][0], 12));
});

test('FastICA deflation extracts multiple sklearn-equivalent independent components', () => {
    const X = waveB.fast_ica.deflation_X;
    const expected = waveB.fast_ica.deflation;
    const model = new FastICA({ nComponents: 2, algorithm: 'deflation', whiten: 'unit-variance', randomState: 0, maxIter: 1000, tol: 1e-8 });
    const actual = model.fitTransform(X);
    const correlation = (left: number, right: number): number => {
        const leftMean = actual.reduce((sum, row) => sum + row[left], 0) / actual.length;
        const rightMean = expected.reduce((sum: number, row: number[]) => sum + row[right], 0) / expected.length;
        let covariance = 0, leftScale = 0, rightScale = 0;
        for (let i = 0; i < actual.length; i++) {
            const a = actual[i][left] - leftMean, b = expected[i][right] - rightMean;
            covariance += a * b; leftScale += a * a; rightScale += b * b;
        }
        return Math.abs(covariance / Math.sqrt(leftScale * rightScale));
    };
    const direct = [correlation(0, 0), correlation(1, 1)];
    const swapped = [correlation(0, 1), correlation(1, 0)];
    const matched = direct[0] + direct[1] >= swapped[0] + swapped[1] ? direct : swapped;
    matched.forEach(value => expect(value).toBeGreaterThan(.9));
    model.inverseTransform(actual).forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(X[i][j], 6)));
});

test('FastICA whiten=false does not center input and ignores nComponents', () => {
    const X = [[2, 3], [3, 4], [4, 3], [3, 2]];
    const model = new FastICA({ nComponents: 1, whiten: false, randomState: 0 });
    const transformed = model.fitTransform(X);
    expect(model.components).toHaveLength(2);
    expect(model.mean).toEqual([0, 0]);
    model.inverseTransform(transformed).forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(X[i][j], 10)));
});

test('NMF finds a non-negative low-rank reconstruction', () => {
    const W = [[1, 0], [.5, .5], [0, 1], [1, 1]];
    const H = [[1, 2, 0], [0, 1, 3]];
    const X = W.map(row => H[0].map((_, j) => row[0] * H[0][j] + row[1] * H[1][j]));
    const model = new NMF({ nComponents: 2, init: 'nndsvda', maxIter: 1000, tol: 1e-7, randomState: 0 });
    const transformed = model.fitTransform(X);
    expect(model.reconstructionErr).toBeLessThan(.05);
    expect(transformed.flat().every(value => value >= 0 && Number.isFinite(value))).toBe(true);
    expect(model.components.flat().every(value => value >= 0 && Number.isFinite(value))).toBe(true);
});

test('NMF multiplicative updates match the pinned sklearn deterministic NNDSVD fixture', () => {
    const { X, W: expectedW, H: expectedH, reconstruction_error: reconstructionError } = waveB.nmf;
    const model = new NMF({ nComponents: 2, init: 'nndsvda', maxIter: 1000, tol: 1e-7, randomState: 0 });
    model.fitTransform(X).forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(expectedW[i][j], 10)));
    model.components.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(expectedH[i][j], 10)));
    expect(model.reconstructionErr).toBeCloseTo(reconstructionError, 10);
});

test('NMF applies sklearn NNDSVD epsilon truncation before nndsvda filling', () => {
    const fixture = waveB.nmf.eps_truncation;
    const model = new NMF({ nComponents: 2, init: 'nndsvda', maxIter: 20, tol: 0, randomState: 0 });
    model.fitTransform(fixture.X).forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(fixture.W[i][j], 10)));
    model.components.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(fixture.H[i][j], 10)));
    expect(model.reconstructionErr).toBeCloseTo(fixture.reconstruction_error, 14);
});

test('IncrementalPCA chunking matches batch PCA covariance results', () => {
    const X = Array.from({ length: 50 }, (_, i) => [i / 10, Math.sin(i), (i % 5) - 2]);
    const incremental = new IncrementalPCA({ nComponents: 3 });
    incremental.partialFit(X.slice(0, 17)).partialFit(X.slice(17, 33)).partialFit(X.slice(33));
    const batch = new PCA({ nComponents: 3 });
    batch.fit(X);
    incremental.explainedVariance.forEach((value, i) => expect(value).toBeCloseTo(batch.getExplainedVariance()[i], 8));
    expect(incremental.nSamplesSeen).toBe(50);
    expect(incremental.transform(X)[0]).toHaveLength(3);
});

test('IncrementalPCA matches the pinned sklearn three-batch truncated-SVD fixture', () => {
    const { X, rows, partial } = waveB.incremental_pca;
    const model = new IncrementalPCA({ nComponents: 2 });
    model.partialFit(X.slice(0, 17)).partialFit(X.slice(17, 33)).partialFit(X.slice(33));
    model.mean.forEach((value, i) => expect(value).toBeCloseTo(partial.mean[i], 12));
    model.explainedVariance.forEach((value, i) => expect(value).toBeCloseTo(partial.explained_variance[i], 10));
    model.singularValues.forEach((value, i) => expect(value).toBeCloseTo(partial.singular_values[i], 10));
    model.transform(rows.map((row: number) => X[row])).forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(partial.transform[i][j], 10)));
});

test('IncrementalPCA fit uses sklearn default batchSize of five times nFeatures', () => {
    const { X, fit } = waveB.incremental_pca;
    const model = new IncrementalPCA({ nComponents: 2 });
    model.fit(X);
    model.explainedVariance.forEach((value, i) => expect(value).toBeCloseTo(fit.explained_variance[i], 10));
    model.singularValues.forEach((value, i) => expect(value).toBeCloseTo(fit.singular_values[i], 10));
});

test('IncrementalPCA merges an undersized final batch like sklearn gen_batches', () => {
    const fixture = waveB.incremental_pca.merged_remainder;
    const model = new IncrementalPCA({ nComponents: 2 });
    model.fit(fixture.X);
    model.explainedVariance.forEach((value, i) => expect(value).toBeCloseTo(fixture.explained_variance[i], 10));
    model.singularValues.forEach((value, i) => expect(value).toBeCloseTo(fixture.singular_values[i], 10));
});
