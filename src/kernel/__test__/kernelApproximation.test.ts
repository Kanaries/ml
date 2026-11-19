import fs from 'fs';
import path from 'path';
import { CSRMatrix, FeatureExtraction, Kernel, Linear, Pipeline, loadModel } from '../../index';

const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/kernel_approximation.json'), 'utf8'));

function expectCloseMatrix(actual: number[][], expected: number[][], precision = 9): void {
    expect(actual).toHaveLength(expected.length);
    expected.forEach((row, i) => {
        expect(actual[i]).toHaveLength(row.length);
        row.forEach((value, j) => expect(actual[i][j]).toBeCloseTo(value, precision));
    });
}

describe('RBFSampler', () => {
    test.each(fixture.rbfSampler)('reproduces sklearn Fourier features for gamma=$gamma', reference => {
        const options = { gamma: reference.gamma, nComponents: 5, randomState: fixture.seed };
        const dense = new Kernel.RBFSampler(options);
        dense.fit(fixture.X);
        expectCloseMatrix(dense.randomWeights, reference.weights);
        expectCloseMatrix([dense.randomOffset], [reference.offset]);
        expectCloseMatrix(dense.transform(fixture.query), reference.projection);

        const sparse = new Kernel.RBFSampler(options);
        const train = CSRMatrix.fromDense(fixture.X);
        const query = CSRMatrix.fromDense(fixture.query);
        const forbidden = () => { throw new Error('sparse samples must stay sparse'); };
        for (const batch of [train, query]) {
            jest.spyOn(batch, 'toDense').mockImplementation(forbidden);
            jest.spyOn(batch, 'row').mockImplementation(forbidden);
        }
        sparse.fit(train);
        expectCloseMatrix(sparse.transform(query), reference.projection);
    });

    test.each([1e-200, 1e200])('scale features are invariant under a change of units by %p', factor => {
        const model = new Kernel.RBFSampler({ gamma: 'scale', nComponents: 5, randomState: fixture.seed });
        const rescale = (X: number[][]) => X.map(row => row.map(value => value * factor));
        model.fit(fixture.X);
        const expected = model.transform(fixture.query);
        model.fit(CSRMatrix.fromDense(rescale(fixture.X)));
        expectCloseMatrix(model.transform(rescale(fixture.query)), expected);
    });

    test('scale falls back to unit gamma for a constant matrix', () => {
        const constant = [[3, 3], [3, 3]];
        const auto = new Kernel.RBFSampler({ gamma: 'scale', randomState: 8 });
        const unit = new Kernel.RBFSampler({ gamma: 1, randomState: 8 });
        expect(auto.fitTransform(constant)).toEqual(unit.fitTransform(constant));
    });

    test('a sparse text pipeline predicts after serializing the sampled features', () => {
        const documents = ['apple pear', 'apple plum', 'ocean tide', 'ocean wave'];
        const labels = [0, 0, 1, 1];
        const pipeline = new Pipeline({
            steps: [
                ['text', new FeatureExtraction.TfidfVectorizer()],
                ['map', new Kernel.RBFSampler({ nComponents: 64, randomState: 7 })],
                ['classifier', new Linear.LogisticRegression()],
            ],
        });
        pipeline.fit(documents, labels);
        expect(pipeline.predict(documents)).toEqual(labels);
        const revived = loadModel(JSON.stringify(pipeline)) as Pipeline;
        expect(revived.predict(documents)).toEqual(labels);
    });
});

describe('Nystroem', () => {
    test.each(fixture.nystroem)('matches sklearn landmark selection and held-out $kernel features', reference => {
        const model = new Kernel.Nystroem({
            kernel: reference.kernel, ...reference.params, nComponents: 3, randomState: fixture.seed,
        });
        model.fit(fixture.X);
        expect(model.componentIndices).toEqual(reference.indices);
        expect(model.components).toEqual(reference.indices.map((i: number) => fixture.X[i]));
        expectCloseMatrix(model.transform(fixture.query), reference.projection);
    });

    test.each([1e-200, 1e200])('cosine projections retain directional similarity at scale %p', factor => {
        const model = new Kernel.Nystroem({ kernel: 'cosine', nComponents: 3, randomState: fixture.seed });
        const rescale = (X: number[][]) => X.map(row => row.map(value => value * factor));
        model.fit(fixture.X);
        const expected = model.transform(fixture.query);
        model.fit(rescale(fixture.X));
        expectCloseMatrix(model.transform(rescale(fixture.query)), expected);
    });

    test('caps basis size and handles duplicate landmarks in a singular kernel', () => {
        const X = [[1, 0], [1, 0], [0, 1]];
        const model = new Kernel.Nystroem({ kernel: 'linear', nComponents: 12, randomState: 19 });
        const features = model.fitTransform(X);
        expect(features[0]).toHaveLength(X.length);
        const gram = features.map(a => features.map(b => a.reduce((sum, value, i) => sum + value * b[i], 0)));
        expectCloseMatrix(gram, [[1, 1, 0], [1, 1, 0], [0, 0, 1]], 7);
    });
});

describe.each([
    ['RBFSampler', Kernel.RBFSampler],
    ['Nystroem', Kernel.Nystroem],
] as const)('%s validation', (_name, Transformer) => {
    test('rejects invalid component counts, unfitted transform and mismatched feature widths', () => {
        expect(() => new Transformer({ nComponents: 0 })).toThrow();
        expect(() => new Transformer({ nComponents: 1.2 })).toThrow();
        const model = new Transformer({ nComponents: 2, randomState: 1 });
        expect(() => model.transform([[1, 2]])).toThrow();
        expect(() => model.fit([[1, 2], [3]])).toThrow();
        expect(() => model.fit([[1, 2], [3, Infinity]])).toThrow();
        model.fit([[1, 0], [0, 1]]);
        expect(() => model.transform([[1]])).toThrow();
    });
});
