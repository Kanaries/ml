import { CSRMatrix } from '../../data';
import { BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB } from '..';

const X = [
    [2, 0, 1, 0],
    [1, 0, 2, 0],
    [0, 2, 0, 1],
    [0, 1, 0, 2],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
];
const y = [0, 0, 1, 1, 0, 1];
const sparse = CSRMatrix.fromDense(X);

describe('Naive Bayes sparse input parity', () => {
    it.each([
        ['MultinomialNB', () => new MultinomialNB()],
        ['ComplementNB', () => new ComplementNB()],
        ['BernoulliNB', () => new BernoulliNB()],
        ['CategoricalNB', () => new CategoricalNB()],
        ['GaussianNB', () => new GaussianNB()],
    ] as const)('%s matches dense fit and prediction', (_name, create) => {
        const denseModel = create();
        const sparseModel = create();
        denseModel.fit(X, y);
        sparseModel.fit(sparse, y);
        expect(sparseModel.predict(sparse)).toEqual(denseModel.predict(X));
        expect(sparseModel.predict(CSRMatrix.fromDense([[0, 2, 0, 0]])))
            .toEqual(denseModel.predict([[0, 2, 0, 0]]));
    });

    it('handles BernoulliNB thresholds where implicit zeros become present', () => {
        const denseModel = new BernoulliNB({ binarize: -0.5 });
        const sparseModel = new BernoulliNB({ binarize: -0.5 });
        denseModel.fit(X, y);
        sparseModel.fit(sparse, y);
        expect(sparseModel.predict(sparse)).toEqual(denseModel.predict(X));
    });

    it('retains raw feature counts when BernoulliNB binarize is null', () => {
        const raw = [[0, 2, 0], [1, 0.5, 0], [0, 0, 3], [1, 0, 1]];
        const labels = [0, 0, 1, 1];
        const denseModel = new BernoulliNB({ binarize: null });
        const sparseModel = new BernoulliNB({ binarize: null });
        denseModel.fit(raw, labels);
        sparseModel.fit(CSRMatrix.fromDense(raw), labels);
        const denseState = denseModel.toJSON().state as Record<string, unknown>;
        const sparseState = sparseModel.toJSON().state as Record<string, unknown>;
        expect(denseState.featureCount).toEqual([[1, 2.5, 0], [1, 0, 4]]);
        expect(sparseState.featureCount).toEqual(denseState.featureCount);
    });

    it('matches sklearn GaussianNB for large offsets with small variance', () => {
        const train = [
            [1e12, 0], [1e12 + 1, 1], [1e12 + 2, 0],
            [1e12 + 10, 1], [1e12 + 11, 0], [1e12 + 12, 1],
        ];
        const labels = [0, 0, 0, 1, 1, 1];
        const test = [[1e12 + 1.5, 0], [1e12 + 10.5, 1]];
        const expected = [[1, 2.2877195122349133e-30], [2.2877195122349133e-30, 1]];
        for (const matrix of [train, CSRMatrix.fromDense(train)]) {
            const model = new GaussianNB();
            model.fit(matrix, labels);
            const actual = model.predictProba(CSRMatrix.fromDense(test));
            for (let i = 0; i < expected.length; i++) {
                expect(actual[i][0]).toBeCloseTo(expected[i][0], 12);
                expect(actual[i][1]).toBeCloseTo(expected[i][1], 12);
            }
        }
    });
});
