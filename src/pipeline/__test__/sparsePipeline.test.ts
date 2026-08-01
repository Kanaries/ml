import { Params, registerEstimator } from '../../base/estimator';
import { TransformerBase } from '../../base/transformer';
import { MultinomialNB } from '../../bayes';
import { CSRMatrix } from '../../data';
import type { FeatureData } from '../../data';
import { loadModel } from '../../base';
import { LogisticRegression } from '../../linear';
import { Pipeline } from '../pipeline';

class PresenceVectorizer extends TransformerBase<string[], CSRMatrix> {
    public readonly acceptedInputKinds = ['text'] as const;
    public getParams(): Params { return {}; }
    public fit(_documents: string[]): void {}
    public transform(documents: string[]): CSRMatrix {
        return CSRMatrix.fromDense(documents.map(document => [
            document.includes('red') ? 1 : 0,
            document.includes('blue') ? 1 : 0,
        ]));
    }
}
registerEstimator('TestPresenceVectorizer', PresenceVectorizer);

class DenseToSparse extends TransformerBase<number[][], CSRMatrix> {
    public getParams(): Params { return {}; }
    public fit(_X: number[][]): void {}
    public transform(X: number[][]): CSRMatrix { return CSRMatrix.fromDense(X); }
}

test('Pipeline carries raw strings into a CSR-aware final estimator', () => {
    const documents = ['red apple', 'red rose', 'blue sky', 'deep blue'];
    const labels = [0, 0, 1, 1];
    const pipeline = new Pipeline({
        steps: [
            ['vectorizer', new PresenceVectorizer()],
            ['classifier', new MultinomialNB()],
        ],
    });
    pipeline.fit(documents, labels);
    expect(pipeline.predict(['red flower', 'blue ocean'])).toEqual([0, 1]);

    const revived = loadModel(JSON.stringify(pipeline)) as Pipeline;
    expect(revived.predict(['red flower', 'blue ocean'])).toEqual([0, 1]);
});

test('Pipeline rejects raw strings that reach a numeric estimator', () => {
    const pipeline = new Pipeline({ steps: [['classifier', new MultinomialNB()]] });
    expect(() => pipeline.fit(['red', 'blue'], [0, 1])).toThrow(/Add a text transformer/);
});

test('Pipeline rejects raw strings at a dense-only intermediate transformer', () => {
    class DenseOnly extends TransformerBase {
        public getParams(): Params { return {}; }
        public fit(_X: number[][]): void {}
        public transform(X: number[][]): number[][] { return X; }
    }
    const pipeline = new Pipeline({
        steps: [['dense', new DenseOnly()], ['classifier', new MultinomialNB()]],
    });
    expect(() => pipeline.fit(['red', 'blue'], [0, 1])).toThrow(/step "dense" does not accept text/);
});

test('Pipeline rejects CSR at a dense-only final predictor boundary', () => {
    const pipeline = new Pipeline({
        steps: [['sparsify', new DenseToSparse()], ['lr', new LogisticRegression()]],
    });
    expect(() => pipeline.fit([[1, 0], [0, 1]], [0, 1])).toThrow(/step "lr" does not accept csr/);
});

test('Pipeline infers transform output from its final transformer', () => {
    const pipeline = new Pipeline({ steps: [['sparsify', new DenseToSparse()]] });
    const sparse: CSRMatrix = pipeline.fitTransform([[1, 0], [0, 1]]);
    expect(sparse.shape).toEqual([2, 2]);

    // Dense input does not imply dense output; the final transformer does.
    const denseToSparse: CSRMatrix = pipeline.transform([[0, 2]]);
    expect(denseToSparse.toDense()).toEqual([[0, 2]]);

    const generic: Pipeline = pipeline;
    const genericOutput: FeatureData = generic.transform([[3, 0]]);
    expect(genericOutput).toBeInstanceOf(CSRMatrix);
});

test('Pipeline.fit accepts a final text transformer without forcing numeric input', () => {
    const pipeline = new Pipeline({ steps: [['vectorizer', new PresenceVectorizer()]] });
    expect(() => pipeline.fit(['red', 'blue'])).not.toThrow();
    expect(pipeline.transform(['red']).toDense()).toEqual([[1, 0]]);
});
