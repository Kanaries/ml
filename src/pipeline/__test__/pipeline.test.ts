import { BaseEstimator, Params, loadModel, registerEstimator } from '../../base/estimator';
import { TransformerBase } from '../../base/transformer';
import { Pipeline, makePipeline } from '../pipeline';
import { FeatureUnion } from '../featureUnion';
import { ColumnTransformer } from '../columnTransformer';
import { LogisticRegression } from '../../linear/logisticRegression';
import { binaryDataset } from '../../__test__/conformance/datasets';

/** Minimal deterministic transformer: scales features by a constant. */
class ScaleBy extends TransformerBase {
    private factor: number;
    private fitted: boolean;
    constructor(props: { factor?: number } = {}) {
        super();
        this.factor = props.factor ?? 2;
        this.fitted = false;
    }
    public getParams(): Params {
        return { factor: this.factor };
    }
    public fit(): void {
        this.fitted = true;
    }
    public transform(X: number[][]): number[][] {
        if (!this.fitted) throw new Error('not fitted');
        return X.map((row) => row.map((v) => v * this.factor));
    }
}
registerEstimator('test.ScaleBy', ScaleBy);

describe('Pipeline', () => {
    const { X, y } = binaryDataset();

    it('validates steps', () => {
        expect(() => new Pipeline({ steps: [] })).toThrow(/non-empty/);
        expect(() => new Pipeline({ steps: [['a', new ScaleBy()], ['a', new ScaleBy()]] })).toThrow(/Duplicate/);
        expect(() => new Pipeline({ steps: [['a__b', new ScaleBy()]] })).toThrow(/__/);
        expect(() => new Pipeline({
            steps: [['clf', new LogisticRegression()], ['scale', new ScaleBy()]],
        })).toThrow(/Intermediate step "clf"/);
    });

    it('fits transformers in sequence and predicts through them', () => {
        const pipe = new Pipeline({
            steps: [
                ['scale', new ScaleBy({ factor: 0.5 })],
                ['clf', new LogisticRegression({ learningRate: 0.2, maxIter: 300 })],
            ],
        });
        pipe.fit(X, y);
        const preds = pipe.predict(X);
        expect(preds).toHaveLength(X.length);
        // must equal manually chaining the same transform
        const manual = new LogisticRegression({ learningRate: 0.2, maxIter: 300 });
        const Xs = X.map((row) => row.map((v) => v * 0.5));
        manual.fit(Xs, y);
        expect(preds).toEqual(manual.predict(Xs));
        expect(pipe.score(X, y)).toBeGreaterThan(0.9);
    });

    it('supports step__param addressing in setParams', () => {
        const pipe = new Pipeline({
            steps: [['scale', new ScaleBy()], ['clf', new LogisticRegression()]],
        });
        pipe.setParams({ scale__factor: 3, clf__maxIter: 50 });
        expect(pipe.getStep('scale').getParams()).toEqual({ factor: 3 });
        expect(pipe.getStep('clf').getParams()).toMatchObject({ maxIter: 50 });
        expect(() => pipe.setParams({ missing__x: 1 })).toThrow(/Unknown step/);
        expect(() => pipe.setParams({ clf__nope: 1 })).toThrow(/Invalid parameter/);
    });

    it('clone deep-clones the step estimators', () => {
        const pipe = new Pipeline({
            steps: [['scale', new ScaleBy()], ['clf', new LogisticRegression()]],
        });
        const copy = pipe.clone();
        expect(copy.getStep('clf')).not.toBe(pipe.getStep('clf'));
        expect(copy.getStep('clf').getParams()).toEqual(pipe.getStep('clf').getParams());
    });

    it('serializes and revives a fitted pipeline with identical predictions', () => {
        const pipe = new Pipeline({
            steps: [
                ['scale', new ScaleBy({ factor: 0.5 })],
                ['clf', new LogisticRegression({ learningRate: 0.2, maxIter: 300 })],
            ],
        });
        pipe.fit(X, y);
        const revived = loadModel(JSON.stringify(pipe)) as Pipeline;
        expect(revived).toBeInstanceOf(Pipeline);
        expect(revived.predict(X)).toEqual(pipe.predict(X));
    });

    it('makePipeline derives unique step names', () => {
        const pipe = makePipeline(new ScaleBy(), new ScaleBy(), new LogisticRegression());
        const names = Object.keys(pipe.namedSteps);
        expect(names).toEqual(['scaleby-1', 'scaleby-2', 'logisticregression']);
    });

    it('transform/fitTransform work for all-transformer pipelines', () => {
        const pipe = makePipeline(new ScaleBy({ factor: 2 }), new ScaleBy({ factor: 5 }));
        const out = pipe.fitTransform([[1, 2]]);
        expect(out).toEqual([[10, 20]]);
        expect(pipe.transform([[2, 0]])).toEqual([[20, 0]]);
    });
});

describe('FeatureUnion', () => {
    it('concatenates transformer outputs column-wise', () => {
        const union = new FeatureUnion({
            transformerList: [
                ['double', new ScaleBy({ factor: 2 })],
                ['halve', new ScaleBy({ factor: 0.5 })],
            ],
        });
        const out = union.fitTransform([[2, 4], [6, 8]]);
        expect(out).toEqual([[4, 8, 1, 2], [12, 16, 3, 4]]);
    });

    it('supports nested param addressing and serialization', () => {
        const union = new FeatureUnion({ transformerList: [['a', new ScaleBy()]] });
        union.setParams({ a__factor: 7 });
        expect(union.getTransformer('a').getParams()).toEqual({ factor: 7 });
        union.fit([[1]]);
        const revived = loadModel(JSON.stringify(union)) as FeatureUnion;
        expect(revived.transform([[1]])).toEqual([[7]]);
    });
});

describe('ColumnTransformer', () => {
    const X = [
        [1, 10, 100],
        [2, 20, 200],
    ];

    it('applies transformers to column subsets in declaration order', () => {
        const ct = new ColumnTransformer({
            transformers: [
                ['double', new ScaleBy({ factor: 2 }), [0]],
                ['keep', 'passthrough', [2]],
            ],
        });
        expect(ct.fitTransform(X)).toEqual([[2, 100], [4, 200]]);
    });

    it('handles remainder=passthrough and drop entries', () => {
        const ct = new ColumnTransformer({
            transformers: [
                ['gone', 'drop', [2]],
                ['double', new ScaleBy({ factor: 2 }), [1]],
            ],
            remainder: 'passthrough',
        });
        expect(ct.fitTransform(X)).toEqual([[20, 1], [40, 2]]);
    });

    it('serializes and revives', () => {
        const ct = new ColumnTransformer({
            transformers: [['double', new ScaleBy({ factor: 2 }), [0, 1]]],
            remainder: 'passthrough',
        });
        ct.fit(X);
        const revived = loadModel(JSON.stringify(ct)) as ColumnTransformer;
        expect(revived.transform(X)).toEqual(ct.transform(X));
    });

    it('validates entries', () => {
        expect(() => new ColumnTransformer({ transformers: [] })).toThrow(/non-empty/);
        expect(() => new ColumnTransformer({
            transformers: [['a', new ScaleBy(), [-1]]],
        })).toThrow(/integer/);
        const dummy = new ColumnTransformer({ transformers: [['a', 'passthrough', [99]]] });
        expect(() => dummy.fitTransform(X)).toThrow(/out of range/);
    });
});

describe('Pipeline conformance basics', () => {
    it('params round-trip and unknown-key validation', () => {
        const pipe = new Pipeline({ steps: [['clf', new LogisticRegression()]] });
        const params = pipe.getParams();
        pipe.setParams(params);
        expect((pipe.getParams().steps as unknown[][])[0][0]).toBe('clf');
        expect(() => pipe.setParams({ bogus: 1 })).toThrow(/Invalid parameter/);
        expect((pipe.getParams().steps as unknown[][])[0][1]).toBeInstanceOf(BaseEstimator);
    });
});
