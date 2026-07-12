import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { getRegisteredEstimators, loadModel } from '../../base/estimator';
import { LogisticRegression } from '../../linear/logisticRegression';
import {
    Binarizer,
    CategoricalValue,
    FunctionTransformer,
    KBinsDiscretizer,
    KNNImputer,
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    MissingIndicator,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SelectKBest,
    SimpleImputer,
    StandardScaler,
    VarianceThreshold,
} from '../preprocessing';
import { GridSearchCV, KFold, RandomizedSearchCV } from '../modelSelection';

// ---------------------------------------------------------------------------
// Numeric X-matrix transformers: full harness on the blobs dataset.
// ---------------------------------------------------------------------------
runEstimatorConformance([
    {
        name: 'StandardScaler',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new StandardScaler({ withMean: true, withStd: true }),
    },
    {
        name: 'MinMaxScaler',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new MinMaxScaler({ featureRange: [0, 1] }),
    },
    {
        name: 'MaxAbsScaler',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new MaxAbsScaler(),
    },
    {
        name: 'Normalizer',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new Normalizer({ norm: 'l2' }),
    },
    {
        name: 'Binarizer',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new Binarizer({ threshold: 1 }),
    },
    {
        name: 'SimpleImputer',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new SimpleImputer({ strategy: 'mean' }),
    },
    {
        name: 'VarianceThreshold',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new VarianceThreshold({ threshold: 0.01 }),
    },
    {
        name: 'SelectKBest',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new SelectKBest({ k: 1, scoreFunc: 'fRegression' }),
    },
]);

// ---------------------------------------------------------------------------
// preprocessingExtra transformers: full harness on the blobs dataset.
// KNNImputer and MissingIndicator see no NaN in blobs, so they act as
// passthrough / all-zeros there — acceptable harness behavior (their real
// semantics are covered in preprocessingExtra.test.ts).
// ---------------------------------------------------------------------------
runEstimatorConformance([
    {
        name: 'RobustScaler',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new RobustScaler({ quantileRange: [25, 75], unitVariance: false }),
    },
    {
        name: 'PowerTransformer',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new PowerTransformer({ method: 'yeo-johnson', standardize: true }),
    },
    {
        name: 'QuantileTransformer',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new QuantileTransformer({ nQuantiles: 20, outputDistribution: 'uniform', randomState: 42 }),
    },
    {
        name: 'PolynomialFeatures',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new PolynomialFeatures({ degree: 2, interactionOnly: false, includeBias: true }),
    },
    {
        name: 'KBinsDiscretizer',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new KBinsDiscretizer({ nBins: 3, encode: 'ordinal', strategy: 'quantile' }),
    },
    {
        name: 'KNNImputer',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new KNNImputer({ nNeighbors: 3, weights: 'uniform' }),
    },
    {
        name: 'MissingIndicator',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new MissingIndicator({ features: 'all' }),
    },
    {
        name: 'FunctionTransformer',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new FunctionTransformer({ func: 'expm1', inverseFunc: 'log1p' }),
    },
]);

// ---------------------------------------------------------------------------
// LabelBinarizer operates on 1-D label arrays: hand-rolled conformance.
// ---------------------------------------------------------------------------
describe('LabelBinarizer conformance (hand-rolled)', () => {
    const y = [1, 2, 6, 4, 2];

    it('is registered under its declared name', () => {
        expect(getRegisteredEstimators().get('LabelBinarizer')).toBe(LabelBinarizer);
    });

    it('getParams/setParams round-trips and validates keys', () => {
        const lb = new LabelBinarizer({ negLabel: -1, posLabel: 1 });
        const params = lb.getParams();
        expect(params).toEqual({ negLabel: -1, posLabel: 1 });
        lb.setParams(params);
        expect(lb.getParams()).toEqual(params);
        expect(() => lb.setParams({ __definitely_not_a_param__: 1 })).toThrow(/Invalid parameter/);
    });

    it('clone copies params onto a fresh instance', () => {
        const lb = new LabelBinarizer({ negLabel: -1, posLabel: 2 });
        const copy = lb.clone();
        expect(copy).not.toBe(lb);
        expect(copy.constructor).toBe(LabelBinarizer);
        expect(copy.getParams()).toEqual(lb.getParams());
    });

    it('survives serialize → JSON text → revive with identical behavior', () => {
        const lb = new LabelBinarizer();
        const out = lb.fitTransform(y);
        const revived = loadModel(JSON.stringify(lb)) as LabelBinarizer;
        expect(revived.constructor).toBe(LabelBinarizer);
        expect(revived.getParams()).toEqual(lb.getParams());
        expect(revived.transform(y)).toEqual(out);
        expect(revived.inverseTransform(out)).toEqual(y);
        expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(lb)));
    });

    it('setParams after fit resets to a working unfitted estimator', () => {
        const lb = new LabelBinarizer();
        lb.fit(y);
        lb.setParams({});
        expect(() => lb.transform(y)).toThrow(/must be fitted/);
        expect(lb.fitTransform([0, 1])).toEqual([[0], [1]]);
    });
});

// ---------------------------------------------------------------------------
// FunctionTransformer function-valued param handling (mirrors SelectKBest).
// ---------------------------------------------------------------------------
describe('FunctionTransformer func param', () => {
    it('serializes when configured with built-in names', () => {
        const ft = new FunctionTransformer({ func: 'log1p', inverseFunc: 'expm1' });
        expect(ft.getParams()).toEqual({ func: 'log1p', inverseFunc: 'expm1' });
        expect(() => JSON.stringify(ft)).not.toThrow();
    });

    it('toJSON throws for a raw function param (documented codec behavior)', () => {
        const ft = new FunctionTransformer({ func: (v: number) => v * 2 });
        expect(() => JSON.stringify(ft)).toThrow(/Cannot serialize a value of type "function"/);
    });

    it('throws at transform time for an unknown built-in name', () => {
        const ft = new FunctionTransformer({ func: 'notAFunc' });
        expect(() => ft.transform([[1]])).toThrow(/Unknown function/);
    });
});

// ---------------------------------------------------------------------------
// Categorical-matrix transformers: the harness datasets are continuous
// floats, so OrdinalEncoder / OneHotEncoder get hand-rolled conformance
// tests on categorical data covering the same points.
// ---------------------------------------------------------------------------
const CATEGORICAL_X: CategoricalValue[][] = [
    ['a', 0, true],
    ['b', 1, false],
    ['a', 2, true],
    ['c', 1, false],
    ['b', 0, true],
];

describe('OrdinalEncoder conformance (hand-rolled)', () => {
    it('is registered under its declared name', () => {
        expect(getRegisteredEstimators().get('OrdinalEncoder')).toBe(OrdinalEncoder);
    });

    it('getParams/setParams round-trips and validates keys', () => {
        const enc = new OrdinalEncoder();
        expect(enc.getParams()).toEqual({});
        enc.setParams({});
        expect(() => enc.setParams({ __definitely_not_a_param__: 1 })).toThrow(/Invalid parameter/);
    });

    it('clone copies params onto a fresh instance', () => {
        const enc = new OrdinalEncoder();
        const copy = enc.clone();
        expect(copy).not.toBe(enc);
        expect(copy.constructor).toBe(OrdinalEncoder);
        expect(copy.getParams()).toEqual(enc.getParams());
    });

    it('survives serialize → JSON text → revive with identical behavior', () => {
        const enc = new OrdinalEncoder();
        const out = enc.fitTransform(CATEGORICAL_X);
        const revived = loadModel(JSON.stringify(enc)) as OrdinalEncoder;
        expect(revived.constructor).toBe(OrdinalEncoder);
        expect(revived.transform(CATEGORICAL_X)).toEqual(out);
        expect(revived.inverseTransform(out)).toEqual(CATEGORICAL_X);
        expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(enc)));
    });
});

describe('OneHotEncoder conformance (hand-rolled)', () => {
    it('is registered under its declared name', () => {
        expect(getRegisteredEstimators().get('OneHotEncoder')).toBe(OneHotEncoder);
    });

    it('getParams/setParams round-trips and validates keys', () => {
        const enc = new OneHotEncoder({ drop: 'first' });
        const params = enc.getParams();
        expect(params).toEqual({ drop: 'first' });
        enc.setParams(params);
        expect(enc.getParams()).toEqual(params);
        expect(() => enc.setParams({ __definitely_not_a_param__: 1 })).toThrow(/Invalid parameter/);
    });

    it('clone copies params onto a fresh instance', () => {
        const enc = new OneHotEncoder({ drop: 'ifBinary' });
        const copy = enc.clone();
        expect(copy).not.toBe(enc);
        expect(copy.constructor).toBe(OneHotEncoder);
        expect(copy.getParams()).toEqual(enc.getParams());
    });

    it('survives serialize → JSON text → revive with identical behavior', () => {
        const enc = new OneHotEncoder({ drop: 'first' });
        const out = enc.fitTransform(CATEGORICAL_X);
        const revived = loadModel(JSON.stringify(enc)) as OneHotEncoder;
        expect(revived.constructor).toBe(OneHotEncoder);
        expect(revived.getParams()).toEqual(enc.getParams());
        expect(revived.transform(CATEGORICAL_X)).toEqual(out);
        expect(revived.inverseTransform(out)).toEqual(CATEGORICAL_X);
        expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(enc)));
    });
});

// ---------------------------------------------------------------------------
// LabelEncoder operates on 1-D label arrays: hand-rolled conformance.
// ---------------------------------------------------------------------------
describe('LabelEncoder conformance (hand-rolled)', () => {
    const y = [30, 10, 30, 20, 10, 20];

    it('is registered under its declared name', () => {
        expect(getRegisteredEstimators().get('LabelEncoder')).toBe(LabelEncoder);
    });

    it('getParams/setParams round-trips and validates keys', () => {
        const enc = new LabelEncoder();
        expect(enc.getParams()).toEqual({});
        enc.setParams({});
        expect(() => enc.setParams({ __definitely_not_a_param__: 1 })).toThrow(/Invalid parameter/);
    });

    it('clone copies params onto a fresh instance', () => {
        const enc = new LabelEncoder();
        const copy = enc.clone();
        expect(copy).not.toBe(enc);
        expect(copy.constructor).toBe(LabelEncoder);
    });

    it('survives serialize → JSON text → revive with identical behavior', () => {
        const enc = new LabelEncoder();
        const out = enc.fitTransform(y);
        const revived = loadModel(JSON.stringify(enc)) as LabelEncoder;
        expect(revived.constructor).toBe(LabelEncoder);
        expect(revived.transform(y)).toEqual(out);
        expect(revived.inverseTransform(out)).toEqual(y);
        expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(enc)));
    });

    it('setParams after fit resets to a working unfitted estimator', () => {
        const enc = new LabelEncoder();
        enc.fit(y);
        enc.setParams({});
        expect(() => enc.transform(y)).toThrow(/must be fitted/);
        expect(enc.fitTransform(y)).toEqual([2, 0, 2, 1, 0, 1]);
    });
});

// ---------------------------------------------------------------------------
// SelectKBest score-function param handling.
// ---------------------------------------------------------------------------
describe('SelectKBest scoreFunc param', () => {
    it('serializes when configured with a built-in name', () => {
        const sel = new SelectKBest({ k: 1, scoreFunc: 'fRegression' });
        expect(sel.getParams()).toEqual({ k: 1, scoreFunc: 'fRegression' });
        expect(() => JSON.stringify(sel)).not.toThrow();
    });

    it('toJSON throws for a raw function param (documented codec behavior)', () => {
        const sel = new SelectKBest({ k: 1, scoreFunc: (X, _y) => X[0].map(() => 1) });
        expect(() => JSON.stringify(sel)).toThrow(/Cannot serialize a value of type "function"/);
    });

    it('throws at fit time for an unknown built-in name', () => {
        const sel = new SelectKBest({ k: 1, scoreFunc: 'notAScoreFunc' });
        expect(() => sel.fit([[1, 2], [3, 4]], [0, 1])).toThrow(/Unknown score function/);
    });
});

// ---------------------------------------------------------------------------
// Search meta-estimators: hand-rolled conformance.
// ---------------------------------------------------------------------------
const SEARCH_X = [[0], [1], [2], [10], [11], [12]];
const SEARCH_Y = [0, 0, 0, 1, 1, 1];

describe('GridSearchCV conformance (hand-rolled)', () => {
    const create = () => new GridSearchCV({
        estimator: new LogisticRegression({ maxIter: 100 }),
        paramGrid: { learningRate: [0.1, 0.5] },
        cv: new KFold({ nSplits: 3, shuffle: true, randomState: 42 }),
        scoring: 'accuracyScore',
    });

    it('is registered under its declared name', () => {
        expect(getRegisteredEstimators().get('GridSearchCV')).toBe(GridSearchCV);
    });

    it('requires exactly one of estimator/estimatorFactory', () => {
        expect(() => new GridSearchCV({ paramGrid: { a: [1] } })).toThrow(/exactly one/);
        expect(() => new GridSearchCV({
            estimator: new LogisticRegression(),
            estimatorFactory: p => new LogisticRegression(p),
            paramGrid: { a: [1] },
        })).toThrow(/exactly one/);
    });

    it('getParams/setParams round-trips and validates keys', () => {
        const search = create();
        const params = search.getParams();
        search.setParams(params);
        expect(search.getParams()).toEqual(params);
        expect(() => search.setParams({ __definitely_not_a_param__: 1 })).toThrow(/Invalid parameter/);
    });

    it('clone copies params onto a fresh instance', () => {
        const search = create();
        const copy = search.clone();
        expect(copy).not.toBe(search);
        expect(copy.constructor).toBe(GridSearchCV);
        expect(copy.getParams()).toEqual(search.getParams());
    });

    it('fits from an estimator prototype and refitting a clone reproduces the result', () => {
        const search = create();
        search.fit(SEARCH_X, SEARCH_Y);
        expect(search.bestParams).not.toBeNull();
        const pred = search.predict([[0.2], [11.5]]);
        expect(pred).toEqual([0, 1]);

        const again = search.clone();
        again.fit(SEARCH_X, SEARCH_Y);
        expect(again.bestParams).toEqual(search.bestParams);
        expect(again.bestScore).toBe(search.bestScore);
    });

    it('survives serialize → JSON text → revive with identical state and predictions', () => {
        const search = create();
        search.fit(SEARCH_X, SEARCH_Y);
        const pred = search.predict(SEARCH_X);
        const revived = loadModel(JSON.stringify(search)) as GridSearchCV;
        expect(revived.constructor).toBe(GridSearchCV);
        expect(revived.getParams()).toEqual(search.getParams());
        expect(revived.bestParams).toEqual(search.bestParams);
        expect(revived.predict(SEARCH_X)).toEqual(pred);
        expect(revived.score(SEARCH_X, SEARCH_Y)).toBe(search.score(SEARCH_X, SEARCH_Y));
        expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(search)));
    });

    it('toJSON throws for a factory-based search (functions are not serializable)', () => {
        const legacy = new GridSearchCV({
            estimatorFactory: params => new LogisticRegression(params),
            paramGrid: { learningRate: [0.1] },
        });
        expect(() => JSON.stringify(legacy)).toThrow(/Cannot serialize a value of type "function"/);
    });

    it('throws at fit time for an unknown scoring name', () => {
        const search = new GridSearchCV({
            estimator: new LogisticRegression({ maxIter: 10 }),
            paramGrid: { learningRate: [0.1] },
            scoring: 'notAScorer',
        });
        expect(() => search.fit(SEARCH_X, SEARCH_Y)).toThrow(/Unknown scoring/);
    });
});

describe('RandomizedSearchCV conformance (hand-rolled)', () => {
    const create = () => new RandomizedSearchCV({
        estimator: new LogisticRegression({ maxIter: 100 }),
        paramDistributions: { learningRate: [0.05, 0.1, 0.5] },
        nIter: 2,
        cv: new KFold({ nSplits: 3, shuffle: true, randomState: 42 }),
        scoring: 'accuracyScore',
        randomState: 9,
    });

    it('is registered under its declared name', () => {
        expect(getRegisteredEstimators().get('RandomizedSearchCV')).toBe(RandomizedSearchCV);
    });

    it('requires exactly one of estimator/estimatorFactory', () => {
        expect(() => new RandomizedSearchCV({ paramDistributions: { a: [1] }, nIter: 1 })).toThrow(/exactly one/);
    });

    it('getParams/setParams round-trips and validates keys', () => {
        const search = create();
        const params = search.getParams();
        search.setParams(params);
        expect(search.getParams()).toEqual(params);
        expect(() => search.setParams({ __definitely_not_a_param__: 1 })).toThrow(/Invalid parameter/);
    });

    it('clone copies params onto a fresh instance', () => {
        const search = create();
        const copy = search.clone();
        expect(copy).not.toBe(search);
        expect(copy.constructor).toBe(RandomizedSearchCV);
        expect(copy.getParams()).toEqual(search.getParams());
    });

    it('fits from an estimator prototype; refitting a clone is reproducible (seeded)', () => {
        const search = create();
        search.fit(SEARCH_X, SEARCH_Y);
        expect(search.bestParams).not.toBeNull();
        expect(search.predict([[0.2], [11.5]])).toEqual([0, 1]);

        const again = search.clone();
        again.fit(SEARCH_X, SEARCH_Y);
        expect(again.bestParams).toEqual(search.bestParams);
        expect(again.bestScore).toBe(search.bestScore);
    });

    it('survives serialize → JSON text → revive with identical state and predictions', () => {
        const search = create();
        search.fit(SEARCH_X, SEARCH_Y);
        const pred = search.predict(SEARCH_X);
        const revived = loadModel(JSON.stringify(search)) as RandomizedSearchCV;
        expect(revived.constructor).toBe(RandomizedSearchCV);
        expect(revived.getParams()).toEqual(search.getParams());
        expect(revived.bestParams).toEqual(search.bestParams);
        expect(revived.predict(SEARCH_X)).toEqual(pred);
        expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(search)));
    });
});
