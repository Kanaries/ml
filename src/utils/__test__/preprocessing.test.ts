import {
    Binarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    SelectKBest,
    SimpleImputer,
    StandardScaler,
    VarianceThreshold,
    fRegression,
} from '../preprocessing';

test('StandardScaler fitTransform centers and scales features', () => {
    const X = [
        [1, 2],
        [3, 4],
        [5, 6],
    ];
    const scaler = new StandardScaler();
    const transformed = scaler.fitTransform(X);

    expect(transformed[0][0]).toBeCloseTo(-1.2247448714);
    expect(transformed[1][0]).toBeCloseTo(0);
    expect(transformed[2][0]).toBeCloseTo(1.2247448714);
    expect(transformed[0][1]).toBeCloseTo(-1.2247448714);
    expect(transformed[1][1]).toBeCloseTo(0);
    expect(transformed[2][1]).toBeCloseTo(1.2247448714);
});

test('StandardScaler inverseTransform recovers original data', () => {
    const X = [
        [10, -1],
        [20, 0],
        [30, 1],
    ];
    const scaler = new StandardScaler();
    const transformed = scaler.fitTransform(X);
    const restored = scaler.inverseTransform(transformed);

    expect(restored).toHaveLength(X.length);
    for (let i = 0; i < X.length; i++) {
        for (let j = 0; j < X[0].length; j++) {
            expect(restored[i][j]).toBeCloseTo(X[i][j], 8);
        }
    }
});

test('StandardScaler handles constant features', () => {
    const X = [
        [2, 1],
        [2, 3],
        [2, 5],
    ];
    const scaler = new StandardScaler();
    const transformed = scaler.fitTransform(X);
    expect(transformed[0][0]).toBeCloseTo(0);
    expect(transformed[1][0]).toBeCloseTo(0);
    expect(transformed[2][0]).toBeCloseTo(0);
});

test('MinMaxScaler maps values into target feature range', () => {
    const X = [
        [0, 10],
        [5, 20],
        [10, 30],
    ];
    const scaler = new MinMaxScaler({ featureRange: [-1, 1] });
    const transformed = scaler.fitTransform(X);

    expect(transformed[0][0]).toBeCloseTo(-1);
    expect(transformed[1][0]).toBeCloseTo(0);
    expect(transformed[2][0]).toBeCloseTo(1);
    expect(transformed[0][1]).toBeCloseTo(-1);
    expect(transformed[1][1]).toBeCloseTo(0);
    expect(transformed[2][1]).toBeCloseTo(1);
});

test('MinMaxScaler inverseTransform recovers original data', () => {
    const X = [
        [4, 100],
        [6, 200],
        [8, 300],
    ];
    const scaler = new MinMaxScaler();
    const transformed = scaler.fitTransform(X);
    const restored = scaler.inverseTransform(transformed);

    expect(restored).toHaveLength(X.length);
    for (let i = 0; i < X.length; i++) {
        for (let j = 0; j < X[0].length; j++) {
            expect(restored[i][j]).toBeCloseTo(X[i][j], 8);
        }
    }
});

test('scalers validate fit and options', () => {
    const stdScaler = new StandardScaler();
    expect(() => stdScaler.transform([[1, 2]])).toThrow('StandardScaler must be fitted before calling transform');
    expect(() => new MinMaxScaler({ featureRange: [1, 1] })).toThrow('featureRange min must be less than max');
});

test('MaxAbsScaler scales by per-feature max absolute value', () => {
    const X = [
        [-2, 0],
        [4, -3],
        [0, 6],
    ];
    const scaler = new MaxAbsScaler();
    const transformed = scaler.fitTransform(X);

    expect(transformed).toEqual([
        [-0.5, 0],
        [1, -0.5],
        [0, 1],
    ]);
    expect(scaler.inverseTransform(transformed)).toEqual(X);
});

test('Normalizer supports l1, l2, and max norms', () => {
    const X = [[3, 4]];

    expect(new Normalizer({ norm: 'l2' }).transform(X)[0]).toEqual([0.6, 0.8]);
    expect(new Normalizer({ norm: 'l1' }).transform(X)[0]).toEqual([3 / 7, 4 / 7]);
    expect(new Normalizer({ norm: 'max' }).transform(X)[0]).toEqual([0.75, 1]);
});

test('LabelEncoder encodes and decodes labels in sorted order', () => {
    const encoder = new LabelEncoder();
    const encoded = encoder.fitTransform([5, 2, 5, 3]);

    expect(encoded).toEqual([2, 0, 2, 1]);
    expect(encoder.inverseTransform([0, 1, 2])).toEqual([2, 3, 5]);
});

test('Binarizer thresholds feature values', () => {
    const binarizer = new Binarizer({ threshold: 1.5 });
    expect(binarizer.transform([[1, 2, 1.5]])).toEqual([[0, 1, 0]]);
});

test('SimpleImputer supports mean and constant strategies', () => {
    const X = [
        [1, NaN],
        [3, 4],
        [5, NaN],
    ];

    const meanImputer = new SimpleImputer();
    expect(meanImputer.fitTransform(X)).toEqual([
        [1, 4],
        [3, 4],
        [5, 4],
    ]);

    const constantImputer = new SimpleImputer({ strategy: 'constant', fillValue: -1 });
    expect(constantImputer.fitTransform(X)).toEqual([
        [1, -1],
        [3, 4],
        [5, -1],
    ]);
});

test('VarianceThreshold removes low-variance features', () => {
    const selector = new VarianceThreshold({ threshold: 0.01 });
    const transformed = selector.fitTransform([
        [1, 0, 10],
        [1, 1, 11],
        [1, 0, 12],
    ]);

    expect(transformed).toEqual([
        [0, 10],
        [1, 11],
        [0, 12],
    ]);
});

test('SelectKBest keeps highest-scoring features using fRegression', () => {
    const X = [
        [0, 10, 0],
        [1, 10, 0],
        [2, 10, 1],
        [3, 10, 1],
    ];
    const y = [0, 1, 2, 3];

    const selector = new SelectKBest({ k: 1, scoreFunc: fRegression });
    const transformed = selector.fitTransform(X, y);

    expect(transformed).toEqual([[0], [1], [2], [3]]);
});
