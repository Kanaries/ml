import { MinMaxScaler, StandardScaler } from '../preprocessing';

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
