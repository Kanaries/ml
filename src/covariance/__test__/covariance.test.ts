import { getRegisteredEstimators, loadModel } from '../../base';
import { EllipticEnvelope } from '../ellipticEnvelope';
import { MinCovDet } from '../minCovDet';

const X = [[0, 0], [.1, .2], [.2, .1], [-.1, 0], [.05, -.1], [8, 8]];
const robustX = [
    [0, 0], [.1, .2], [.2, .1], [-.1, 0], [.05, -.1],
    [.12, .05], [-.05, .08], [.07, .12], [8, 8], [-7, 9],
];

test('MinCovDet rejects the remote outlier and estimates the sklearn fixture center', () => {
    const model = new MinCovDet({ randomState: 0, supportFraction: .7 });
    model.fit(robustX);
    expect(model.rawSupport).toEqual([true, true, true, true, false, true, true, true, false, false]);
    expect(model.support).toEqual([true, true, true, true, true, true, true, true, false, false]);
    [0.048571428571428585, 0.07857142857142858]
        .forEach((value, i) => expect(model.rawLocation[i]).toBeCloseTo(value, 12));
    [[0.009326530612244898, 0.0033836734693877555], [0.0033836734693877555, 0.004297959183673468]]
        .forEach((row, i) => row.forEach((value, j) => expect(model.rawCovariance[i][j]).toBeCloseTo(value, 12)));
    [0.04875, 0.05625].forEach((value, i) => expect(model.location[i]).toBeCloseTo(value, 12));
    [[0.0081609375, 0.0029328125], [0.0029328125, 0.0072484375]]
        .forEach((row, i) => row.forEach((value, j) => expect(model.covariance[i][j]).toBeCloseTo(value, 12)));
    const expectedDistances = [
        0.5333733081533181, 2.857682428903032, 2.821334190525349, 2.712536742320045,
        3.964155567987589, 0.7858730222509323, 1.7614028689145291, 0.5636418709452087,
        11923.319329782993, 27352.049892284347,
    ];
    expectedDistances.forEach((value, i) => expect(model.mahalanobis(robustX)[i]).toBeCloseTo(value, 8));
    const revived = loadModel(JSON.stringify(model)) as MinCovDet;
    expect(revived.mahalanobis(robustX)).toEqual(model.mahalanobis(robustX));
    expect(getRegisteredEstimators().get('MinCovDet')).toBe(MinCovDet);
});

test('EllipticEnvelope matches sklearn fixture labels', () => {
    const model = new EllipticEnvelope({ randomState: 0, supportFraction: .8, contamination: 1 / 6 });
    model.fit(X);
    expect(model.predict(X)).toEqual([1, 1, 1, 1, 1, -1]);
    [0.05, 0.04].forEach((value, i) => expect(model.location[i]).toBeCloseTo(value, 12));
    [[0.01, 0.005], [0.005, 0.0104]]
        .forEach((row, i) => row.forEach((value, j) => expect(model.covariance[i][j]).toBeCloseTo(value, 12)));
    [-0.27848101265822783, -2.5569620253164556, -2.278481012658228, -2.4050632911392418, -2.481012658227848, -8330.40506329114]
        .forEach((value, i) => expect(model.scoreSamples(X)[i]).toBeCloseTo(value, 8));
    expect(model.offset).toBeCloseTo(-1390.5316455696213, 8);
    [1390.2531645569632, 1387.974683544305, 1388.2531645569632, 1388.1265822784821, 1388.0506329113934, -6939.873417721517]
        .forEach((value, i) => expect(model.decisionFunction(X)[i]).toBeCloseTo(value, 8));
    const revived = loadModel(JSON.stringify(model)) as EllipticEnvelope;
    expect(revived.predict(X)).toEqual(model.predict(X));
});

test('EllipticEnvelope treats a zero decision value as an inlier like sklearn', () => {
    const model = new EllipticEnvelope({ randomState: 0, supportFraction: .8, contamination: .2 });
    model.fit(X);
    const decisions = model.decisionFunction(X);
    const boundary = decisions.findIndex(value => value === 0);
    expect(boundary).toBeGreaterThanOrEqual(0);
    expect(model.predict([X[boundary]])).toEqual([1]);
});

test('MinCovDet uses a symmetric pseudoinverse for rank-deficient covariance', () => {
    const collinear = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [100, 100]];
    const model = new MinCovDet({ supportFraction: 5 / 6, randomState: 0 });
    expect(() => model.fit(collinear)).not.toThrow();
    expect(model.precision.flat().every(Number.isFinite)).toBe(true);
    expect(model.mahalanobis(collinear).every(Number.isFinite)).toBe(true);
    expect(model.mahalanobis([[100, 100]])[0]).toBeGreaterThan(model.mahalanobis([[2, 2]])[0]);
});

test('assumeCentered keeps the ordinary FAST-MCD support and recomputes zero-mean covariance', () => {
    const ordinary = new MinCovDet({ supportFraction: .7, randomState: 0 });
    const centered = new MinCovDet({ supportFraction: .7, randomState: 0, assumeCentered: true });
    ordinary.fit(robustX);
    centered.fit(robustX);
    expect(centered.rawSupport).toEqual(ordinary.rawSupport);
    expect(centered.rawLocation).toEqual([0, 0]);
    expect(centered.location).toEqual([0, 0]);
});

test('large-sample FAST-MCD uses staged candidate pools and rejects remote points', () => {
    // 899 rows produce two unequal balanced partitions (449/450), exercising
    // sklearn's ceil-based proportional h calculation inside each subset.
    const large = Array.from({ length: 899 }, (_, i) => i < 879
        ? [Math.sin(i * .37) + (i % 5) * .01, Math.cos(i * .23) + (i % 7) * .01]
        : [30 + i, -40 - i]);
    const model = new MinCovDet({ supportFraction: .8, randomState: 42 });
    const started = Date.now();
    model.fit(large);
    expect(Date.now() - started).toBeLessThan(15000);
    const distances = model.mahalanobis(large);
    expect(Math.min(...distances.slice(879))).toBeGreaterThan(Math.max(...distances.slice(0, 879)));
});
