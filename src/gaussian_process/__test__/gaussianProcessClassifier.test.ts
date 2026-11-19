import { GaussianProcessClassifier } from '../gaussianProcessClassifier';

const X = [[-2, 1], [-1, 1], [1, 1], [2, 1]];
const labels = [-7, -7, 10, 10];

test('uncertain predictions return the class prior at vanishing covariance', () => {
    const model = new GaussianProcessClassifier({ lengthScale: [0.8, 1.4], signalVariance: 2 });
    model.fit(X, labels);
    expect(model.predict(X)).toEqual(labels);
    const [left, right, far] = model.predictProba([X[0], X[3], [1e6, 1e6]]);
    expect(left[0]).toBeGreaterThan(0.5);
    expect(left[0]).toBeCloseTo(right[1], 11);
    expect(far).toEqual([0.5, 0.5]);
    expect(model.predict([[1e6, 1e6]])).toEqual([-7]);
});

test('repeated contradictory observations fit with symmetric probabilities', () => {
    const model = new GaussianProcessClassifier();
    model.fit([[0], [0], [0], [0]], [2, 9, 2, 9]);
    expect(model.predictProba([[0], [1]])).toEqual([[0.5, 0.5], [0.5, 0.5]]);
    expect(model.predictProba([])).toEqual([]);
});

test('multi-class probabilities normalize and a failed refit preserves the previous model', () => {
    const model = new GaussianProcessClassifier({ lengthScale: 0.6 });
    const train = [[0], [0.1], [3], [3.1], [6], [6.1]];
    const y = [4, 4, -3, -3, 0, 0];
    model.fit(train, y);
    expect(model.getClasses()).toEqual([-3, 0, 4]);
    expect(model.predict(train)).toEqual(y);
    const [p] = model.predictProba([[3.2]]);
    expect(p[0]).toBeGreaterThan(p[1]);
    expect(p[0]).toBeGreaterThan(p[2]);
    expect(p.reduce((a, b) => a + b)).toBeCloseTo(1, 12);
    expect(() => model.fit(train, y.map(() => 2))).toThrow(/two classes/);
    expect(model.predictProba([[3.2]])[0]).toEqual(p);
    train[0][0] = 3.2;
    y[0] = 99;
    expect(model.predictProba([[3.2]])[0]).toEqual(p);
});

test('kernel parameters and data dimensions are validated', () => {
    const model = new GaussianProcessClassifier({ lengthScale: [1, 2] });
    expect(() => model.predictProba([[2, 3]])).toThrow(/fit/);
    expect(() => model.fit([[1], [2]], [0, 1])).toThrow();
    expect(() => model.fit(X, [1, 1, NaN, 2])).toThrow();
    model.fit(X, labels);
    expect(() => model.predict([[1]])).toThrow();
    expect(() => model.predict([[1, Infinity]])).toThrow();
    expect(() => new GaussianProcessClassifier({ signalVariance: 0 })).toThrow();
    expect(() => new GaussianProcessClassifier({ maxIterPredict: 0 })).toThrow(/maxIterPredict/);
    expect(() => new GaussianProcessClassifier({ tol: NaN })).toThrow(/tol/);
    const stopped = new GaussianProcessClassifier({ maxIterPredict: 1 });
    expect(() => stopped.fit(X, labels)).toThrow(/converge/);
    expect(() => stopped.predict(X)).toThrow(/fit/);
});
