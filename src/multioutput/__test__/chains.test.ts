import { getRegisteredEstimators, loadModel } from '../../base';
import { GaussianNB } from '../../bayes';
import { RidgeRegression } from '../../linear';
import { ClassifierChain } from '../classifierChain';
import { RegressorChain } from '../regressorChain';
import fs from 'fs';
import path from 'path';

const waveB = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_b.json'), 'utf8'));

test('ClassifierChain learns dependent binary outputs and serializes', () => {
    const X = Array.from({ length: 20 }, (_, i) => [i - 10, i % 3]);
    const Y = X.map(row => [row[0] >= 0 ? 1 : 0, row[0] >= 0 && row[1] > 0 ? 1 : 0]);
    const model = new ClassifierChain({ estimator: new GaussianNB(), cv: 4 });
    model.fit(X, Y);
    const prediction = model.predict(X);
    expect(model.score(X, Y)).toBeGreaterThan(0.8);
    expect(model.predictProba(X)).toHaveLength(X.length);
    const revived = loadModel(JSON.stringify(model)) as ClassifierChain;
    expect(revived.predict(X)).toEqual(prediction);
    revived.fit(X, Y);
    expect(revived.predict(X)).toEqual(prediction);
});

test('ClassifierChain matches the sklearn GaussianNB prediction and probability fixture', () => {
    const { X, Y, prediction, rows, probability } = waveB.classifier_chain;
    const model = new ClassifierChain({ estimator: new GaussianNB() });
    model.fit(X, Y);
    expect(model.predict(X)).toEqual(prediction);
    model.predictProba(rows.map((row: number) => X[row])).forEach((result, i) => result.forEach((value, j) => expect(value).toBeCloseTo(probability[i][j], 12)));
});

test('RegressorChain restores output order and matches linear targets', () => {
    const X = Array.from({ length: 20 }, (_, i) => [i / 3, (i % 4) - 2]);
    const Y = X.map(row => [2 * row[0] - row[1] + 1, -row[0] + 3 * row[1] - 2]);
    const model = new RegressorChain({ estimator: new RidgeRegression({ alpha: 1e-10 }), order: [1, 0] });
    model.fit(X, Y);
    model.predict(X).forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(Y[i][j], 6)));
    expect(model.chainOrder).toEqual([1, 0]);
    expect(model.score(X, Y)).toBeCloseTo(1, 8);
    const revived = loadModel(JSON.stringify(model)) as RegressorChain;
    expect(revived.predict(X)).toEqual(model.predict(X));
    const fixture = waveB.regressor_chain;
    model.predict(fixture.rows.map((row: number) => X[row])).forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(fixture.prediction[i][j], 10)));
});

test('random chain order is seeded and deterministic', () => {
    const X = [[0], [1], [2], [3], [4], [5]];
    const Y = X.map(([x]) => [x, x + 1, 2 * x]);
    const a = new RegressorChain({ estimator: new RidgeRegression({ alpha: 1e-10 }), order: 'random', randomState: 7 });
    const b = new RegressorChain({ estimator: new RidgeRegression({ alpha: 1e-10 }), order: 'random', randomState: 7 });
    a.fit(X, Y); b.fit(X, Y);
    expect(a.chainOrder).toEqual(b.chainOrder);
    expect(a.predict(X)).toEqual(b.predict(X));
});

test('chain estimators satisfy registration, cloning, and validation contracts', () => {
    expect(getRegisteredEstimators().has('ClassifierChain')).toBe(true);
    expect(getRegisteredEstimators().has('RegressorChain')).toBe(true);
    const source = new RegressorChain({ estimator: new RidgeRegression({ alpha: .5 }), order: [1, 0], cv: 2 });
    const cloned = source.clone() as RegressorChain;
    expect(cloned).not.toBe(source);
    expect(cloned.getParams()).toEqual(source.getParams());
    source.fit([[0], [1], [2], [3]], [[0, 1], [1, 2], [2, 3], [3, 4]]);
    source.setParams({ estimator__alpha: 2 });
    expect((source.getParams().estimator as RidgeRegression).getParams().alpha).toBe(2);
    expect(() => source.predict([[4]])).toThrow('not fitted');
    expect(() => source.setParams({ doesNotExist: 1 })).toThrow('Invalid parameter');
    expect(() => new RegressorChain({ estimator: new RidgeRegression(), order: [2, 0] }).fit([[0], [1]], [[0, 1], [1, 2]])).toThrow('permutation');
});

test('ClassifierChain rejects one-class probability columns instead of reporting them as positive', () => {
    const model = new ClassifierChain({ estimator: new GaussianNB() });
    model.fit([[0], [1], [2], [3]], [[0, 0], [0, 0], [1, 0], [1, 0]]);
    expect(() => model.predictProba([[1]])).toThrow('exactly two classes');
});
