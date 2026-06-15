import { AdaBoostClassifier } from '../adaBoostClassifier';

test('init', () => {
    const clf = new AdaBoostClassifier();
    expect(clf).toBeDefined();
});

test('simple classification', () => {
    const X = [[0], [1], [2], [3]];
    const Y = [0, 0, 1, 1];
    const clf = new AdaBoostClassifier({ nEstimators: 5 });
    clf.fit(X, Y);
    const pred = clf.predict(X);
    expect(pred).toEqual(Y);
});

test('supports snake_case constructor aliases', () => {
    const X = [[0], [1], [2], [3]];
    const Y = [0, 0, 1, 1];
    const clf = new AdaBoostClassifier({ n_estimators: 5, learning_rate: 0.5 });
    clf.fit(X, Y);
    const pred = clf.predict(X);
    expect(pred).toEqual(Y);
});
