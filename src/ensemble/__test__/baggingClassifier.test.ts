import { BaggingClassifier } from '../baggingClassifier';

test('BaggingClassifier initializes', () => {
    expect(new BaggingClassifier()).toBeDefined();
});

test('BaggingClassifier fits and predicts a separable dataset', () => {
    const X = [[0], [1], [2], [10], [11], [12]];
    const y = [0, 0, 0, 1, 1, 1];

    const clf = new BaggingClassifier({ nEstimators: 7, randomState: 42 });
    clf.fit(X, y);

    expect(clf.predict([[0.2], [11.5]])).toEqual([0, 1]);
});

test('BaggingClassifier validates fit lifecycle', () => {
    const clf = new BaggingClassifier();
    expect(() => clf.predict([[0]])).toThrow('model is not fitted');
});
