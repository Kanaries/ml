import { RandomForestClassifier } from '../randomForestClassifier';

test('RandomForestClassifier initializes', () => {
    expect(new RandomForestClassifier()).toBeDefined();
});

test('RandomForestClassifier fits and predicts a separable dataset', () => {
    const X = [[0], [1], [2], [10], [11], [12]];
    const y = [0, 0, 0, 1, 1, 1];

    const clf = new RandomForestClassifier({ nEstimators: 15, randomState: 42 });
    clf.fit(X, y);

    expect(clf.predict([[0.2], [11.5]])).toEqual([0, 1]);
});
