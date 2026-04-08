import fs from 'fs';
import path from 'path';
import { RidgeClassifier } from '../ridgeClassifier';

test('RidgeClassifier initializes', () => {
    expect(new RidgeClassifier()).toBeDefined();
});

test('RidgeClassifier fits binary labels and predicts classes', () => {
    const X = [[0], [1], [2], [3]];
    const y = [0, 0, 1, 1];

    const clf = new RidgeClassifier({ alpha: 1 });
    clf.fit(X, y);

    expect(clf.predict([[0.2], [2.8]])).toEqual([0, 1]);
});

test('RidgeClassifier supports multi-class one-vs-rest fitting', () => {
    const X = [[0], [1], [2], [10], [11], [12], [20], [21], [22]];
    const y = [0, 0, 0, 1, 1, 1, 2, 2, 2];

    const clf = new RidgeClassifier({ alpha: 1 });
    clf.fit(X, y);

    const pred = clf.predict([[0.2], [11.5], [21.4]]);
    expect(pred).toHaveLength(3);
    pred.forEach(label => {
        expect([0, 1, 2]).toContain(label);
    });
});

test('RidgeClassifier compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/ridge_classifier.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new RidgeClassifier({ alpha: 1 });
    clf.fit(data.trainX, data.trainY);
    expect(clf.predict(data.testX)).toEqual(data.expected);
});
