import fs from 'fs';
import path from 'path';
import { AdaBoostClassifier } from '../adaBoostClassifier';

// three well-separated 2-D clusters (same layout as multiclass.test.ts)
function threeClusters() {
    const X: number[][] = [];
    const y: number[] = [];
    const centers: Array<[number, number, number]> = [
        [0, 0, 10],
        [8, 0, 20],
        [4, 7, 30],
    ];
    for (const [cx, cy, label] of centers) {
        for (let i = 0; i < 12; i++) {
            const dx = ((i * 7) % 12) / 6 - 1;
            const dy = ((i * 5) % 12) / 6 - 1;
            X.push([cx + dx, cy + dy]);
            y.push(label);
        }
    }
    return { X, y };
}

describe('AdaBoostClassifier multiclass (SAMME)', () => {
    test('separates three clusters with arbitrary labels', () => {
        const { X, y } = threeClusters();
        const clf = new AdaBoostClassifier({ nEstimators: 50 });
        clf.fit(X, y);
        expect(clf.predict(X)).toEqual(y);
        expect(clf.predict([[0, 0.5], [8, -0.5], [4, 6.5]])).toEqual([10, 20, 30]);
    });

    test('predictProba has K columns ordered by sorted labels, rows sum to 1', () => {
        const { X, y } = threeClusters();
        const clf = new AdaBoostClassifier({ nEstimators: 50 });
        clf.fit(X, y);
        const proba = clf.predictProba(X);
        for (let i = 0; i < X.length; i++) {
            expect(proba[i]).toHaveLength(3);
            expect(proba[i][0] + proba[i][1] + proba[i][2]).toBeCloseTo(1, 9);
            // argmax of proba agrees with predict
            const best = proba[i].indexOf(Math.max(...proba[i]));
            expect([10, 20, 30][best]).toBe(clf.predict([X[i]])[0]);
        }
    });

    test('a failed refit does not corrupt a fitted multiclass model', () => {
        const { X, y } = threeClusters();
        const clf = new AdaBoostClassifier({ nEstimators: 20 });
        clf.fit(X, y);
        expect(() => clf.fit([[1], [2]], [1, 1])).toThrow();
        expect(clf.predict([[0, 0.5], [8, -0.5], [4, 6.5]])).toEqual([10, 20, 30]);
    });

    test('binary fit after a multiclass fit works (state fully reset)', () => {
        const { X, y } = threeClusters();
        const clf = new AdaBoostClassifier({ nEstimators: 20 });
        clf.fit(X, y);
        const Xb = [[1], [2], [3], [4], [6], [7], [8], [9]];
        const yb = [0, 0, 0, 0, 1, 1, 1, 1];
        clf.fit(Xb, yb);
        expect(clf.predict(Xb)).toEqual(yb);
        expect(clf.predictProba([[1]])[0]).toHaveLength(2);
    });

    test('getFeatureImportances covers the multiclass path', () => {
        const { X, y } = threeClusters();
        const clf = new AdaBoostClassifier({ nEstimators: 20 });
        clf.fit(X, y);
        const importances = clf.getFeatureImportances();
        expect(importances.length).toBeGreaterThan(0);
        const sum = importances.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1, 9);
    });

    test('identical rows with conflicting labels fall back to the majority class', () => {
        const X = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]];
        const y = [7, 7, 7, 3, 5];
        const clf = new AdaBoostClassifier({ nEstimators: 10 });
        clf.fit(X, y);
        expect(clf.predict([[1, 1]])).toEqual([7]);
    });
});

test('multiclass AdaBoostClassifier compare with sklearn (SAMME)', () => {
    const p = path.join(__dirname, '../../../test_data/adaboost_classifier_multi.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new AdaBoostClassifier({ nEstimators: 50, learningRate: 1.0 });
    clf.fit(data.trainX, data.trainY);
    const pred = clf.predict(data.testX);
    let agree = 0;
    for (let i = 0; i < pred.length; i++) {
        if (pred[i] === data.expected[i]) {
            agree++;
        }
    }
    // our stump minimizes weighted misclassification while sklearn's
    // depth-1 tree minimizes weighted gini, so stump choices can differ
    expect(agree / pred.length).toBeGreaterThanOrEqual(0.8);
});
