import fs from 'fs';
import path from 'path';
import { GradientBoostingClassifier } from '../gradientBoostingClassifier';
import { XGBoostClassifier } from '../xgboost';

function agreement(pred: number[], expected: number[]): number {
    let agree = 0;
    for (let i = 0; i < pred.length; i++) {
        if (pred[i] === expected[i]) {
            agree++;
        }
    }
    return agree / pred.length;
}

test('multiclass GradientBoostingClassifier compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/gradient_boosting_classifier_multi.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new GradientBoostingClassifier({
        nEstimators: 100,
        learningRate: 0.1,
        maxDepth: 3,
        randomState: 0,
    });
    clf.fit(data.trainX, data.trainY);
    expect(agreement(clf.predict(data.testX), data.expected)).toBeGreaterThanOrEqual(0.9);
});

test('multiclass XGBoostClassifier compare with python xgboost', () => {
    const p = path.join(__dirname, '../../../test_data/xgboost_classifier_multi.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new XGBoostClassifier({
        nEstimators: 50,
        learningRate: 0.3,
        maxDepth: 6,
        lambda: 1,
        randomState: 0,
    });
    clf.fit(data.trainX, data.trainY);
    expect(agreement(clf.predict(data.testX), data.expected)).toBeGreaterThanOrEqual(0.9);
});
