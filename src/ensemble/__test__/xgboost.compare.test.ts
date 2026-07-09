import fs from 'fs';
import path from 'path';
import { XGBoostClassifier, XGBoostRegressor } from '../xgboost';

test('XGBoostRegressor compare with python xgboost', () => {
    const p = path.join(__dirname, '../../../test_data/xgboost_regressor.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const reg = new XGBoostRegressor({
        nEstimators: 50,
        learningRate: 0.3,
        maxDepth: 6,
        lambda: 1,
        baseScore: 0.5,
        randomState: 0,
    });
    reg.fit(data.trainX, data.trainY);
    const pred = reg.predict(data.testX);
    const meanExpected = data.expected.reduce((a: number, b: number) => a + b, 0) / data.expected.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < pred.length; i++) {
        ssRes += (pred[i] - data.expected[i]) ** 2;
        ssTot += (data.expected[i] - meanExpected) ** 2;
    }
    const r2 = 1 - ssRes / ssTot;
    expect(r2).toBeGreaterThanOrEqual(0.95);
});

test('XGBoostClassifier compare with python xgboost', () => {
    const p = path.join(__dirname, '../../../test_data/xgboost_classifier.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new XGBoostClassifier({
        nEstimators: 50,
        learningRate: 0.3,
        maxDepth: 6,
        lambda: 1,
        baseScore: 0.5,
        randomState: 0,
    });
    clf.fit(data.trainX, data.trainY);
    const pred = clf.predict(data.testX);
    let agree = 0;
    for (let i = 0; i < pred.length; i++) {
        if (pred[i] === data.expected[i]) {
            agree++;
        }
    }
    expect(agree / pred.length).toBeGreaterThanOrEqual(0.9);
});
