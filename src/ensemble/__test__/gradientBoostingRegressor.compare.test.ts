import fs from 'fs';
import path from 'path';
import { GradientBoostingRegressor } from '../gradientBoostingRegressor';

test('GradientBoostingRegressor compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/gradient_boosting_regressor.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const reg = new GradientBoostingRegressor({
        nEstimators: 100,
        learningRate: 0.1,
        maxDepth: 3,
        randomState: 0,
    });
    reg.fit(data.trainX, data.trainY);
    const pred = reg.predict(data.testX);
    // R^2 against sklearn's predictions
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
