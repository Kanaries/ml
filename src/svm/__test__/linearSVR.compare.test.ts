import { LinearSVR } from '../linearSVR';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/linear_svr.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const reg = new LinearSVR({ maxIter: 500, randomState: 0 });
    reg.fit(data.trainX, data.trainY);
    const pred = reg.predict(data.testX);
    expect(pred.length).toBe(data.expected.length);

    // R^2 against sklearn's predictions: the old `mse < 1000` bound was
    // vacuous — the sklearn prediction variance on this fixture is ~913, so a
    // constant-zero model passed it
    const mean = data.expected.reduce((a: number, b: number) => a + b, 0) / data.expected.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < pred.length; i++) {
        ssRes += (pred[i] - data.expected[i]) ** 2;
        ssTot += (data.expected[i] - mean) ** 2;
    }
    const r2 = 1 - ssRes / ssTot;
    expect(r2).toBeGreaterThan(0.95);
});
