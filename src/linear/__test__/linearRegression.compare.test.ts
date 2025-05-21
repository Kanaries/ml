import { LinearRegression } from '../linearRegression';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/linear_regression.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const lr = new LinearRegression();
    lr.fit(data.trainX, data.trainY);
    const pred = lr.predict(data.testX);
    expect(pred.length).toBe(data.expected.length);
    for (let i = 0; i < pred.length; i++) {
        expect(pred[i]).toBeCloseTo(data.expected[i], 4);
    }
});
