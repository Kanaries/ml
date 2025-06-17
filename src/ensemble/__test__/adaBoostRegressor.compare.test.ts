import { AdaBoostRegressor } from '../adaBoostRegressor';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/ada_boost_regressor.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const reg = new AdaBoostRegressor({ n_estimators: 50 });
    reg.fit(data.trainX, data.trainY);
    const pred = reg.predict(data.testX);
    expect(pred.length).toBe(data.expected.length);
    let mse = 0;
    for (let i = 0; i < pred.length; i++) {
        mse += (pred[i] - data.expected[i]) ** 2;
    }
    mse /= pred.length;
    expect(mse).toBeLessThan(100);
});
