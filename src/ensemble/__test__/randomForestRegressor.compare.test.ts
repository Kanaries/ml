import fs from 'fs';
import path from 'path';
import { RandomForestRegressor } from '../randomForestRegressor';

test('RandomForestRegressor compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/random_forest_regressor.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const reg = new RandomForestRegressor({ nEstimators: 25, randomState: 0, maxFeatures: 1 });
    reg.fit(data.trainX, data.trainY);
    const pred = reg.predict(data.testX);
    let mse = 0;
    for (let i = 0; i < pred.length; i++) {
        mse += (pred[i] - data.expected[i]) ** 2;
    }
    mse /= pred.length;
    expect(mse).toBeLessThan(5000);
});
