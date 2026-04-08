import fs from 'fs';
import path from 'path';
import { KNeighborsRegressor } from '../kneighborsRegressor';

test('KNeighborsRegressor compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/kneighbors_regressor.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const reg = new KNeighborsRegressor({ nNeighbors: 3 });
    reg.fit(data.trainX, data.trainY);
    const pred = reg.predict(data.testX);
    expect(pred.length).toBe(data.expected.length);
    for (let i = 0; i < pred.length; i++) {
        expect(pred[i]).toBeCloseTo(data.expected[i], 8);
    }
});
