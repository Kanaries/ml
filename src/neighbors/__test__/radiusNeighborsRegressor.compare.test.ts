import fs from 'fs';
import path from 'path';
import { RadiusNeighborsRegressor } from '../radiusNeighborsRegressor';

test('RadiusNeighborsRegressor compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/radius_neighbors_regressor.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const reg = new RadiusNeighborsRegressor({ radius: 1.6 });
    reg.fit(data.trainX, data.trainY);
    const pred = reg.predict(data.testX);
    expect(pred.length).toBe(data.expected.length);
    for (let i = 0; i < pred.length; i++) {
        expect(pred[i]).toBeCloseTo(data.expected[i], 8);
    }
});
