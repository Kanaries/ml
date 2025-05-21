import { KNearstNeighbors } from '../knn';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/knn.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const knn = new KNearstNeighbors(3);
    knn.fit(data.trainX, data.trainY);
    const pred = knn.predict(data.testX);
    expect(pred).toEqual(data.expected);
});
