import { IsolationForest } from '../isolationForest';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/isolation_forest.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const iso = new IsolationForest(256, 100, 0.1);
    iso.fit(data.trainX);
    const pred = iso.predict(data.testX);
    expect(pred).toEqual(data.expected);
});
