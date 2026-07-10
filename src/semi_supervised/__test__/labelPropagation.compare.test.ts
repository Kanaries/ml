import { LabelPropagation } from '../labelPropagation';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/label_propagation.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const lp = new LabelPropagation({ gamma: data.gamma });
    lp.fit(data.trainX, data.trainY);
    const pred = lp.predict(data.testX);
    expect(pred).toEqual(data.expected);
    const proba = lp.predictProba(data.testX);
    expect(proba.length).toBe(data.expectedProba.length);
    for (let i = 0; i < proba.length; i++) {
        expect(proba[i].length).toBe(data.expectedProba[i].length);
        for (let j = 0; j < proba[i].length; j++) {
            expect(proba[i][j]).toBeCloseTo(data.expectedProba[i][j], 3);
        }
    }
});
