import { LabelSpreading } from '../labelSpreading';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/label_spreading.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const ls = new LabelSpreading({ gamma: data.gamma, alpha: 0.2 });
    ls.fit(data.trainX, data.trainY);
    const pred = ls.predict(data.testX);
    expect(pred).toEqual(data.expected);
    const proba = ls.predictProba(data.testX);
    expect(proba.length).toBe(data.expectedProba.length);
    for (let i = 0; i < proba.length; i++) {
        expect(proba[i].length).toBe(data.expectedProba[i].length);
        for (let j = 0; j < proba[i].length; j++) {
            expect(proba[i][j]).toBeCloseTo(data.expectedProba[i][j], 3);
        }
    }
});
