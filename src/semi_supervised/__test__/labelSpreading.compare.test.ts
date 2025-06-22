import { LabelSpreading } from '../labelSpreading';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/label_spreading.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const ls = new LabelSpreading();
    ls.fit(data.trainX, data.trainY);
    const pred = ls.predict(data.testX);
    expect(pred).toEqual(data.expected);
});
