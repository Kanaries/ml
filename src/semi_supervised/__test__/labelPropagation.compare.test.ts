import { LabelPropagation } from '../labelPropagation';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/label_propagation.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const lp = new LabelPropagation();
    lp.fit(data.trainX, data.trainY);
    const pred = lp.predict(data.testX);
    expect(pred).toEqual(data.expected);
});
