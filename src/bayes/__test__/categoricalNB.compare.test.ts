import { CategoricalNB } from '../categoricalNB';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/categorical_nb.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new CategoricalNB();
    clf.fit(data.trainX, data.trainY);
    const pred = clf.predict(data.testX);
    expect(pred).toEqual(data.expected);
});
