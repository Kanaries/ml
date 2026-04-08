import fs from 'fs';
import path from 'path';
import { ComplementNB } from '../complementNB';

test('ComplementNB compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/complement_nb.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new ComplementNB();
    clf.fit(data.trainX, data.trainY);
    expect(clf.predict(data.testX)).toEqual(data.expected);
});
