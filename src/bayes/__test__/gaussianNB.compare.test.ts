import fs from 'fs';
import path from 'path';
import { GaussianNB } from '../gaussianNB';

test('GaussianNB compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/gaussian_nb.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new GaussianNB();
    clf.fit(data.trainX, data.trainY);
    expect(clf.predict(data.testX)).toEqual(data.expected);
});
