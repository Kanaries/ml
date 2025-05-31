import { BernoulliNB } from '../bernoulliNB';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/bernoulli_nb.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new BernoulliNB();
    clf.fit(data.trainX, data.trainY);
    const pred = clf.predict(data.testX);
    expect(pred).toEqual(data.expected);
});
