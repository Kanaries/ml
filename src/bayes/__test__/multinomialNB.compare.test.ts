import fs from 'fs';
import path from 'path';
import { MultinomialNB } from '../multinomialNB';

test('MultinomialNB compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/multinomial_nb.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new MultinomialNB();
    clf.fit(data.trainX, data.trainY);
    expect(clf.predict(data.testX)).toEqual(data.expected);
});
