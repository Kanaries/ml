import fs from 'fs';
import path from 'path';
import { NearestCentroid } from '../nearestCentroid';

test('NearestCentroid compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/nearest_centroid.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new NearestCentroid();
    clf.fit(data.trainX, data.trainY);
    expect(clf.predict(data.testX)).toEqual(data.expected);
});
