import fs from 'fs';
import path from 'path';
import { RadiusNeighborsClassifier } from '../radiusNeighborsClassifier';

test('RadiusNeighborsClassifier compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/radius_neighbors_classifier.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new RadiusNeighborsClassifier({ radius: 1.6 });
    clf.fit(data.trainX, data.trainY);
    expect(clf.predict(data.testX)).toEqual(data.expected);
});
