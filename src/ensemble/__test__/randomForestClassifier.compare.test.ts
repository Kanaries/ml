import fs from 'fs';
import path from 'path';
import { RandomForestClassifier } from '../randomForestClassifier';

test('RandomForestClassifier compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/random_forest_classifier.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new RandomForestClassifier({ nEstimators: 25, randomState: 0, maxFeatures: 'sqrt' });
    clf.fit(data.trainX, data.trainY);
    const pred = clf.predict(data.testX);
    let correct = 0;
    for (let i = 0; i < pred.length; i++) {
        if (pred[i] === data.expected[i]) {
            correct++;
        }
    }
    expect(correct / pred.length).toBeGreaterThanOrEqual(0.8);
});
