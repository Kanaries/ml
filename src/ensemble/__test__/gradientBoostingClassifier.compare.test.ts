import fs from 'fs';
import path from 'path';
import { GradientBoostingClassifier } from '../gradientBoostingClassifier';

test('GradientBoostingClassifier compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/gradient_boosting_classifier.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new GradientBoostingClassifier({
        nEstimators: 100,
        learningRate: 0.1,
        maxDepth: 3,
        randomState: 0,
    });
    clf.fit(data.trainX, data.trainY);
    const pred = clf.predict(data.testX);
    let agree = 0;
    for (let i = 0; i < pred.length; i++) {
        if (pred[i] === data.expected[i]) {
            agree++;
        }
    }
    expect(agree / pred.length).toBeGreaterThanOrEqual(0.95);
});
