import { AdaBoostClassifier } from '../adaBoostClassifier';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/adaboost_classifier.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const clf = new AdaBoostClassifier();
    clf.fit(data.trainX, data.trainY);
    const pred = clf.predict(data.testX);
    let correct = 0;
    for (let i = 0; i < pred.length; i++) {
        if (pred[i] === data.expected[i]) correct++;
    }
    const acc = correct / pred.length;
    expect(acc).toBeGreaterThanOrEqual(0.7);
});
