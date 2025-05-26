import { LogisticRegression } from '../logisticRegression';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/logistic_regression.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const lr = new LogisticRegression({ learningRate: 0.1, maxIter: 1000 });
    lr.fit(data.trainX, data.trainY);
    const pred = lr.predict(data.testX);
    expect(pred).toEqual(data.expected);
});
