import { IsolationForest } from '../isolationForest';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/isolation_forest.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    // random_state = 0 makes the run deterministic; with this seed the
    // predictions match the sklearn labels exactly (the outlier's score is
    // the highest at every probed seed, but the training-quantile offset
    // fluctuates slightly across seeds, so the seed is part of the fixture)
    const iso = new IsolationForest(256, 100, 0.1, 0);
    iso.fit(data.trainX);
    const pred = iso.predict(data.testX);
    expect(pred).toEqual(data.expected);
});
