import { BernoulliRBM } from '../bernoulliRBM';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/bernoulli_rbm.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const rbm = new BernoulliRBM({ nComponents: 2, learningRate: 0.1, batchSize: 10, nIter: 20 });
    rbm.fit(data.trainX);
    const pred = rbm.transform(data.X_test);
    for (let i = 0; i < pred.length; i++) {
        for (let j = 0; j < pred[i].length; j++) {
            expect(pred[i][j]).toBeCloseTo(data.expected[i][j], 1);
        }
    }
});
