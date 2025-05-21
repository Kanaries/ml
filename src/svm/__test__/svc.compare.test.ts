import { SVC } from '../svc';
import { LinearSVC } from '../linearSVC';
import { NuSVC } from '../nuSVC';
import fs from 'fs';
import path from 'path';

test('compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/svc.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const svc = new SVC({ kernel: 'linear', maxIter: 100, learningRate: 0.1 });
    svc.fit(data.trainX, data.trainY);
    const pred1 = svc.predict(data.testX);
    expect(pred1).toEqual(data.expected_svc);

    const lsvc = new LinearSVC({ maxIter: 100, learningRate: 0.1 });
    lsvc.fit(data.trainX, data.trainY);
    const pred2 = lsvc.predict(data.testX);
    expect(pred2).toEqual(data.expected_linsvc);

    const nsvc = new NuSVC({ kernel: 'linear', maxIter: 100, learningRate: 0.1 });
    nsvc.fit(data.trainX, data.trainY);
    const pred3 = nsvc.predict(data.testX);
    expect(pred3).toEqual(data.expected_nusvc);
});
