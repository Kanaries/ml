import { SVC } from '../svc';
import { LinearSVC } from '../linearSVC';
import fs from 'fs';
import path from 'path';

const p = path.join(__dirname, '../../../test_data/svc.json');
const data = JSON.parse(fs.readFileSync(p, 'utf8'));

describe('SVC vs sklearn: rbf kernel on circles', () => {
    const clf = new SVC({ kernel: 'rbf', C: 1, gamma: 'scale' });
    clf.fit(data.rbf.trainX, data.rbf.trainY);

    test('predictions match point-for-point', () => {
        expect(clf.predict(data.rbf.testX)).toEqual(data.rbf.expected);
    });

    test('decision function matches sklearn values', () => {
        const dec = clf.decisionFunction(data.rbf.testX);
        // precision 2, not 3: both solvers stop at tol=1e-3, and sklearn's own
        // convergence slack leaves ~6e-4 residual even if ours is tightened
        dec.forEach((d, i) => expect(d).toBeCloseTo(data.rbf.decision[i], 2));
    });

    test('support vector count is within 15% of sklearn', () => {
        const ours = clf.getNSupport().reduce((s: number, v: number) => s + v, 0);
        const sk = data.rbf.nSupport.reduce((s: number, v: number) => s + v, 0);
        expect(Math.abs(ours - sk) / sk).toBeLessThanOrEqual(0.15);
    });
});

describe('SVC vs sklearn: linear kernel', () => {
    const clf = new SVC({ kernel: 'linear' });
    clf.fit(data.linear.trainX, data.linear.trainY);

    test('predictions match point-for-point', () => {
        expect(clf.predict(data.linear.testX)).toEqual(data.linear.expected);
    });

    test('decision function matches sklearn values', () => {
        const dec = clf.decisionFunction(data.linear.testX);
        dec.forEach((d, i) => expect(d).toBeCloseTo(data.linear.decision[i], 3));
    });
});

describe('SVC vs sklearn: one-vs-one multiclass on blobs', () => {
    const clf = new SVC({ kernel: 'rbf', C: 1, gamma: 'scale' });
    clf.fit(data.multiclass.trainX, data.multiclass.trainY);

    test('predictions match point-for-point', () => {
        expect(clf.predict(data.multiclass.testX)).toEqual(data.multiclass.expected);
    });

    test('support vector count is within 15% of sklearn', () => {
        const ours = clf.getNSupport().reduce((s: number, v: number) => s + v, 0);
        const sk = data.multiclass.nSupport.reduce((s: number, v: number) => s + v, 0);
        expect(Math.abs(ours - sk) / sk).toBeLessThanOrEqual(0.15);
    });
});

test('LinearSVC compare with sklearn', () => {
    const lsvc = new LinearSVC({ maxIter: 100 });
    lsvc.fit(data.linear.trainX, data.linear.trainY);
    expect(lsvc.predict(data.linear.testX)).toEqual(data.linear.expected_linsvc);
});
