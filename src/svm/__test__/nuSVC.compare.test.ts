import { NuSVC } from '../nuSVC';
import fs from 'fs';
import path from 'path';

const p = path.join(__dirname, '../../../test_data/nu_svc.json');
const data = JSON.parse(fs.readFileSync(p, 'utf8'));

function fitCase(nu: number): NuSVC {
    const clf = new NuSVC({ nu, kernel: 'rbf', gamma: 'scale' });
    clf.fit(data.trainX, data.trainY);
    return clf;
}

describe.each([['nu_0_3'], ['nu_0_6']])('NuSVC vs sklearn: circles, %s', (key: string) => {
    const expected = data.cases[key];
    const clf = fitCase(expected.nu);

    test('predictions match point-for-point', () => {
        expect(clf.predict(data.testX)).toEqual(expected.expected);
    });

    test('decision function matches sklearn values', () => {
        const dec = clf.decisionFunction(data.testX);
        dec.forEach((d, i) => expect(d).toBeCloseTo(expected.decision[i], 3));
    });

    test('support vector count is within 15% of sklearn', () => {
        const ours = clf.getNSupport().reduce((s: number, v: number) => s + v, 0);
        const sk = expected.nSupport.reduce((s: number, v: number) => s + v, 0);
        expect(Math.abs(ours - sk) / sk).toBeLessThanOrEqual(0.15);
    });
});

test('larger nu yields more support vectors (matching the sklearn fixture)', () => {
    const skLow = data.cases.nu_0_3.nSupport.reduce((s: number, v: number) => s + v, 0);
    const skHigh = data.cases.nu_0_6.nSupport.reduce((s: number, v: number) => s + v, 0);
    expect(skHigh).toBeGreaterThan(skLow);

    const low = fitCase(0.3).getNSupport().reduce((s: number, v: number) => s + v, 0);
    const high = fitCase(0.6).getNSupport().reduce((s: number, v: number) => s + v, 0);
    expect(high).toBeGreaterThan(low);
});
