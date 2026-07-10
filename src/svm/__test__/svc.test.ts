import { SVC } from '../svc';

// four tight clusters at the XOR corners: not linearly separable
const xorX: number[][] = [];
const xorY: number[] = [];
for (const [cx, cy, label] of [
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
]) {
    for (let i = 0; i < 4; i++) {
        xorX.push([cx + 0.05 * (i % 2), cy + 0.05 * Math.floor(i / 2)]);
        xorY.push(label);
    }
}

describe('SVC (SMO)', () => {
    test('rbf kernel fits XOR with perfect training accuracy', () => {
        const clf = new SVC({ kernel: 'rbf', C: 10 });
        clf.fit(xorX, xorY);
        expect(clf.score(xorX, xorY)).toBe(1);
    });

    test('linear kernel separates linearly separable data', () => {
        const X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]];
        const y = [0, 0, 0, 1, 1, 1];
        const clf = new SVC({ kernel: 'linear' });
        clf.fit(X, y);
        expect(clf.predict([[-1, -1], [2, 2], [3, 2]])).toEqual([0, 1, 1]);
    });

    test('poly kernel learns a quadratic boundary', () => {
        // class 1 = |x| large, class 0 = |x| small: needs the x^2 feature
        const X = [[-3], [-2.5], [-2.2], [2.2], [2.5], [3], [-1], [-0.5], [0], [0.5], [1]];
        const y = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
        const clf = new SVC({ kernel: 'poly', degree: 2, coef0: 1, C: 10 });
        clf.fit(X, y);
        expect(clf.score(X, y)).toBe(1);
        expect(clf.predict([[-2.8], [0.2], [2.8]])).toEqual([1, 0, 1]);
    });

    test('one-vs-one multiclass separates three blobs', () => {
        const X: number[][] = [];
        const Y: number[] = [];
        const centers = [[0, 0], [5, 5], [0, 5]];
        for (let c = 0; c < 3; c++) {
            for (let i = 0; i < 6; i++) {
                X.push([
                    centers[c][0] + 0.3 * Math.sin(i * 2.3 + c),
                    centers[c][1] + 0.3 * Math.cos(i * 1.7 + c),
                ]);
                Y.push(c);
            }
        }
        const clf = new SVC({ kernel: 'rbf' });
        clf.fit(X, Y);
        expect(clf.score(X, Y)).toBe(1);
        expect(clf.predict([[0.1, 0.1], [5.1, 4.9], [-0.2, 5.2]])).toEqual([0, 1, 2]);
    });

    test('binary decisionFunction is positive exactly for classes[1] predictions', () => {
        const clf = new SVC({ kernel: 'rbf', C: 10 });
        clf.fit(xorX, xorY);
        const dec = clf.decisionFunction(xorX);
        const pred = clf.predict(xorX);
        dec.forEach((d, i) => expect(d > 0 ? 1 : 0).toBe(pred[i]));
    });

    test('exposes support vectors after fit', () => {
        const clf = new SVC({ kernel: 'rbf', C: 10 });
        clf.fit(xorX, xorY);
        const sv = clf.getSupportVectors();
        expect(sv.length).toBeGreaterThan(0);
        expect(sv.length).toBeLessThanOrEqual(xorX.length);
        for (let i = 1; i < sv.length; i++) {
            expect(sv[i]).toBeGreaterThan(sv[i - 1]);
        }
        const nSupport = clf.getNSupport();
        expect(nSupport).toHaveLength(2);
        expect(nSupport[0] + nSupport[1]).toBe(sv.length);
    });

    test('predict before fit throws', () => {
        expect(() => new SVC().predict([[0, 0]])).toThrow();
    });
});
