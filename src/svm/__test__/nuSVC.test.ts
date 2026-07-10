import { NuSVC } from '../nuSVC';

// two concentric rings: needs a non-linear boundary, classes balanced 20/20
const ringX: number[][] = [];
const ringY: number[] = [];
for (let i = 0; i < 20; i++) {
    const t = (i / 20) * 2 * Math.PI;
    ringX.push([0.4 * Math.cos(t), 0.4 * Math.sin(t)]);
    ringY.push(0);
    ringX.push([Math.cos(t), Math.sin(t)]);
    ringY.push(1);
}

describe('NuSVC (SMO, nu-SVM)', () => {
    test('rbf kernel fits XOR-style data with perfect training accuracy', () => {
        const X: number[][] = [];
        const Y: number[] = [];
        for (const [cx, cy, label] of [
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]) {
            for (let i = 0; i < 4; i++) {
                X.push([cx + 0.05 * (i % 2), cy + 0.05 * Math.floor(i / 2)]);
                Y.push(label);
            }
        }
        const clf = new NuSVC({ nu: 0.3 });
        clf.fit(X, Y);
        expect(clf.score(X, Y)).toBe(1);
    });

    test('nu lower-bounds the support-vector fraction: SV count grows with nu', () => {
        const total = (nu: number): number => {
            const clf = new NuSVC({ nu });
            clf.fit(ringX, ringY);
            return clf.getNSupport().reduce((s, v) => s + v, 0);
        };
        const low = total(0.2);
        const high = total(0.8);
        expect(high).toBeGreaterThan(low);
        // nu-SVM property: #SV / n >= nu
        expect(low).toBeGreaterThanOrEqual(Math.floor(0.2 * ringX.length));
        expect(high).toBeGreaterThanOrEqual(Math.floor(0.8 * ringX.length));
    });

    test("infeasible nu (larger than 2*min(n+,n-)/n) throws sklearn's message", () => {
        const X = [[0, 0], [0.1, 0], [0.2, 0], [1, 1], [1.1, 1], [1.2, 1]];
        const y = [0, 0, 0, 0, 0, 1];
        const clf = new NuSVC({ nu: 0.5 });
        expect(() => clf.fit(X, y)).toThrow('specified nu is infeasible');
    });

    test('nu outside (0, 1] is rejected', () => {
        expect(() => new NuSVC({ nu: 0 })).toThrow();
        expect(() => new NuSVC({ nu: -0.1 })).toThrow();
        expect(() => new NuSVC({ nu: 1.5 })).toThrow();
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
        const clf = new NuSVC({ nu: 0.3 });
        clf.fit(X, Y);
        expect(clf.score(X, Y)).toBe(1);
        expect(clf.predict([[0.1, 0.1], [5.1, 4.9], [-0.2, 5.2]])).toEqual([0, 1, 2]);
    });

    test('fit copies the training data: mutating the caller array does not move predictions', () => {
        const X = ringX.map(row => row.slice());
        const clf = new NuSVC({ nu: 0.4 });
        clf.fit(X, ringY);
        const probe = [[0.1, 0.1], [0.9, 0.4]];
        const before = clf.predict(probe);
        for (const row of X) {
            row[0] += 100;
            row[1] -= 100;
        }
        expect(clf.predict(probe)).toEqual(before);
    });
});
