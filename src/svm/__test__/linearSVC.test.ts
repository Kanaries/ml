import { LinearSVC } from '../linearSVC';
import { LinearSVR } from '../linearSVR';
import { SVC } from '../svc';

describe('LinearSVC (Pegasos)', () => {
    test('separates perfectly separable data whose boundary is far from the origin', () => {
        // audit failure case: the old per-sample full regularization clamped
        // ||w|| so hard that this trivially separable 1-D problem sat at 50%
        const X: number[][] = [];
        const Y: number[] = [];
        for (let i = 0; i < 9; i++) {
            X.push([8 + i * 0.2]);
            Y.push(0);
        }
        for (let i = 0; i < 9; i++) {
            X.push([10.4 + i * 0.2]);
            Y.push(1);
        }
        const clf = new LinearSVC({ C: 10, maxIter: 200, randomState: 1 });
        clf.fit(X, Y);
        const acc = clf.predict(X).filter((p, i) => p === Y[i]).length / Y.length;
        expect(acc).toBe(1);
    });

    test('is reproducible with randomState', () => {
        const X = Array.from({ length: 30 }, (_, i) => [Math.sin(i) * 5, Math.cos(i * 1.3) * 5]);
        const Y = X.map(r => (r[0] + r[1] > 0 ? 1 : 0));
        const a = new LinearSVC({ randomState: 7, maxIter: 50 });
        const b = new LinearSVC({ randomState: 7, maxIter: 50 });
        a.fit(X, Y);
        b.fit(X, Y);
        expect(a.predict(X)).toEqual(b.predict(X));
    });

    test('multiclass one-vs-rest still works', () => {
        const X = [
            [0, 0], [0.3, 0.2], [0.1, 0.4],
            [5, 5], [5.2, 4.8], [4.9, 5.3],
            [0, 5], [0.2, 5.1], [-0.1, 4.9],
        ];
        const Y = [0, 0, 0, 1, 1, 1, 2, 2, 2];
        const clf = new LinearSVC({ C: 10, maxIter: 300, randomState: 3 });
        clf.fit(X, Y);
        const acc = clf.predict(X).filter((p, i) => p === Y[i]).length / Y.length;
        expect(acc).toBe(1);
    });
});

describe('LinearSVR (Pegasos)', () => {
    test('fits a steep slope without the regularization-clamp underfit', () => {
        // audit failure case: y = 100x learned w ~ 9.8 and predicted 188 at x=10
        const X = Array.from({ length: 20 }, (_, i) => [i]);
        const Y = X.map(r => 100 * r[0]);
        const reg = new LinearSVR({ C: 100, maxIter: 2000, randomState: 1 });
        reg.fit(X, Y);
        const pred = reg.predict([[10]])[0];
        expect(Math.abs(pred - 1000)).toBeLessThan(30);
    });
});

describe('SVC hardening', () => {
    test('fit copies the training data: mutating the caller array does not move predictions', () => {
        const X = [
            [0, 0], [0.2, 0.1], [0.1, 0.3], [1, 1], [0.9, 0.9], [1.1, 0.8],
        ];
        const Y = [0, 0, 0, 1, 1, 1];
        const svc = new SVC({ kernel: 'rbf' });
        svc.fit(X, Y);
        const before = svc.predict([[0.9, 0.9], [0.1, 0.1]]);
        for (const row of X) {
            row[0] += 100;
            row[1] -= 100;
        }
        const after = svc.predict([[0.9, 0.9], [0.1, 0.1]]);
        expect(after).toEqual(before);
    });

    test("default gamma is 'scale', so tiny-scale features remain separable", () => {
        // with the old fixed gamma=1, a 0.01-scale dataset gives an all-ones
        // gram matrix (no information); gamma='scale' adapts to the variance
        const X: number[][] = [];
        const Y: number[] = [];
        for (let i = 0; i < 10; i++) {
            X.push([i * 0.001, (i * 7 % 10) * 0.001]);
            Y.push(0);
        }
        for (let i = 0; i < 10; i++) {
            X.push([0.05 + i * 0.001, 0.05 + (i * 3 % 10) * 0.001]);
            Y.push(1);
        }
        const svc = new SVC({ kernel: 'rbf', C: 10 });
        svc.fit(X, Y);
        const acc = svc.predict(X).filter((p, i) => p === Y[i]).length / Y.length;
        expect(acc).toBeGreaterThan(0.9);
    });
});
