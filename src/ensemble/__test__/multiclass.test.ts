import { GradientBoostingClassifier } from '../gradientBoostingClassifier';
import { XGBoostClassifier } from '../xgboost';

// three well-separated 2-D clusters
function threeClusters() {
    const X: number[][] = [];
    const y: number[] = [];
    const centers: Array<[number, number, number]> = [
        [0, 0, 10],
        [8, 0, 20],
        [4, 7, 30],
    ];
    for (const [cx, cy, label] of centers) {
        for (let i = 0; i < 12; i++) {
            // deterministic jitter in [-1, 1]
            const dx = ((i * 7) % 12) / 6 - 1;
            const dy = ((i * 5) % 12) / 6 - 1;
            X.push([cx + dx, cy + dy]);
            y.push(label);
        }
    }
    return { X, y };
}

describe.each([
    ['GradientBoostingClassifier', (props: any) => new GradientBoostingClassifier(props)],
    ['XGBoostClassifier', (props: any) => new XGBoostClassifier(props)],
])('%s multiclass', (_name, make) => {
    test('separates three clusters with arbitrary labels', () => {
        const { X, y } = threeClusters();
        const clf = make({ nEstimators: 30, randomState: 0 });
        clf.fit(X, y);
        expect(clf.predict(X)).toEqual(y);
        // held-out points near each center
        expect(clf.predict([[0, 0.5], [8, -0.5], [4, 6.5]])).toEqual([10, 20, 30]);
    });

    test('predictProba has K columns ordered by sorted labels, rows sum to 1', () => {
        const { X, y } = threeClusters();
        const clf = make({ nEstimators: 30, randomState: 0 });
        clf.fit(X, y);
        const proba = clf.predictProba(X);
        for (let i = 0; i < X.length; i++) {
            expect(proba[i]).toHaveLength(3);
            expect(proba[i][0] + proba[i][1] + proba[i][2]).toBeCloseTo(1, 9);
            const trueCol = [10, 20, 30].indexOf(y[i]);
            expect(proba[i][trueCol]).toBeGreaterThan(0.5);
        }
    });

    test('multiclass fit is reproducible with randomState under subsampling', () => {
        const { X, y } = threeClusters();
        const a = make({ nEstimators: 15, subsample: 0.7, randomState: 3 });
        const b = make({ nEstimators: 15, subsample: 0.7, randomState: 3 });
        a.fit(X, y);
        b.fit(X, y);
        expect(a.predict(X)).toEqual(b.predict(X));
    });

    test('a failed refit does not corrupt a fitted multiclass model', () => {
        const { X, y } = threeClusters();
        const clf = make({ nEstimators: 15, randomState: 0 });
        clf.fit(X, y);
        expect(() => clf.fit([[1], [2]], [1, 1])).toThrow();
        expect(clf.predict([[0, 0.5], [8, -0.5], [4, 6.5]])).toEqual([10, 20, 30]);
    });

    test('binary fit after a multiclass fit works (state fully reset)', () => {
        const { X, y } = threeClusters();
        const clf = make({ nEstimators: 15, randomState: 0 });
        clf.fit(X, y);
        const Xb = [[1], [2], [3], [4], [6], [7], [8], [9]];
        const yb = [0, 0, 0, 0, 1, 1, 1, 1];
        clf.fit(Xb, yb);
        expect(clf.predict(Xb)).toEqual(yb);
        expect(clf.predictProba([[1]])[0]).toHaveLength(2);
    });
});
