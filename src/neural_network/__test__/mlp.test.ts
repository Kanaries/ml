import { MLPClassifier } from '../mlpClassifier';
import { MLPRegressor } from '../mlpRegressor';

/** Deterministic LCG so the datasets are identical on every platform. */
function lcg(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

/** Three well-separated 2-D blobs with deterministically shuffled rows. */
function blobs3(nPerClass: number, seed: number): { X: number[][]; y: number[] } {
    const rand = lcg(seed);
    const centers = [[0, 0], [6, 6], [0, 8]];
    const X: number[][] = [];
    const y: number[] = [];
    centers.forEach((c, label) => {
        for (let i = 0; i < nPerClass; i++) {
            X.push(c.map(v => v + (rand() * 2 - 1)));
            y.push(label);
        }
    });
    // shuffle so tail-split early stopping sees all classes
    for (let i = X.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [X[i], X[j]] = [X[j], X[i]];
        [y[i], y[j]] = [y[j], y[i]];
    }
    return { X, y };
}

const XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const XOR_Y = [0, 1, 1, 0];

describe('MLPClassifier', () => {
    it('learns XOR to 100% accuracy with adam (seeded)', () => {
        const clf = new MLPClassifier({
            hiddenLayerSizes: [8],
            solver: 'adam',
            learningRateInit: 0.1,
            maxIter: 500,
            tol: 1e-7,
            nIterNoChange: 50,
            randomState: 0,
        });
        clf.fit(XOR_X, XOR_Y);
        expect(clf.predict(XOR_X)).toEqual(XOR_Y);
        expect(clf.score(XOR_X, XOR_Y)).toBe(1);
        expect(clf.classes).toEqual([0, 1]);
    });

    it('binary predictProba has two columns that sum to 1', () => {
        const clf = new MLPClassifier({
            hiddenLayerSizes: [8],
            learningRateInit: 0.1,
            maxIter: 500,
            tol: 1e-7,
            nIterNoChange: 50,
            randomState: 0,
        });
        clf.fit(XOR_X, XOR_Y);
        const proba = clf.predictProba(XOR_X);
        expect(proba.length).toBe(4);
        for (const row of proba) {
            expect(row.length).toBe(2);
            expect(row[0] + row[1]).toBeCloseTo(1, 10);
        }
    });

    it('reaches > 0.95 accuracy on 3-class blobs; proba rows sum to 1 and argmax matches predict', () => {
        const { X, y } = blobs3(50, 7);
        const clf = new MLPClassifier({ hiddenLayerSizes: [16], maxIter: 300, randomState: 0 });
        clf.fit(X, y);
        expect(clf.classes).toEqual([0, 1, 2]);
        expect(clf.score(X, y)).toBeGreaterThan(0.95);

        const proba = clf.predictProba(X);
        const preds = clf.predict(X);
        expect(proba.length).toBe(X.length);
        for (let i = 0; i < proba.length; i++) {
            expect(proba[i].length).toBe(3);
            expect(proba[i].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
            let best = 0;
            for (let k = 1; k < 3; k++) {
                if (proba[i][k] > proba[i][best]) best = k;
            }
            expect(clf.classes[best]).toBe(preds[i]);
        }
    });

    it('records a decreasing loss curve', () => {
        const { X, y } = blobs3(30, 11);
        const clf = new MLPClassifier({ hiddenLayerSizes: [8], maxIter: 100, randomState: 3 });
        clf.fit(X, y);
        const curve = clf.lossCurve;
        expect(curve.length).toBeGreaterThan(1);
        expect(curve.length).toBe(clf.nIter);
        expect(curve[curve.length - 1]).toBeLessThan(curve[0]);
    });

    it('is deterministic under a fixed randomState', () => {
        const { X, y } = blobs3(30, 13);
        const make = () => new MLPClassifier({ hiddenLayerSizes: [8], maxIter: 100, randomState: 5 });
        const a = make();
        const b = make();
        a.fit(X, y);
        b.fit(X, y);
        expect(b.predict(X)).toEqual(a.predict(X));
        expect(b.predictProba(X)).toEqual(a.predictProba(X));
        expect(b.lossCurve).toEqual(a.lossCurve);
    });

    it('trains with the sgd solver (nesterov momentum, constant schedule)', () => {
        const { X, y } = blobs3(40, 17);
        const clf = new MLPClassifier({
            hiddenLayerSizes: [16],
            solver: 'sgd',
            learningRateInit: 0.05,
            maxIter: 300,
            randomState: 1,
        });
        clf.fit(X, y);
        expect(clf.score(X, y)).toBeGreaterThan(0.95);
    });

    it('sgd adaptive schedule keeps reducing the rate and stops before maxIter', () => {
        const { X, y } = blobs3(40, 19);
        const clf = new MLPClassifier({
            hiddenLayerSizes: [16],
            solver: 'sgd',
            learningRate: 'adaptive',
            learningRateInit: 0.05,
            tol: 1e-3,
            nIterNoChange: 5,
            maxIter: 1000,
            randomState: 1,
        });
        clf.fit(X, y);
        expect(clf.score(X, y)).toBeGreaterThan(0.95);
        // adaptive: lr is divided by 5 on each plateau and training stops
        // once lr <= 1e-6, well before the generous maxIter
        expect(clf.nIter).toBeLessThan(1000);
    });

    it('sgd invscaling schedule trains successfully', () => {
        const { X, y } = blobs3(40, 23);
        const clf = new MLPClassifier({
            hiddenLayerSizes: [16],
            solver: 'sgd',
            learningRate: 'invscaling',
            learningRateInit: 0.5,
            maxIter: 300,
            randomState: 1,
        });
        clf.fit(X, y);
        expect(clf.score(X, y)).toBeGreaterThan(0.9);
    });

    it('early stopping halts before maxIter on easy data and still predicts well', () => {
        const { X, y } = blobs3(50, 29);
        const clf = new MLPClassifier({
            hiddenLayerSizes: [16],
            learningRateInit: 0.05,
            earlyStopping: true,
            validationFraction: 0.2,
            nIterNoChange: 5,
            maxIter: 500,
            randomState: 4,
        });
        clf.fit(X, y);
        expect(clf.nIter).toBeLessThan(500);
        expect(clf.score(X, y)).toBeGreaterThan(0.95);
    });

    it('throws when predicting before fit and on single-class y', () => {
        const clf = new MLPClassifier({ randomState: 0 });
        expect(() => clf.predict([[0, 0]])).toThrow(/not fitted/);
        expect(() => clf.fit([[0], [1]], [1, 1])).toThrow(/at least 2 classes/);
    });
});

describe('MLPRegressor', () => {
    function linearData(n: number, seed: number): { X: number[][]; y: number[] } {
        const rand = lcg(seed);
        const X: number[][] = [];
        const y: number[] = [];
        for (let i = 0; i < n; i++) {
            const x1 = rand() * 4 - 2;
            const x2 = rand() * 4 - 2;
            X.push([x1, x2]);
            y.push(2 * x1 - x2);
        }
        return { X, y };
    }

    it('learns y = 2*x1 - x2 to R² > 0.95 with single-output shape', () => {
        const { X, y } = linearData(200, 31);
        const reg = new MLPRegressor({
            hiddenLayerSizes: [16],
            learningRateInit: 0.01,
            maxIter: 500,
            tol: 1e-6,
            randomState: 42,
        });
        reg.fit(X, y);
        const preds = reg.predict(X);
        expect(preds.length).toBe(X.length);
        expect(typeof preds[0]).toBe('number');
        expect(reg.score(X, y)).toBeGreaterThan(0.95);
    });

    it('records a decreasing loss curve and is deterministic when seeded', () => {
        const { X, y } = linearData(100, 37);
        const make = () => new MLPRegressor({ hiddenLayerSizes: [8], maxIter: 100, randomState: 7 });
        const a = make();
        const b = make();
        a.fit(X, y);
        b.fit(X, y);
        const curve = a.lossCurve;
        expect(curve.length).toBeGreaterThan(1);
        expect(curve[curve.length - 1]).toBeLessThan(curve[0]);
        expect(b.predict(X)).toEqual(a.predict(X));
        expect(b.lossCurve).toEqual(a.lossCurve);
    });

    it('supports the sgd solver', () => {
        const { X, y } = linearData(150, 41);
        const reg = new MLPRegressor({
            hiddenLayerSizes: [16],
            solver: 'sgd',
            learningRateInit: 0.01,
            maxIter: 500,
            tol: 1e-6,
            randomState: 3,
        });
        reg.fit(X, y);
        expect(reg.score(X, y)).toBeGreaterThan(0.95);
    });

    it('early stopping halts before maxIter on easy data', () => {
        const { X, y } = linearData(200, 43);
        const reg = new MLPRegressor({
            hiddenLayerSizes: [16],
            learningRateInit: 0.01,
            earlyStopping: true,
            validationFraction: 0.2,
            nIterNoChange: 5,
            maxIter: 2000,
            randomState: 5,
        });
        reg.fit(X, y);
        expect(reg.nIter).toBeLessThan(2000);
        expect(reg.score(X, y)).toBeGreaterThan(0.9);
    });

    it('throws when predicting before fit', () => {
        const reg = new MLPRegressor({ randomState: 0 });
        expect(() => reg.predict([[0, 0]])).toThrow(/not fitted/);
    });
});
