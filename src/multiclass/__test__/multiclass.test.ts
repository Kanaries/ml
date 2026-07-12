import { ClassifierBase } from '../../base';
import { Params, loadModel, registerEstimator } from '../../base/estimator';
import { makeBlobs } from '../../datasets';
import { LogisticRegression } from '../../linear/logisticRegression';
import { DecisionTreeClassifier } from '../../tree/decisionTreeClassifier';
import { accuracyScore } from '../../metrics';
import { OneVsRestClassifier } from '../oneVsRest';
import { OneVsOneClassifier, ovrDecisionFunction } from '../oneVsOne';

// ---------------------------------------------------------------------------
// deterministic test-local stubs
// ---------------------------------------------------------------------------

/**
 * Nearest class-mean binary classifier trained on labels {0, 1}.
 * decisionFunction = dist(x, mean0) - dist(x, mean1): positive favors 1.
 */
class NearestMeanStub extends ClassifierBase {
    private means: number[][];

    constructor(props: Params = {}) {
        super();
        void props;
        this.means = [];
    }

    public getParams(): Params {
        return {};
    }

    public fit(X: number[][], y: number[]): void {
        const meanOf = (label: number): number[] => {
            const rows = X.filter((_, i) => y[i] === label);
            const m = new Array(X[0].length).fill(0);
            for (const r of rows) {
                r.forEach((v, j) => {
                    m[j] += v / rows.length;
                });
            }
            return m;
        };
        this.means = [meanOf(0), meanOf(1)];
    }

    private dist(x: number[], m: number[]): number {
        return Math.sqrt(x.reduce((acc, v, j) => acc + (v - m[j]) ** 2, 0));
    }

    public decisionFunction(X: number[][]): number[] {
        return X.map((x) => this.dist(x, this.means[0]) - this.dist(x, this.means[1]));
    }

    public predict(X: number[][]): number[] {
        return this.decisionFunction(X).map((d) => (d > 0 ? 1 : 0));
    }
}
registerEstimator('test.NearestMeanStub', NearestMeanStub);

/** NearestMeanStub with a sigmoid predictProba on top of the decision value. */
class ProbaStub extends NearestMeanStub {
    public predictProba(X: number[][]): number[][] {
        return this.decisionFunction(X).map((d) => {
            const p1 = 1 / (1 + Math.exp(-d));
            return [1 - p1, p1];
        });
    }
}
registerEstimator('test.ProbaStub', ProbaStub);

/** Predicts label 0 for everything; exposes no confidence source. */
class ConstantZeroStub extends ClassifierBase {
    constructor(props: Params = {}) {
        super();
        void props;
    }

    public getParams(): Params {
        return {};
    }

    public fit(): void {
        /* nothing to learn */
    }

    public predict(X: number[][]): number[] {
        return X.map(() => 0);
    }
}
registerEstimator('test.ConstantZeroStub', ConstantZeroStub);

// ---------------------------------------------------------------------------
// shared fixtures
// ---------------------------------------------------------------------------

const blobs3 = makeBlobs({
    nSamples: 90,
    centers: [
        [0, 0],
        [10, 10],
        [0, 10],
    ],
    clusterStd: 1,
    randomState: 42,
});

/** 3 classes on a line: means 0.5, 10.5, 20.5 — hand-checkable geometry. */
const lineX = [[0], [1], [10], [11], [20], [21]];
const lineY = [0, 0, 1, 1, 2, 2];

const makeLR = () => new LogisticRegression({ learningRate: 0.5, maxIter: 500 });

describe('OneVsRestClassifier', () => {
    it('turns a binary-only classifier into a multiclass classifier', () => {
        // LogisticRegression alone rejects 3-class targets...
        expect(() => makeLR().fit(blobs3.X, blobs3.y)).toThrow(/exactly 2 classes/);

        // ...but wrapped in OvR it learns the 3-class problem
        const ovr = new OneVsRestClassifier({ estimator: makeLR() });
        ovr.fit(blobs3.X, blobs3.y);
        const acc = accuracyScore(ovr.predict(blobs3.X), blobs3.y);
        expect(acc).toBeGreaterThan(0.95);

        // on par with a natively-multiclass learner on the same data
        const dt = new DecisionTreeClassifier({ randomState: 1 });
        dt.fit(blobs3.X, blobs3.y);
        const dtAcc = accuracyScore(dt.predict(blobs3.X), blobs3.y);
        expect(Math.abs(acc - dtAcc)).toBeLessThanOrEqual(0.05);
    });

    it('exposes the sorted class labels', () => {
        const ovr = new OneVsRestClassifier({ estimator: makeLR() });
        ovr.fit(blobs3.X, [2, 0, 1, ...blobs3.y.slice(3)]);
        expect(ovr.classes).toEqual([0, 1, 2]);
    });

    it('predictProba rows are normalized to sum to 1', () => {
        const ovr = new OneVsRestClassifier({ estimator: new ProbaStub() });
        ovr.fit(blobs3.X, blobs3.y);
        const proba = ovr.predictProba(blobs3.X);
        expect(proba.length).toBe(blobs3.X.length);
        for (const row of proba) {
            expect(row.length).toBe(3);
            expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
            for (const p of row) {
                expect(p).toBeGreaterThanOrEqual(0);
                expect(p).toBeLessThanOrEqual(1);
            }
        }
        // predict agrees with the argmax of predictProba
        const pred = ovr.predict(blobs3.X);
        proba.forEach((row, i) => {
            expect(ovr.classes[row.indexOf(Math.max(...row))]).toBe(pred[i]);
        });
    });

    it('predictProba / decisionFunction throw when the members lack them', () => {
        const ovr = new OneVsRestClassifier({ estimator: makeLR() });
        ovr.fit(blobs3.X, blobs3.y);
        expect(() => ovr.predictProba(blobs3.X)).toThrow(/predictProba/);
        expect(() => ovr.decisionFunction(blobs3.X)).toThrow(/decisionFunction/);
    });

    it('decisionFunction returns one column per class', () => {
        const ovr = new OneVsRestClassifier({ estimator: new NearestMeanStub() });
        ovr.fit(lineX, lineY);
        const dec = ovr.decisionFunction([[2], [12]]);
        expect(dec.length).toBe(2);
        expect(dec[0].length).toBe(3);
        // predict agrees with the argmax of decisionFunction
        const pred = ovr.predict([[2], [12]]);
        dec.forEach((row, i) => {
            expect(ovr.classes[row.indexOf(Math.max(...row))]).toBe(pred[i]);
        });
    });

    it('falls back to raw 0/1 predict scores when members expose no proba/decision', () => {
        const ovr = new OneVsRestClassifier({ estimator: makeLR() });
        ovr.fit(blobs3.X, blobs3.y);
        // recompute the expected argmax over the members' raw 0/1 votes
        // (first class wins ties), which is all LogisticRegression offers
        const votes = ovr.estimators.map((m) => m.predict(blobs3.X));
        const expected = blobs3.X.map((_, i) => {
            let best = 0;
            for (let c = 1; c < votes.length; c++) {
                if (votes[c][i] > votes[best][i]) best = c;
            }
            return ovr.classes[best];
        });
        expect(ovr.predict(blobs3.X)).toEqual(expected);
    });

    it('routes estimator__param through setParams to the prototype estimator', () => {
        const ovr = new OneVsRestClassifier({ estimator: makeLR() });
        ovr.setParams({ estimator__maxIter: 7 });
        const nested = ovr.getParams().estimator as LogisticRegression;
        expect(nested.getParams()).toEqual({ learningRate: 0.5, maxIter: 7 });
        expect(() => ovr.setParams({ estimator__nope: 1 })).toThrow(/Invalid parameter/);
        expect(() => ovr.setParams({ other__maxIter: 1 })).toThrow(/Invalid parameter/);
        expect(() => ovr.setParams({ bogus: 1 })).toThrow(/Invalid parameter/);
    });

    it('survives serialize → revive → predict with identical output', () => {
        const ovr = new OneVsRestClassifier({ estimator: makeLR() });
        ovr.fit(blobs3.X, blobs3.y);
        const revived = loadModel(JSON.stringify(ovr)) as OneVsRestClassifier;
        expect(revived).toBeInstanceOf(OneVsRestClassifier);
        expect(revived.classes).toEqual(ovr.classes);
        expect(revived.predict(blobs3.X)).toEqual(ovr.predict(blobs3.X));
    });
});

describe('OneVsOneClassifier', () => {
    it('reaches the same accuracy as OvR with a binary-only member', () => {
        const ovo = new OneVsOneClassifier({ estimator: makeLR() });
        ovo.fit(blobs3.X, blobs3.y);
        expect(accuracyScore(ovo.predict(blobs3.X), blobs3.y)).toBeGreaterThan(0.95);
        expect(ovo.classes).toEqual([0, 1, 2]);
        // one member per unordered class pair
        expect(ovo.estimators.length).toBe(3);
    });

    it('counts pairwise votes exactly as hand-computed (nearest-mean stubs)', () => {
        const ovo = new OneVsOneClassifier({ estimator: new NearestMeanStub() });
        ovo.fit(lineX, lineY);

        // x = 2: pair(0,1) → 0, pair(0,2) → 0, pair(1,2) → 1  ⇒ votes [2,1,0]
        // x = 12: pair(0,1) → 1, pair(0,2) → 2, pair(1,2) → 1 ⇒ votes [0,2,1]
        const dec = ovo.decisionFunction([[2], [12]]) as number[][];
        expect(dec.map((row) => row.map((v) => Math.round(v) + 0))).toEqual([
            [2, 1, 0],
            [0, 2, 1],
        ]);
        expect(ovo.predict([[2], [12]])).toEqual([0, 1]);

        // full hand computation for x = 2 with member confidences
        // d(0,1) = 1.5-8.5 = -7, d(0,2) = 1.5-18.5 = -17, d(1,2) = 8.5-18.5 = -10
        // sums: s0 = +7+17 = 24, s1 = -7+10 = 3, s2 = -17-10 = -27
        // decision = votes + s / (3(|s|+1))
        expect(dec[0][0]).toBeCloseTo(2 + 24 / 75, 12);
        expect(dec[0][1]).toBeCloseTo(1 + 3 / 12, 12);
        expect(dec[0][2]).toBeCloseTo(0 - 27 / 84, 12);
    });

    it('ovrDecisionFunction breaks a cyclic vote tie by summed confidences', () => {
        // one sample, 3 classes, cyclic pairwise outcome: every class gets 1 vote
        // pair(0,1) → class 0 (conf -2), pair(0,2) → class 2 (conf +5),
        // pair(1,2) → class 1 (conf -1)
        const predictions = [[0], [1], [0]];
        const confidences = [[-2], [5], [-1]];
        const dec = ovrDecisionFunction(predictions, confidences, 3);
        // sums: s0 = 2-5 = -3, s1 = -2+1 = -1, s2 = 5-1 = 4
        expect(dec[0][0]).toBeCloseTo(1 - 3 / 12, 12);
        expect(dec[0][1]).toBeCloseTo(1 - 1 / 6, 12);
        expect(dec[0][2]).toBeCloseTo(1 + 4 / 15, 12);
        // the squashed confidence stays inside (-1/3, 1/3): only breaks ties
        for (const v of dec[0]) {
            expect(Math.abs(v - 1)).toBeLessThan(1 / 3);
        }
        // class 2 wins the 1-1-1 tie on confidence
        expect(dec[0].indexOf(Math.max(...dec[0]))).toBe(2);
    });

    it('falls back to first-class-wins on ties when members expose no confidence', () => {
        const ovo = new OneVsOneClassifier({ estimator: new ConstantZeroStub() });
        ovo.fit(lineX, lineY);
        // every member predicts label 0 ⇒ votes [2,1,0] with zero confidences
        expect(ovo.predict([[5]])).toEqual([0]);
        const dec = ovo.decisionFunction([[5]]) as number[][];
        expect(dec).toEqual([[2, 1, 0]]);
    });

    it('returns a flat decisionFunction for binary problems (sklearn behavior)', () => {
        const binary = makeBlobs({
            nSamples: 40,
            centers: [
                [0, 0],
                [8, 8],
            ],
            clusterStd: 1,
            randomState: 7,
        });
        const ovo = new OneVsOneClassifier({ estimator: makeLR() });
        ovo.fit(binary.X, binary.y);
        const dec = ovo.decisionFunction(binary.X) as number[];
        expect(dec.length).toBe(binary.X.length);
        expect(typeof dec[0]).toBe('number');
        expect(accuracyScore(ovo.predict(binary.X), binary.y)).toBeGreaterThan(0.95);
    });

    it('routes estimator__param through setParams to the prototype estimator', () => {
        const ovo = new OneVsOneClassifier({ estimator: makeLR() });
        ovo.setParams({ estimator__learningRate: 0.05 });
        const nested = ovo.getParams().estimator as LogisticRegression;
        expect(nested.getParams()).toEqual({ learningRate: 0.05, maxIter: 500 });
        expect(() => ovo.setParams({ other__x: 1 })).toThrow(/Invalid parameter/);
    });

    it('survives serialize → revive → predict with identical output', () => {
        const ovo = new OneVsOneClassifier({ estimator: makeLR() });
        ovo.fit(blobs3.X, blobs3.y);
        const revived = loadModel(JSON.stringify(ovo)) as OneVsOneClassifier;
        expect(revived).toBeInstanceOf(OneVsOneClassifier);
        expect(revived.classes).toEqual(ovo.classes);
        expect(revived.predict(blobs3.X)).toEqual(ovo.predict(blobs3.X));
    });
});
