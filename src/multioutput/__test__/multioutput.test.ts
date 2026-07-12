import { ClassifierBase } from '../../base';
import { Params, getRegisteredEstimators, loadModel, registerEstimator } from '../../base/estimator';
import { makeBlobs } from '../../datasets';
import { RidgeRegression } from '../../linear/ridgeRegression';
import { DecisionTreeClassifier } from '../../tree/decisionTreeClassifier';
import { accuracyScore, r2Score } from '../../metrics';
import { MultiOutputClassifier } from '../multiOutputClassifier';
import { MultiOutputRegressor } from '../multiOutputRegressor';

// ---------------------------------------------------------------------------
// deterministic test-local stub with predictProba
// ---------------------------------------------------------------------------

/** Predicts the majority class; predictProba returns the training priors. */
class PriorStub extends ClassifierBase {
    private classes_: number[];
    private priors_: number[];

    constructor(props: Params = {}) {
        super();
        void props;
        this.classes_ = [];
        this.priors_ = [];
    }

    public getParams(): Params {
        return {};
    }

    public fit(X: number[][], y: number[]): void {
        this.classes_ = Array.from(new Set(y)).sort((a, b) => a - b);
        this.priors_ = this.classes_.map((c) => y.filter((v) => v === c).length / y.length);
    }

    public predict(X: number[][]): number[] {
        const best = this.priors_.indexOf(Math.max(...this.priors_));
        return X.map(() => this.classes_[best]);
    }

    public predictProba(X: number[][]): number[][] {
        return X.map(() => this.priors_.slice());
    }
}
registerEstimator('test.PriorStub', PriorStub);

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/** 40 single-feature samples with exact linear targets Y = [2x, -x + 1]. */
function linearData(): { X: number[][]; Y: number[][] } {
    const X: number[][] = [];
    const Y: number[][] = [];
    for (let i = 0; i < 40; i++) {
        const x = -5 + (10 * i) / 39;
        X.push([x]);
        Y.push([2 * x, -x + 1]);
    }
    return { X, Y };
}

/** Two separated blobs; targets are the label and its complement. */
function blobData(): { X: number[][]; Y: number[][] } {
    const { X, y } = makeBlobs({
        nSamples: 40,
        centers: [
            [0, 0],
            [6, 6],
        ],
        clusterStd: 1,
        randomState: 11,
    });
    return { X, Y: y.map((label) => [label, 1 - label]) };
}

describe('MultiOutputRegressor', () => {
    it('learns Y = [2x, -x+1] with R² > 0.99 on both columns', () => {
        const { X, Y } = linearData();
        const mor = new MultiOutputRegressor({ estimator: new RidgeRegression({ alpha: 0.01 }) });
        mor.fit(X, Y);
        const pred = mor.predict(X);
        expect(pred.length).toBe(X.length);
        expect(pred[0].length).toBe(2);
        for (let j = 0; j < 2; j++) {
            const r2 = r2Score(pred.map((row) => row[j]), Y.map((row) => row[j]));
            expect(r2).toBeGreaterThan(0.99);
        }
        // score() is the uniform average of the per-output R²
        const perOutput = [0, 1].map((j) => r2Score(pred.map((row) => row[j]), Y.map((row) => row[j])));
        expect(mor.score(X, Y)).toBeCloseTo((perOutput[0] + perOutput[1]) / 2, 12);
        expect(mor.score(X, Y)).toBeGreaterThan(0.99);
    });

    it('fits one independent member per output column', () => {
        const { X, Y } = linearData();
        const mor = new MultiOutputRegressor({ estimator: new RidgeRegression({ alpha: 0.01 }) });
        mor.fit(X, Y);
        expect(mor.estimators.length).toBe(2);
        expect(mor.estimators[0]).not.toBe(mor.estimators[1]);
        // members equal a Ridge fitted directly on the corresponding column
        const direct = new RidgeRegression({ alpha: 0.01 });
        direct.fit(X, Y.map((row) => row[0]));
        expect(mor.estimators[0].predict(X)).toEqual(direct.predict(X));
    });

    it('validates the target matrix', () => {
        const { X, Y } = linearData();
        const mor = new MultiOutputRegressor({ estimator: new RidgeRegression() });
        expect(() => mor.fit(X, Y.slice(0, 10))).toThrow(/same length/);
        expect(() => mor.fit(X, [[1, 2], ...Y.slice(1, 39), [3]])).toThrow(/same number of outputs/);
        expect(() => mor.fit(X, [[], ...Y.slice(1)])).toThrow(/at least one output/);
        expect(() => mor.fit([], [])).toThrow(/non-empty/);
        expect(() => mor.predict(X)).toThrow(/fitted/);
    });

    it('survives serialize → revive → predict with identical output', () => {
        const { X, Y } = linearData();
        const mor = new MultiOutputRegressor({ estimator: new RidgeRegression({ alpha: 0.01 }) });
        mor.fit(X, Y);
        const revived = loadModel(JSON.stringify(mor)) as MultiOutputRegressor;
        expect(revived).toBeInstanceOf(MultiOutputRegressor);
        expect(revived.predict(X)).toEqual(mor.predict(X));
        expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(mor)));
    });
});

describe('MultiOutputClassifier', () => {
    const makeDT = () => new DecisionTreeClassifier({ randomState: 7 });

    it('learns two correlated binary targets with high per-output accuracy', () => {
        const { X, Y } = blobData();
        const moc = new MultiOutputClassifier({ estimator: makeDT() });
        moc.fit(X, Y);
        const pred = moc.predict(X);
        expect(pred.length).toBe(X.length);
        expect(pred[0].length).toBe(2);
        for (let j = 0; j < 2; j++) {
            const acc = accuracyScore(pred.map((row) => row[j]), Y.map((row) => row[j]));
            expect(acc).toBeGreaterThan(0.95);
        }
    });

    it('score() is exact-match subset accuracy, matching a hand computation', () => {
        const { X, Y } = blobData();
        const moc = new MultiOutputClassifier({ estimator: makeDT() });
        moc.fit(X, Y);
        // the tree fits the training data perfectly
        expect(moc.score(X, Y)).toBe(1);

        // flip output 0 in rows 0 and 2, output 1 in rows 1 and 2:
        // rows 0, 1, 2 no longer match exactly ⇒ subset accuracy (n-3)/n,
        // although each single output is wrong in only 2 of n rows
        const n = X.length;
        const Ymod = Y.map((row) => row.slice());
        Ymod[0][0] = 1 - Ymod[0][0];
        Ymod[1][1] = 1 - Ymod[1][1];
        Ymod[2][0] = 1 - Ymod[2][0];
        Ymod[2][1] = 1 - Ymod[2][1];
        expect(moc.score(X, Ymod)).toBeCloseTo((n - 3) / n, 12);
        const pred = moc.predict(X);
        for (let j = 0; j < 2; j++) {
            const acc = accuracyScore(pred.map((row) => row[j]), Ymod.map((row) => row[j]));
            expect(acc).toBeCloseTo((n - 2) / n, 12);
        }
    });

    it('predictProba has shape [nOutputs][nSamples][nClasses per output]', () => {
        const { X } = blobData();
        // output 0 is binary, output 1 has three classes
        const Y = X.map((_, i) => [i % 2, i % 3]);
        const moc = new MultiOutputClassifier({ estimator: new PriorStub() });
        moc.fit(X, Y);
        const proba = moc.predictProba(X);
        expect(proba.length).toBe(2);
        expect(proba[0].length).toBe(X.length);
        expect(proba[1].length).toBe(X.length);
        expect(proba[0][0].length).toBe(2);
        expect(proba[1][0].length).toBe(3);
        for (const perOutput of proba) {
            for (const row of perOutput) {
                expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
            }
        }
    });

    it('predictProba throws when the members lack it', () => {
        const { X, Y } = blobData();
        const moc = new MultiOutputClassifier({ estimator: makeDT() });
        moc.fit(X, Y);
        expect(() => moc.predictProba(X)).toThrow(/predictProba/);
    });

    it('validates the target matrix', () => {
        const { X, Y } = blobData();
        const moc = new MultiOutputClassifier({ estimator: makeDT() });
        expect(() => moc.fit(X.slice(0, 5), Y)).toThrow(/same length/);
        expect(() => moc.fit(X, [[0], ...Y.slice(1)])).toThrow(/same number of outputs/);
        expect(() => moc.fit(X, Y.map((row) => row[0]) as unknown as number[][])).toThrow(/2-D array/);
        expect(() => moc.predict(X)).toThrow(/fitted/);
    });

    it('survives serialize → revive → predict with identical output', () => {
        const { X, Y } = blobData();
        const moc = new MultiOutputClassifier({ estimator: makeDT() });
        moc.fit(X, Y);
        const revived = loadModel(JSON.stringify(moc)) as MultiOutputClassifier;
        expect(revived).toBeInstanceOf(MultiOutputClassifier);
        expect(revived.predict(X)).toEqual(moc.predict(X));
        expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(moc)));
    });
});

// ---------------------------------------------------------------------------
// hand-rolled conformance (the shared harness expects 1-D y; these wrappers
// take Y: number[][], so the same contract points are covered here directly)
// ---------------------------------------------------------------------------

type MultiOutputWrapper = MultiOutputClassifier | MultiOutputRegressor;

interface ConformanceSpec {
    name: string;
    create: () => MultiOutputWrapper;
    data: { X: number[][]; Y: number[][] };
}

const conformanceSpecs: ConformanceSpec[] = [
    {
        name: 'MultiOutputClassifier',
        create: () => new MultiOutputClassifier({ estimator: new DecisionTreeClassifier({ randomState: 3 }) }),
        data: blobData(),
    },
    {
        name: 'MultiOutputRegressor',
        create: () => new MultiOutputRegressor({ estimator: new RidgeRegression({ alpha: 0.01 }) }),
        data: linearData(),
    },
];

for (const spec of conformanceSpecs) {
    describe(`${spec.name} conformance`, () => {
        const { X, Y } = spec.data;

        it('is registered under its declared name', () => {
            const est = spec.create();
            expect(getRegisteredEstimators().get(spec.name)).toBe(est.constructor);
        });

        it('getParams/setParams round-trips and validates keys', () => {
            const est = spec.create();
            const params = est.getParams();
            est.setParams(params);
            expect(est.getParams()).toEqual(params);
            expect(() => est.setParams({ __definitely_not_a_param__: 1 })).toThrow(/Invalid parameter/);
        });

        it('routes estimator__param to the prototype estimator', () => {
            const est = spec.create();
            const nested = est.getParams().estimator as { getParams(): Params };
            const before = nested.getParams();
            const key = Object.keys(before)[0];
            est.setParams({ [`estimator__${key}`]: before[key] });
            expect((est.getParams().estimator as { getParams(): Params }).getParams()).toEqual(before);
            expect(() => est.setParams({ estimator__not_a_real_param: 1 })).toThrow(/Invalid parameter/);
            expect(() => est.setParams({ wrong__prefix: 1 })).toThrow(/Invalid parameter/);
        });

        it('clone copies params onto a fresh instance without sharing the nested estimator', () => {
            const est = spec.create();
            const copy = est.clone();
            expect(copy).not.toBe(est);
            expect(copy.constructor).toBe(est.constructor);
            expect(copy.getParams()).toEqual(est.getParams());
            // deep-cloned prototype estimator: equal params, different instance
            expect(copy.getParams().estimator).not.toBe(est.getParams().estimator);
            // fitted state is not shared either: a clone of a fitted model stays unfitted
            est.fit(X, Y);
            const freshCopy = est.clone();
            expect(() => freshCopy.predict(X)).toThrow(/fitted/);
        });

        it('refitting a clone reproduces identical output', () => {
            const est = spec.create();
            est.fit(X, Y);
            const copy = est.clone();
            copy.fit(X, Y);
            expect(copy.predict(X)).toEqual(est.predict(X));
        });

        it('survives serialize → JSON text → revive with identical state and behavior', () => {
            const est = spec.create();
            est.fit(X, Y);
            const revived = loadModel(JSON.stringify(est)) as MultiOutputWrapper;
            expect(revived.constructor).toBe(est.constructor);
            expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(est)));
            expect(revived.predict(X)).toEqual(est.predict(X));
        });

        it('setParams after fit resets to a working unfitted estimator', () => {
            const est = spec.create();
            est.fit(X, Y);
            est.setParams(est.getParams());
            expect(() => est.predict(X)).toThrow(/fitted/);
            est.fit(X, Y);
            expect(est.predict(X).length).toBe(X.length);
        });
    });
}
