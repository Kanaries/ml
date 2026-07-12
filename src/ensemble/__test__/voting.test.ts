import { Params, loadModel, registerEstimator } from '../../base/estimator';
import { ClassifierBase, RegressorBase } from '../../base';
import { VotingClassifier, VotingRegressor } from '../voting';
import { LogisticRegression } from '../../linear/logisticRegression';
import { GaussianNB } from '../../bayes/gaussianNB';
import { GradientBoostingClassifier } from '../gradientBoostingClassifier';
import { AdaBoostClassifier } from '../adaBoostClassifier';
import { binaryDataset } from '../../__test__/conformance/datasets';

/** Stub: always predicts a constant label. No predictProba. */
class FixedLabelClassifier extends ClassifierBase {
    private label: number;
    private fitted = false;
    constructor(props: { label?: number } = {}) {
        super();
        this.label = props.label ?? 0;
    }
    public getParams(): Params {
        return { label: this.label };
    }
    public fit(): void {
        this.fitted = true;
    }
    public predict(X: number[][]): number[] {
        if (!this.fitted) throw new Error('not fitted');
        return X.map(() => this.label);
    }
}
registerEstimator('test.FixedLabelClassifier', FixedLabelClassifier);

/** Stub: returns the same probability row for every sample. */
class FixedProbaClassifier extends ClassifierBase {
    private proba: number[];
    private fitted = false;
    constructor(props: { proba?: number[] } = {}) {
        super();
        this.proba = props.proba ?? [0.5, 0.5];
    }
    public getParams(): Params {
        return { proba: this.proba };
    }
    public fit(): void {
        this.fitted = true;
    }
    public predict(X: number[][]): number[] {
        return this.predictProba(X).map(row => row.indexOf(Math.max(...row)));
    }
    public predictProba(X: number[][]): number[][] {
        if (!this.fitted) throw new Error('not fitted');
        return X.map(() => this.proba.slice());
    }
}
registerEstimator('test.FixedProbaClassifier', FixedProbaClassifier);

/** Stub: always predicts a constant value. */
class FixedValueRegressor extends RegressorBase {
    private value: number;
    private fitted = false;
    constructor(props: { value?: number } = {}) {
        super();
        this.value = props.value ?? 0;
    }
    public getParams(): Params {
        return { value: this.value };
    }
    public fit(): void {
        this.fitted = true;
    }
    public predict(X: number[][]): number[] {
        if (!this.fitted) throw new Error('not fitted');
        return X.map(() => this.value);
    }
}
registerEstimator('test.FixedValueRegressor', FixedValueRegressor);

// tiny dataset that contains both labels so classes_ = [0, 1]
const X4 = [[0], [1], [2], [3]];
const y01 = [0, 0, 1, 1];

describe('VotingClassifier (hard)', () => {
    it('predicts the unweighted majority label', () => {
        const clf = new VotingClassifier({
            estimators: [
                ['a', new FixedLabelClassifier({ label: 0 })],
                ['b', new FixedLabelClassifier({ label: 1 })],
                ['c', new FixedLabelClassifier({ label: 1 })],
            ],
        });
        clf.fit(X4, y01);
        // votes per sample: {0: 1, 1: 2} → 1
        expect(clf.predict(X4)).toEqual([1, 1, 1, 1]);
    });

    it('weights flip the majority (weighted argmax over label counts)', () => {
        const clf = new VotingClassifier({
            estimators: [
                ['a', new FixedLabelClassifier({ label: 0 })],
                ['b', new FixedLabelClassifier({ label: 1 })],
                ['c', new FixedLabelClassifier({ label: 1 })],
            ],
            weights: [3, 1, 1],
        });
        clf.fit(X4, y01);
        // weighted votes: {0: 3, 1: 2} → 0
        expect(clf.predict(X4)).toEqual([0, 0, 0, 0]);
    });

    it('breaks exact ties in favor of the first label in sorted class order', () => {
        // Tie rule (documented in voting.ts): on an exact weighted tie the
        // smallest class label wins — sklearn's np.argmax(np.bincount(...))
        // returns the first (lowest-encoded) class, and classes are sorted
        // ascending, so "first seen" = smallest label.
        const clf = new VotingClassifier({
            estimators: [
                ['a', new FixedLabelClassifier({ label: 5 })],
                ['b', new FixedLabelClassifier({ label: 2 })],
            ],
        });
        clf.fit([[0], [1], [2], [3]], [2, 2, 5, 5]);
        // votes: {2: 1, 5: 1} → tie → smallest label 2
        expect(clf.predict([[0], [1]])).toEqual([2, 2]);

        const weighted = new VotingClassifier({
            estimators: [
                ['a', new FixedLabelClassifier({ label: 5 })],
                ['b', new FixedLabelClassifier({ label: 2 })],
            ],
            weights: [2, 2],
        });
        weighted.fit([[0], [1], [2], [3]], [2, 2, 5, 5]);
        expect(weighted.predict([[0]])).toEqual([2]);
    });

    it('does not expose predictProba', () => {
        const clf = new VotingClassifier({
            estimators: [['a', new FixedLabelClassifier({ label: 0 })]],
        });
        clf.fit(X4, y01);
        expect(() => clf.predictProba(X4)).toThrow(/voting='hard'/);
    });
});

describe('VotingClassifier (soft)', () => {
    it('averages weighted probabilities and takes the argmax', () => {
        const clf = new VotingClassifier({
            estimators: [
                ['a', new FixedProbaClassifier({ proba: [0.9, 0.1] })],
                ['b', new FixedProbaClassifier({ proba: [0.2, 0.8] })],
                ['c', new FixedProbaClassifier({ proba: [0.4, 0.6] })],
            ],
            voting: 'soft',
            weights: [2, 1, 1],
        });
        clf.fit(X4, y01);
        // weighted average: ((2*0.9 + 0.2 + 0.4)/4, (2*0.1 + 0.8 + 0.6)/4) = (0.6, 0.4)
        const proba = clf.predictProba([[0]]);
        expect(proba[0][0]).toBeCloseTo(0.6, 12);
        expect(proba[0][1]).toBeCloseTo(0.4, 12);
        expect(clf.predict(X4)).toEqual([0, 0, 0, 0]);
    });

    it('unweighted average; exact probability tie goes to the smallest class', () => {
        const clf = new VotingClassifier({
            estimators: [
                ['a', new FixedProbaClassifier({ proba: [0.9, 0.1] })],
                ['b', new FixedProbaClassifier({ proba: [0.2, 0.8] })],
                ['c', new FixedProbaClassifier({ proba: [0.4, 0.6] })],
            ],
            voting: 'soft',
        });
        clf.fit(X4, y01);
        // average: (1.5/3, 1.5/3) = (0.5, 0.5) → tie → class 0
        expect(clf.predict([[0]])).toEqual([0]);
    });

    it('throws at fit when a member lacks predictProba', () => {
        const clf = new VotingClassifier({
            estimators: [
                ['proba', new FixedProbaClassifier({ proba: [0.9, 0.1] })],
                ['noproba', new FixedLabelClassifier({ label: 0 })],
            ],
            voting: 'soft',
        });
        expect(() => clf.fit(X4, y01)).toThrow(/"noproba".*predictProba|predictProba.*"noproba"/);
    });
});

describe('VotingRegressor', () => {
    it('predicts the weighted mean of member predictions', () => {
        const reg = new VotingRegressor({
            estimators: [
                ['one', new FixedValueRegressor({ value: 1 })],
                ['three', new FixedValueRegressor({ value: 3 })],
            ],
            weights: [3, 1],
        });
        reg.fit(X4, [0, 0, 0, 0]);
        // (3*1 + 1*3) / 4 = 1.5
        expect(reg.predict([[0], [1]])).toEqual([1.5, 1.5]);
    });

    it('defaults to the unweighted mean', () => {
        const reg = new VotingRegressor({
            estimators: [
                ['one', new FixedValueRegressor({ value: 1 })],
                ['three', new FixedValueRegressor({ value: 3 })],
            ],
        });
        reg.fit(X4, [0, 0, 0, 0]);
        expect(reg.predict([[0]])).toEqual([2]);
    });
});

describe('Voting validation and param addressing', () => {
    it('validates the estimators list and weights', () => {
        expect(() => new VotingClassifier({ estimators: [] })).toThrow(/non-empty/);
        expect(() => new VotingClassifier({
            estimators: [['a', new FixedLabelClassifier()], ['a', new FixedLabelClassifier()]],
        })).toThrow(/Duplicate/);
        expect(() => new VotingClassifier({
            estimators: [['a__b', new FixedLabelClassifier()]],
        })).toThrow(/__/);
        expect(() => new VotingClassifier({
            estimators: [['a', new FixedLabelClassifier()]],
            weights: [1, 2],
        })).toThrow(/weights/);
        expect(() => new VotingClassifier({
            estimators: [['a', new FixedLabelClassifier()]],
            voting: 'loud' as never,
        })).toThrow(/voting/);
    });

    it('supports name__param addressing in setParams', () => {
        const clf = new VotingClassifier({
            estimators: [
                ['lr', new LogisticRegression()],
                ['nb', new GaussianNB()],
            ],
        });
        clf.setParams({ lr__maxIter: 50, nb__varSmoothing: 1e-5 });
        expect(clf.getEstimator('lr').getParams()).toMatchObject({ maxIter: 50 });
        expect(clf.getEstimator('nb').getParams()).toMatchObject({ varSmoothing: 1e-5 });
        expect(() => clf.setParams({ missing__x: 1 })).toThrow(/Unknown estimator/);
        expect(() => clf.setParams({ lr__nope: 1 })).toThrow(/Invalid parameter/);
        expect(() => clf.setParams({ bogus: 1 })).toThrow(/Invalid parameter/);
    });

    it('clone deep-clones the members', () => {
        const clf = new VotingClassifier({
            estimators: [['lr', new LogisticRegression()], ['nb', new GaussianNB()]],
        });
        const copy = clf.clone();
        expect(copy.getEstimator('lr')).not.toBe(clf.getEstimator('lr'));
        expect(copy.getEstimator('lr').getParams()).toEqual(clf.getEstimator('lr').getParams());
    });
});

describe('Voting serialization', () => {
    const { X, y } = binaryDataset();

    it('fitted hard-voting classifier round-trips via loadModel', () => {
        const clf = new VotingClassifier({
            estimators: [
                ['lr', new LogisticRegression({ learningRate: 0.2, maxIter: 200 })],
                ['nb', new GaussianNB()],
            ],
            weights: [1, 2],
        });
        clf.fit(X, y);
        const revived = loadModel(JSON.stringify(clf)) as VotingClassifier;
        expect(revived).toBeInstanceOf(VotingClassifier);
        expect(revived.predict(X)).toEqual(clf.predict(X));
    });

    it('fitted soft-voting classifier round-trips with identical probabilities', () => {
        const clf = new VotingClassifier({
            estimators: [
                ['gbc', new GradientBoostingClassifier({ nEstimators: 5, maxDepth: 2, randomState: 42 })],
                ['ada', new AdaBoostClassifier({ nEstimators: 5 })],
            ],
            voting: 'soft',
        });
        clf.fit(X, y);
        expect(clf.score(X, y)).toBeGreaterThan(0.9);
        const revived = loadModel(JSON.stringify(clf)) as VotingClassifier;
        expect(revived.predict(X)).toEqual(clf.predict(X));
        expect(revived.predictProba(X)).toEqual(clf.predictProba(X));
    });

    it('fitted voting regressor round-trips via loadModel', () => {
        const reg = new VotingRegressor({
            estimators: [
                ['a', new FixedValueRegressor({ value: 2 })],
                ['b', new FixedValueRegressor({ value: 4 })],
            ],
        });
        reg.fit(X4, [0, 0, 0, 0]);
        const revived = loadModel(JSON.stringify(reg)) as VotingRegressor;
        expect(revived.predict([[0]])).toEqual(reg.predict([[0]]));
    });
});
