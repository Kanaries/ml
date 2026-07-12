import { Params, loadModel, registerEstimator } from '../../base/estimator';
import { ClassifierBase } from '../../base';
import { StackingClassifier, StackingRegressor } from '../stacking';
import { GaussianNB } from '../../bayes/gaussianNB';
import { DecisionTreeClassifier } from '../../tree/decisionTreeClassifier';
import { DecisionTreeRegressor } from '../../tree/decisionTreeRegressor';
import { KNearestNeighbors } from '../../neighbors/knn';
import { GradientBoostingClassifier } from '../gradientBoostingClassifier';
import { LogisticRegression } from '../../linear/logisticRegression';
import { RidgeRegression } from '../../linear/ridgeRegression';
import { accuracyScore } from '../../metrics';
import { makeBlobs } from '../../datasets';
import { binaryDataset, blobsDataset, regressionDataset } from '../../__test__/conformance/datasets';

/**
 * Stub: 1-nearest-neighbor memorizer. In-sample it always returns the exact
 * training label of the queried point (distance 0 to itself), so any
 * meta-feature built from in-sample predictions would leak the labels.
 */
class OneNNClassifier extends ClassifierBase {
    private trainX: number[][] = [];
    private trainY: number[] = [];
    public getParams(): Params {
        return {};
    }
    public fit(X: number[][], y: number[]): void {
        this.trainX = X.map(r => r.slice());
        this.trainY = y.slice();
    }
    public predict(X: number[][]): number[] {
        return X.map(row => {
            let best = 0;
            let bestDist = Infinity;
            for (let i = 0; i < this.trainX.length; i++) {
                let d = 0;
                for (let j = 0; j < row.length; j++) {
                    const diff = row[j] - this.trainX[i][j];
                    d += diff * diff;
                }
                if (d < bestDist) {
                    bestDist = d;
                    best = i;
                }
            }
            return this.trainY[best];
        });
    }
}
registerEstimator('test.OneNNClassifier', OneNNClassifier);

/**
 * Stub final estimator that records the meta-feature matrix it was trained
 * on, so tests can assert the out-of-fold construction directly.
 */
class RecordingClassifier extends ClassifierBase {
    public trainX: number[][] = [];
    public trainY: number[] = [];
    public getParams(): Params {
        return {};
    }
    public fit(X: number[][], y: number[]): void {
        this.trainX = X.map(r => r.slice());
        this.trainY = y.slice();
    }
    public predict(X: number[][]): number[] {
        return X.map(row => row[0]);
    }
}
registerEstimator('test.RecordingClassifier', RecordingClassifier);

describe('StackingClassifier accuracy', () => {
    it('is at least as good as the best base estimator (minus 0.05) on 3-class blobs', () => {
        // NOTE: the task's suggested LogisticRegression cannot be used here —
        // this library's LogisticRegression is binary-only — so the members
        // and final estimator are multiclass-capable classifiers instead.
        const { X, y } = makeBlobs({
            nSamples: 150,
            centers: [[0, 0], [4, 4], [0, 5]],
            clusterStd: 1.6,
            randomState: 42,
        });
        const trainX = X.slice(0, 100);
        const trainY = y.slice(0, 100);
        const testX = X.slice(100);
        const testY = y.slice(100);

        const makeMembers = (): [string, ClassifierBase][] => [
            ['nb', new GaussianNB()],
            ['dt', new DecisionTreeClassifier({ max_depth: 5, randomState: 1 })],
            ['knn', new KNearestNeighbors({ kNeighbors: 5 })],
        ];

        let bestSingle = 0;
        for (const [, single] of makeMembers()) {
            single.fit(trainX, trainY);
            bestSingle = Math.max(bestSingle, single.score(testX, testY));
        }

        const stack = new StackingClassifier({
            estimators: makeMembers(),
            finalEstimator: new GaussianNB(),
            cv: 5,
        });
        stack.fit(trainX, trainY);
        expect(stack.score(testX, testY)).toBeGreaterThanOrEqual(bestSingle - 0.05);
    });
});

describe('StackingClassifier meta-features', () => {
    it('builds nClasses columns per predictProba member and 1 per predict member', () => {
        const { X, y } = blobsDataset(); // 3 classes, 2 features, 36 samples
        const recorder = new RecordingClassifier();
        const stack = new StackingClassifier({
            estimators: [
                // has predictProba → 3 columns on a 3-class problem
                ['gbc', new GradientBoostingClassifier({ nEstimators: 5, maxDepth: 2, randomState: 3 })],
                // predict-only → 1 label column
                ['nn', new OneNNClassifier()],
            ],
            finalEstimator: recorder,
            cv: 3,
        });
        stack.fit(X, y);
        // out-of-fold matrix the final estimator was trained on
        expect(recorder.trainX).toHaveLength(X.length);
        expect(recorder.trainX[0]).toHaveLength(3 + 1);
        // inference-time meta-features have the same shape
        expect(stack.transform(X)[0]).toHaveLength(3 + 1);
    });

    it('passthrough appends the original features after the meta-features', () => {
        const { X, y } = blobsDataset();
        const recorder = new RecordingClassifier();
        const stack = new StackingClassifier({
            estimators: [['nn', new OneNNClassifier()]], // predict-only → 1 meta column
            finalEstimator: recorder,
            cv: 3,
            passthrough: true,
        });
        stack.fit(X, y);
        expect(recorder.trainX[0]).toHaveLength(1 + 2); // 1 meta column + 2 original features
        expect(recorder.trainX[0].slice(1)).toEqual(X[0]);
        expect(stack.transform(X)[0]).toHaveLength(1 + 2);
    });

    it('drops the first predictProba column on binary problems (sklearn behavior)', () => {
        const { X, y } = binaryDataset();
        const gbc = new GradientBoostingClassifier({ nEstimators: 5, maxDepth: 2, randomState: 7 });
        const stack = new StackingClassifier({
            estimators: [['gbc', gbc]],
            finalEstimator: new LogisticRegression({ maxIter: 100 }),
            cv: 3,
        });
        stack.fit(X, y);
        const meta = stack.transform(X);
        expect(meta[0]).toHaveLength(1);
        // the kept column is the positive-class probability of the
        // full-data-refit base estimator
        const proba = gbc.predictProba(X);
        for (let i = 0; i < X.length; i++) {
            expect(meta[i][0]).toBe(proba[i][1]);
        }
    });

    it('meta-features come from out-of-fold predictions, not in-sample leakage', () => {
        // 60 points on a line, true label = side of the midpoint, with 30%
        // deterministic label noise (every 3rd sample flipped).
        const n = 60;
        const X: number[][] = [];
        const y: number[] = [];
        const yClean: number[] = [];
        for (let i = 0; i < n; i++) {
            X.push([i]);
            const clean = i < n / 2 ? 0 : 1;
            yClean.push(clean);
            y.push(i % 3 === 0 ? 1 - clean : clean);
        }

        // Naive (in-sample) stacking would leak: a 1-NN memorizer predicts its
        // own noisy training labels perfectly at distance 0.
        const naive = new OneNNClassifier();
        naive.fit(X, y);
        expect(naive.predict(X)).toEqual(y); // in-sample accuracy 1.0 despite the noise

        const recorder = new RecordingClassifier();
        const stack = new StackingClassifier({
            estimators: [['nn', new OneNNClassifier()]],
            finalEstimator: recorder,
            cv: 5,
        });
        stack.fit(X, y);
        // Out-of-fold, each sample is predicted by a model that never saw it,
        // so the noisy labels cannot be reproduced exactly (in-sample they
        // would be, as asserted above).
        const metaColumn = recorder.trainX.map(row => row[0]);
        expect(accuracyScore(metaColumn, y)).toBeLessThan(1.0);
        // ... but the OOF predictions still track the true signal: each is a
        // neighbor's label, which is clean 2/3 of the time
        expect(accuracyScore(metaColumn, yClean)).toBeGreaterThan(0.6);
    });
});

describe('StackingClassifier params and validation', () => {
    it('supports name__param and finalEstimator__param addressing', () => {
        const stack = new StackingClassifier({
            estimators: [['nb', new GaussianNB()], ['lr', new LogisticRegression()]],
            finalEstimator: new LogisticRegression(),
        });
        stack.setParams({ nb__varSmoothing: 1e-6, lr__maxIter: 50, finalEstimator__maxIter: 42 });
        expect(stack.getEstimator('nb').getParams()).toMatchObject({ varSmoothing: 1e-6 });
        expect(stack.getEstimator('lr').getParams()).toMatchObject({ maxIter: 50 });
        expect(stack.getEstimator('finalEstimator').getParams()).toMatchObject({ maxIter: 42 });
        expect(() => stack.setParams({ missing__x: 1 })).toThrow(/Unknown estimator/);
    });

    it('defaults finalEstimator to LogisticRegression and cv to 5', () => {
        const stack = new StackingClassifier({ estimators: [['nb', new GaussianNB()]] });
        expect(stack.getParams()).toMatchObject({ cv: 5, stackMethod: 'auto', passthrough: false });
        expect(stack.getEstimator('finalEstimator')).toBeInstanceOf(LogisticRegression);
    });

    it('throws at fit when an explicit stackMethod is unavailable on a member', () => {
        const { X, y } = binaryDataset();
        const stack = new StackingClassifier({
            estimators: [['nn', new OneNNClassifier()]], // has neither predictProba nor decisionFunction
            stackMethod: 'predictProba',
        });
        expect(() => stack.fit(X, y)).toThrow(/"nn".*predictProba/);
    });

    it('validates cv and the estimators list', () => {
        expect(() => new StackingClassifier({ estimators: [['a', new GaussianNB()]], cv: 1 })).toThrow(/cv/);
        expect(() => new StackingClassifier({ estimators: [] })).toThrow(/non-empty/);
        expect(() => new StackingClassifier({
            estimators: [['a', new GaussianNB()], ['a', new GaussianNB()]],
        })).toThrow(/Duplicate/);
    });

    it('clone deep-clones members and the final estimator', () => {
        const stack = new StackingClassifier({
            estimators: [['nb', new GaussianNB()]],
            finalEstimator: new LogisticRegression(),
        });
        const copy = stack.clone();
        expect(copy.getEstimator('nb')).not.toBe(stack.getEstimator('nb'));
        expect(copy.getEstimator('finalEstimator')).not.toBe(stack.getEstimator('finalEstimator'));
        expect(copy.getParams().cv).toBe(5);
    });
});

describe('StackingRegressor', () => {
    const { X, y } = regressionDataset(); // 30 samples, 3 features, near-linear target

    it('stacks regressor predictions through the final estimator', () => {
        const stack = new StackingRegressor({
            estimators: [
                ['ridge', new RidgeRegression({ alpha: 0.1 })],
                ['dt', new DecisionTreeRegressor({ max_depth: 4, randomState: 5 })],
            ],
            cv: 3,
        });
        stack.fit(X, y);
        expect(stack.getEstimator('finalEstimator')).toBeInstanceOf(RidgeRegression);
        expect(stack.transform(X)[0]).toHaveLength(2); // one prediction column per member
        expect(stack.score(X, y)).toBeGreaterThan(0.9);
    });

    it('passthrough appends the original features', () => {
        const stack = new StackingRegressor({
            estimators: [['ridge', new RidgeRegression({ alpha: 0.1 })]],
            cv: 3,
            passthrough: true,
        });
        stack.fit(X, y);
        expect(stack.transform(X)[0]).toHaveLength(1 + 3);
    });
});

describe('Stacking serialization', () => {
    it('fitted StackingClassifier round-trips via loadModel with identical predictions', () => {
        const { X, y } = binaryDataset();
        const stack = new StackingClassifier({
            estimators: [
                ['gbc', new GradientBoostingClassifier({ nEstimators: 5, maxDepth: 2, randomState: 11 })],
                ['nb', new GaussianNB()],
            ],
            cv: 3,
        });
        stack.fit(X, y);
        const revived = loadModel(JSON.stringify(stack)) as StackingClassifier;
        expect(revived).toBeInstanceOf(StackingClassifier);
        expect(revived.predict(X)).toEqual(stack.predict(X));
        expect(revived.transform(X)).toEqual(stack.transform(X));
    });

    it('fitted StackingRegressor round-trips via loadModel with identical predictions', () => {
        const { X, y } = regressionDataset();
        const stack = new StackingRegressor({
            estimators: [
                ['ridge', new RidgeRegression({ alpha: 0.1 })],
                ['dt', new DecisionTreeRegressor({ max_depth: 4, randomState: 5 })],
            ],
            cv: 3,
        });
        stack.fit(X, y);
        const revived = loadModel(JSON.stringify(stack)) as StackingRegressor;
        expect(revived.predict(X)).toEqual(stack.predict(X));
    });
});
