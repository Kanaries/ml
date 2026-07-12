/**
 * Regression tests for the Phase 2 independent review (Opus 4.8) finding:
 * capability detection via `typeof est.predictProba === 'function'` broke for
 * margin-loss SGDClassifier/Perceptron, whose predictProba existed but threw.
 * The capability is now absent (undefined) for those losses, so calibration,
 * one-vs-rest, stacking and soft voting fall back to decisionFunction.
 * See docs/reviews/phase2-review.md.
 */
import { SGDClassifier, Perceptron, LogisticRegression } from '../linear';
import { CalibratedClassifierCV } from '../calibration';
import { OneVsRestClassifier } from '../multiclass';
import { StackingClassifier } from '../ensemble';
import { loadModel } from '../base';
import { binaryDataset, multiclassDataset } from './conformance/datasets';

describe('margin-loss SGD exposes no predictProba capability', () => {
    it('predictProba is absent for hinge/perceptron, present for logLoss/modifiedHuber', () => {
        expect(new SGDClassifier().predictProba).toBeUndefined();
        expect(new SGDClassifier({ loss: 'squaredHinge' }).predictProba).toBeUndefined();
        expect(new Perceptron().predictProba).toBeUndefined();
        expect(typeof new SGDClassifier({ loss: 'logLoss' }).predictProba).toBe('function');
        expect(typeof new SGDClassifier({ loss: 'modifiedHuber' }).predictProba).toBe('function');
    });

    it('the hidden capability survives serialization and setParams', () => {
        const { X, y } = binaryDataset();
        const est = new SGDClassifier({ randomState: 1, maxIter: 50 });
        est.fit(X, y);
        const revived = loadModel(JSON.stringify(est)) as SGDClassifier;
        expect(revived.predictProba).toBeUndefined();
        expect(revived.predict(X)).toEqual(est.predict(X));
        est.setParams({ loss: 'logLoss' });
        expect(typeof est.predictProba).toBe('function');
        est.setParams({ loss: 'hinge' });
        expect(est.predictProba).toBeUndefined();
    });
});

describe('meta-estimators fall back to decisionFunction for margin classifiers', () => {
    const binary = binaryDataset();
    const multi = multiclassDataset();

    it('CalibratedClassifierCV calibrates a default (hinge) SGDClassifier', () => {
        const cal = new CalibratedClassifierCV({
            estimator: new SGDClassifier({ randomState: 7, maxIter: 200 }),
            cv: 3,
        });
        cal.fit(binary.X, binary.y);
        const proba = cal.predictProba(binary.X);
        for (const row of proba) {
            expect(row.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 10);
        }
        expect(cal.score(binary.X, binary.y)).toBeGreaterThan(0.9);
    });

    it('OneVsRestClassifier predicts with a hinge SGDClassifier base', () => {
        const ovr = new OneVsRestClassifier({
            estimator: new SGDClassifier({ randomState: 7, maxIter: 200 }),
        });
        ovr.fit(multi.X, multi.y);
        expect(ovr.score(multi.X, multi.y)).toBeGreaterThan(0.9);
        // members expose no predictProba, so the wrapper's predictProba throws clearly
        expect(() => ovr.predictProba(multi.X)).toThrow(/predictProba/);
    });

    it('StackingClassifier stackMethod auto resolves to decisionFunction for hinge SGD', () => {
        const stack = new StackingClassifier({
            estimators: [
                ['sgd', new SGDClassifier({ randomState: 7, maxIter: 200 })],
                ['lr', new LogisticRegression({ maxIter: 200 })],
            ],
            finalEstimator: new LogisticRegression({ maxIter: 200 }),
            cv: 3,
        });
        stack.fit(binary.X, binary.y);
        expect(stack.score(binary.X, binary.y)).toBeGreaterThan(0.9);
    });
});
