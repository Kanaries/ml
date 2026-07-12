/**
 * The canonical sklearn tutorial, reproduced 1:1 with @kanaries/ml:
 *   make dataset → train/test split → Pipeline(scaler + model)
 *   → GridSearchCV over pipeline params → evaluate → serialize/revive.
 *
 * This is the Phase 1 acceptance test from docs/ML_ROADMAP_NEXT.md.
 */
import { makeClassification } from '../datasets';
import { Pipeline } from '../pipeline';
import { StandardScaler } from '../utils/preprocessing';
import { GridSearchCV, StratifiedKFold } from '../utils/modelSelection';
import { LogisticRegression } from '../linear/logisticRegression';
import { classificationReport, accuracyScore } from '../metrics';
import { loadModel } from '../base';

function splitTrainTest(X: number[][], y: number[], testEvery: number): {
    XTrain: number[][]; yTrain: number[]; XTest: number[][]; yTest: number[];
} {
    const XTrain: number[][] = [], yTrain: number[] = [], XTest: number[][] = [], yTest: number[] = [];
    for (let i = 0; i < X.length; i++) {
        if (i % testEvery === 0) { XTest.push(X[i]); yTest.push(y[i]); }
        else { XTrain.push(X[i]); yTrain.push(y[i]); }
    }
    return { XTrain, yTrain, XTest, yTest };
}

describe('end-to-end sklearn-style workflow', () => {
    it('dataset → pipeline → grid search → report → serialize', () => {
        // 1. dataset
        const { X, y } = makeClassification({
            nSamples: 120,
            nFeatures: 6,
            nInformative: 3,
            nRedundant: 1,
            nClasses: 2,
            nClustersPerClass: 1,
            flipY: 0,
            classSep: 2,
            randomState: 7,
        });
        expect(X).toHaveLength(120);

        // 2. split
        const { XTrain, yTrain, XTest, yTest } = splitTrainTest(X, y, 5);

        // 3. pipeline + 4. grid search over nested pipeline params
        const search = new GridSearchCV({
            estimator: new Pipeline({
                steps: [
                    ['scale', new StandardScaler()],
                    ['clf', new LogisticRegression()],
                ],
            }),
            paramGrid: {
                clf__learningRate: [0.01, 0.3],
                clf__maxIter: [50, 400],
            },
            cv: new StratifiedKFold({ nSplits: 3 }),
            scoring: 'accuracyScore',
        });
        search.fit(XTrain, yTrain);

        expect(search.bestParams).not.toBeNull();
        expect(search.bestScore).toBeGreaterThan(0.85);

        // 5. evaluate on held-out data
        const best = search.bestEstimator as Pipeline;
        const preds = best.predict(XTest);
        const report = classificationReport(preds, yTest);
        expect(report.accuracy).toBeGreaterThan(0.85);
        expect(Object.keys(report.perClass)).toHaveLength(2);

        // 6. persist the whole fitted pipeline and revive it elsewhere
        const revived = loadModel(JSON.stringify(best)) as Pipeline;
        expect(revived.predict(XTest)).toEqual(preds);
        expect(accuracyScore(revived.predict(XTest), yTest)).toBe(report.accuracy);
    });
});
