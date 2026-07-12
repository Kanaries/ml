import { loadModel } from '../../base/estimator';
import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { LabelPropagation } from '../labelPropagation';
import { LabelSpreading } from '../labelSpreading';

// The harness fits on fully-labeled y — valid input for label propagation
// (a y vector with no -1 entries simply clamps every point).
runEstimatorConformance([
    {
        name: 'LabelPropagation',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new LabelPropagation({ kernel: 'rbf', gamma: 20, maxIter: 100, tol: 1e-3 }),
    },
    {
        name: 'LabelSpreading',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new LabelSpreading({ kernel: 'rbf', gamma: 20, alpha: 0.2, maxIter: 30, tol: 1e-3 }),
    },
]);

// Extra coverage for the class's real usage: semi-labeled input (-1 entries)
// must survive serialize → revive with identical transduction and predictions.
describe('semi-supervised (-1 labels) serialization parity', () => {
    const X = [[0, 0], [0.5, 0.5], [1, 0], [5, 5], [5.5, 4.5], [6, 5], [0, 6], [0.5, 6.5], [1, 6]];
    const y = [0, -1, -1, 1, -1, -1, 2, -1, -1];

    it.each([
        ['LabelPropagation', () => new LabelPropagation({ gamma: 1, maxIter: 1000 })],
        ['LabelSpreading', () => new LabelSpreading({ gamma: 1, alpha: 0.2, maxIter: 30 })],
    ])('%s', (_name, create) => {
        const est = create();
        est.fit(X, y);
        const pred = est.predict(X);
        expect(pred).toHaveLength(X.length);
        const revived = loadModel(JSON.stringify(est)) as LabelPropagation | LabelSpreading;
        expect(revived.constructor).toBe(est.constructor);
        expect(revived.getParams()).toEqual(est.getParams());
        expect(revived.getTransduction()).toEqual(est.getTransduction());
        expect(revived.predict(X)).toEqual(pred);
    });
});
