import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { GaussianProcessClassifier, GaussianProcessRegressor } from '../index';

runEstimatorConformance([
    {
        name: 'GaussianProcessRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new GaussianProcessRegressor({ alpha: 0.05, normalizeY: true }),
    },
    {
        name: 'GaussianProcessClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new GaussianProcessClassifier({ lengthScale: 1.2 }),
    },
]);
