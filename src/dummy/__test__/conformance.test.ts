import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { DummyClassifier, DummyRegressor } from '../dummy';

runEstimatorConformance([
    {
        name: 'DummyClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        // 'stratified' exercises the predict-time RNG: a fresh generator is
        // derived from randomState on every call, so output is deterministic.
        create: () => new DummyClassifier({ strategy: 'stratified', randomState: 42 }),
    },
    {
        name: 'DummyRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new DummyRegressor({ strategy: 'quantile', quantile: 0.75 }),
    },
]);
