import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { TransformedTargetRegressor } from '../transformedTargetRegressor';

runEstimatorConformance([
    { name: 'TransformedTargetRegressor', kind: 'regressor', dataset: 'regression', create: () => new TransformedTargetRegressor() },
]);
