import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { IterativeImputer } from '../iterativeImputer';

runEstimatorConformance([
    { name: 'IterativeImputer', kind: 'transformer', dataset: 'regression', create: () => new IterativeImputer({ maxIter: 3 }) },
]);
