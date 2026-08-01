import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { CCA, PLSRegression } from '../pls';

runEstimatorConformance([
    { name: 'PLSRegression', kind: 'transformer', dataset: 'regression', create: () => new PLSRegression({ nComponents: 2 }) },
    { name: 'CCA', kind: 'transformer', dataset: 'regression', create: () => new CCA({ nComponents: 1 }) },
]);
