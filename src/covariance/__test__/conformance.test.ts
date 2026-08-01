import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { EllipticEnvelope } from '../ellipticEnvelope';
import { MinCovDet } from '../minCovDet';

runEstimatorConformance([
    {
        name: 'MinCovDet', kind: 'covariance', dataset: 'blobs',
        create: () => new MinCovDet({ supportFraction: .8, randomState: 42 }),
    },
    {
        name: 'EllipticEnvelope', kind: 'outlier', dataset: 'blobs',
        create: () => new EllipticEnvelope({ supportFraction: .8, contamination: .1, randomState: 42 }),
    },
]);
