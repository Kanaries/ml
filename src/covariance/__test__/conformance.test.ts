import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { EllipticEnvelope } from '../ellipticEnvelope';
import { MinCovDet } from '../minCovDet';
import { EmpiricalCovariance } from '../empiricalCovariance';
import { GraphicalLasso } from '../graphicalLasso';
import { LedoitWolf, OAS, ShrunkCovariance } from '../shrunkCovariance';

runEstimatorConformance([
    {
        name: 'MinCovDet', kind: 'covariance', dataset: 'blobs',
        create: () => new MinCovDet({ supportFraction: .8, randomState: 42 }),
    },
    {
        name: 'EllipticEnvelope', kind: 'outlier', dataset: 'blobs',
        create: () => new EllipticEnvelope({ supportFraction: .8, contamination: .1, randomState: 42 }),
    },
    { name: 'EmpiricalCovariance', kind: 'covariance', dataset: 'blobs', create: () => new EmpiricalCovariance() },
    { name: 'ShrunkCovariance', kind: 'covariance', dataset: 'blobs', create: () => new ShrunkCovariance({ shrinkage: .2 }) },
    { name: 'LedoitWolf', kind: 'covariance', dataset: 'blobs', create: () => new LedoitWolf() },
    { name: 'OAS', kind: 'covariance', dataset: 'blobs', create: () => new OAS() },
    { name: 'GraphicalLasso', kind: 'covariance', dataset: 'blobs', create: () => new GraphicalLasso({ alpha: .1 }) },
]);
