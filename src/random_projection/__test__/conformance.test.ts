import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { GaussianRandomProjection, SparseRandomProjection } from '../randomProjection';

runEstimatorConformance([
    { name: 'GaussianRandomProjection', kind: 'transformer', dataset: 'blobs', create: () => new GaussianRandomProjection({ nComponents: 2, randomState: 42 }) },
    { name: 'SparseRandomProjection', kind: 'transformer', dataset: 'blobs', create: () => new SparseRandomProjection({ nComponents: 2, randomState: 42 }) },
]);
