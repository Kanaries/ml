import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { PCA } from '../pca';
import { SparsePCA } from '../sparsePCA';
import { TruncatedSVD } from '../truncatedSVD';

runEstimatorConformance([
    {
        name: 'PCA',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new PCA({ nComponents: 2 }),
    },
    {
        // deterministic: uses an internally seeded init (no randomState param)
        name: 'SparsePCA',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new SparsePCA({ nComponents: 2, alpha: 0.1 }),
    },
    {
        name: 'TruncatedSVD',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new TruncatedSVD({ nComponents: 2 }),
    },
]);
