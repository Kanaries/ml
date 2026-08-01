import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { PCA } from '../pca';
import { SparsePCA } from '../sparsePCA';
import { TruncatedSVD } from '../truncatedSVD';
import { KernelPCA } from '../kernelPCA';
import { FastICA } from '../fastICA';
import { NMF } from '../nmf';
import { IncrementalPCA } from '../incrementalPCA';

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
    { name: 'KernelPCA', kind: 'transformer', dataset: 'blobs', create: () => new KernelPCA({ nComponents: 2, kernel: 'rbf', gamma: .1 }) },
    { name: 'FastICA', kind: 'transformer', dataset: 'blobs', create: () => new FastICA({ nComponents: 2, randomState: 42, maxIter: 500 }) },
    { name: 'NMF', kind: 'transformer', dataset: 'counts', create: () => new NMF({ nComponents: 2, randomState: 42, maxIter: 100 }) },
    { name: 'IncrementalPCA', kind: 'transformer', dataset: 'blobs', create: () => new IncrementalPCA({ nComponents: 2, batchSize: 10 }) },
]);
