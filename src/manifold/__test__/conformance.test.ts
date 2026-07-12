import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { LocallyLinearEmbedding } from '../lle';
import { MDS } from '../mds';
import { SpectralEmbedding } from '../spectralEmbedding';
import { TSNE } from '../tsne';

runEstimatorConformance([
    {
        // fitTransform-only (no out-of-sample transform)
        name: 'TSNE',
        kind: 'embedding',
        dataset: 'blobs',
        create: () => new TSNE({ nComponents: 2, perplexity: 10, nIter: 250, randomState: 42 }),
    },
    {
        // fitTransform-only (no out-of-sample transform)
        name: 'MDS',
        kind: 'embedding',
        dataset: 'blobs',
        create: () => new MDS({ nComponents: 2, randomState: 42 }),
    },
    {
        // fitTransform-only (no out-of-sample transform)
        name: 'SpectralEmbedding',
        kind: 'embedding',
        dataset: 'blobs',
        create: () => new SpectralEmbedding({ nComponents: 2, nNeighbors: 5, randomState: 42 }),
    },
    {
        // supports out-of-sample transform via reconstruction weights
        name: 'LocallyLinearEmbedding',
        kind: 'transformer',
        dataset: 'blobs',
        create: () => new LocallyLinearEmbedding({ nNeighbors: 5, nComponents: 2, reg: 1e-3, randomState: 42 }),
    },
]);
