import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { KMeans } from '../kmeans';
import { DBScan } from '../dbscan';
import { MeanShift } from '../meanShift';
import { OPTICS } from '../optics';
import { HDBScan } from '../hdbscan';
import { AgglomerativeClustering } from '../agglomerativeClustering';
import { SpectralClustering } from '../spectralClustering';
import { MiniBatchKMeans } from '../miniBatchKMeans';

runEstimatorConformance([
    {
        name: 'KMeans',
        kind: 'cluster',
        create: () => new KMeans({ n_clusters: 3, random_state: 42 }),
    },
    {
        name: 'DBScan',
        kind: 'cluster',
        dataset: 'blobs',
        // blobs have spread 1 and centers >= 6.3 apart: eps 1.5 links points
        // within a blob without bridging blobs
        create: () => new DBScan({ eps: 1.5, minSamples: 3 }),
    },
    {
        name: 'MeanShift',
        kind: 'cluster',
        dataset: 'blobs',
        create: () => new MeanShift({ bandwidth: 2 }),
    },
    {
        name: 'OPTICS',
        kind: 'cluster',
        dataset: 'blobs',
        create: () => new OPTICS({ eps: 1.5, min_samples: 3 }),
    },
    {
        name: 'HDBScan',
        kind: 'cluster',
        dataset: 'blobs',
        create: () => new HDBScan({ min_cluster_size: 5 }),
    },
    {
        name: 'AgglomerativeClustering',
        kind: 'cluster',
        dataset: 'blobs',
        // fully deterministic (no randomness anywhere)
        create: () => new AgglomerativeClustering({ nClusters: 3, linkage: 'ward' }),
    },
    {
        name: 'SpectralClustering',
        kind: 'cluster',
        dataset: 'blobs',
        // blobs are >= 6.3 apart: with gamma=1 the between-blob rbf affinity
        // is ~0, so the graph has 3 near-disconnected components
        create: () => new SpectralClustering({ nClusters: 3, randomState: 42 }),
    },
    {
        name: 'MiniBatchKMeans',
        kind: 'cluster',
        dataset: 'blobs',
        create: () => new MiniBatchKMeans({ nClusters: 3, randomState: 42, batchSize: 16 }),
    },
]);

// Regression tests for CodeX Phase 0 review finding #4: getParams must return
// RAW dependent params, so setParams on the driving param behaves exactly like
// fresh construction.
import { blobsDataset } from '../../__test__/conformance/datasets';

describe('dependent-param resolution (raw props in getParams)', () => {
    const { X } = blobsDataset();

    it('HDBScan: setParams({min_cluster_size}) rederives min_samples', () => {
        const viaSetParams = new HDBScan({ min_cluster_size: 5 });
        viaSetParams.setParams({ min_cluster_size: 10 });
        expect(viaSetParams.getParams()).toEqual(new HDBScan({ min_cluster_size: 10 }).getParams());
        expect(viaSetParams.fitPredict(X)).toEqual(new HDBScan({ min_cluster_size: 10 }).fitPredict(X));
        // explicit min_samples still wins
        expect(new HDBScan({ min_cluster_size: 10, min_samples: 3 }).getParams()).toMatchObject({ min_samples: 3 });
    });

    it('OPTICS: setParams({eps}) rederives max_eps', () => {
        const viaSetParams = new OPTICS({ min_samples: 3 });
        viaSetParams.setParams({ eps: 2 });
        expect(viaSetParams.getParams()).toEqual(new OPTICS({ min_samples: 3, eps: 2 }).getParams());
        expect(viaSetParams.fitPredict(X)).toEqual(new OPTICS({ min_samples: 3, eps: 2 }).fitPredict(X));
        expect(new OPTICS({ eps: 2, max_eps: 5 }).getParams()).toMatchObject({ max_eps: 5 });
    });
});
