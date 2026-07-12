import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { KMeans } from '../kmeans';
import { DBScan } from '../dbscan';
import { MeanShift } from '../meanShift';
import { OPTICS } from '../optics';
import { HDBScan } from '../hdbscan';

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
]);
