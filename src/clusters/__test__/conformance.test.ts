import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { KMeans } from '../kmeans';

runEstimatorConformance([
    {
        name: 'KMeans',
        kind: 'cluster',
        create: () => new KMeans({ n_clusters: 3, random_state: 42 }),
    },
]);
