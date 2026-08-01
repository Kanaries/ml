import { freezeScenario } from './scenario';

export const clusteringComparison = freezeScenario({
    id: 'clustering-comparison', title: 'Clustering algorithm comparison', frozenAt: '2026-07-31',
    dataset: { name: 'varied 2D cluster shapes', source: 'fixed make_blobs/make_moons/circles fixtures', protocol: '500 samples per shape, random_state=42' },
    workflow: ['standardize each dataset', 'fit every included clusterer', 'compare labels up to permutation and adjusted Rand score'],
    algorithms: { include: ['KMeans', 'MiniBatchKMeans', 'DBSCAN', 'AgglomerativeClustering', 'MeanShift', 'SpectralClustering', 'Birch', 'AffinityPropagation', 'GaussianMixture'], exclude: [] },
    parity: { state: 'pending', blockedBy: [], reason: 'Wave C clusterers are implemented; the frozen cross-algorithm fixture remains an exit-gate task.' },
});
