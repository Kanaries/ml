import { freezeScenario } from './scenario';

export const clusteringComparison = freezeScenario({
    id: 'clustering-comparison', title: 'Clustering algorithm comparison', frozenAt: '2026-07-31',
    dataset: { name: 'varied 2D cluster shapes', source: 'fixed make_blobs/make_moons/circles fixtures', protocol: '120 samples per shape (browser/CI projection), random_state=42' },
    workflow: ['standardize each dataset', 'fit every included clusterer', 'compare labels up to permutation and adjusted Rand score'],
    algorithms: { include: ['KMeans', 'MiniBatchKMeans', 'DBSCAN', 'AgglomerativeClustering', 'MeanShift', 'SpectralClustering', 'Birch', 'AffinityPropagation', 'GaussianMixture'], exclude: [] },
    parity: { state: 'green', blockedBy: [], reason: 'Every included clusterer runs on all three standardized shapes and is gated by sklearn adjusted Rand score.' },
});
