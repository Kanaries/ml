import { freezeScenario } from './scenario';

export const manifoldComparison = freezeScenario({
    id: 'manifold-comparison', title: 'Manifold learning comparison', frozenAt: '2026-07-31',
    dataset: { name: 'S-curve', source: 'sklearn.datasets.make_s_curve', protocol: '120 samples (browser/CI projection), noise=0.05, random_state=42' },
    workflow: ['fit two-dimensional embeddings', 'compare pairwise-distance correlation and neighborhood preservation'],
    algorithms: { include: ['TSNE', 'LocallyLinearEmbedding', 'MDS', 'Isomap', 'SpectralEmbedding'], exclude: ['ModifiedLLE', 'HessianLLE', 'LTSA'] },
    parity: { state: 'green', blockedBy: [], reason: 'All five embeddings are gated by rotation-invariant trustworthiness and pairwise-distance correlation; only standard LLE is in scope.' },
});
