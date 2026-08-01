import { freezeScenario } from './scenario';

export const facesDecomposition = freezeScenario({
    id: 'faces-decomposition', title: 'Faces decomposition comparison', frozenAt: '2026-07-31',
    dataset: { name: 'Olivetti faces', source: 'sklearn.datasets.fetch_olivetti_faces', protocol: 'first 100 normalized samples, deterministic 8x8 average pooling, 16 components' },
    workflow: ['fit each included decomposition', 'compare reconstruction error and component shape'],
    algorithms: { include: ['PCA', 'NMF', 'FastICA'], exclude: ['DictionaryLearning', 'MiniBatchDictionaryLearning'] },
    parity: { state: 'green', blockedBy: [], reason: 'PCA, NMF, and FastICA component shapes and reconstruction errors are gated on the browser-safe pooled fixture; deferred dictionary learners remain excluded.' },
});
