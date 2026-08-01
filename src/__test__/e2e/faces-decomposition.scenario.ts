import { freezeScenario } from './scenario';

export const facesDecomposition = freezeScenario({
    id: 'faces-decomposition', title: 'Faces decomposition comparison', frozenAt: '2026-07-31',
    dataset: { name: 'Olivetti faces', source: 'sklearn.datasets.fetch_olivetti_faces', protocol: 'first 100 normalized samples, 16 components' },
    workflow: ['fit each included decomposition', 'compare reconstruction error and component shape'],
    algorithms: { include: ['PCA', 'NMF', 'FastICA'], exclude: ['DictionaryLearning', 'MiniBatchDictionaryLearning'] },
    parity: { state: 'pending', blockedBy: ['Wave B: NMF', 'Wave B: FastICA'], reason: 'The frozen subset deliberately excludes deferred dictionary-learning algorithms.' },
});
