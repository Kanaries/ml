import { freezeScenario } from './scenario';

export const irisClassification = freezeScenario({
    id: 'iris-classification', title: 'Iris classification pipeline', frozenAt: '2026-07-31',
    dataset: { name: 'Iris', source: 'sklearn.datasets.load_iris', protocol: 'stratified 75/25 split, random_state=42' },
    workflow: ['StandardScaler', 'Pipeline', 'GridSearchCV over LogisticRegression regularization', 'accuracy parity'],
    algorithms: { include: ['StandardScaler', 'LogisticRegression', 'GridSearchCV', 'Pipeline'], exclude: [] },
    parity: { state: 'pending', blockedBy: [], reason: 'Frozen fixture and expected predictions still need to be encoded.' },
});
