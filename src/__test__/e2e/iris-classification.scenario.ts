import { freezeScenario } from './scenario';

export const irisClassification = freezeScenario({
    id: 'iris-classification', title: 'Iris classification pipeline', frozenAt: '2026-07-31',
    dataset: { name: 'Iris', source: 'sklearn.datasets.load_iris', protocol: 'stratified 75/25 split, random_state=42' },
    workflow: ['StandardScaler', 'Pipeline', 'GridSearchCV over LogisticRegression regularization', 'accuracy parity'],
    algorithms: { include: ['StandardScaler', 'LogisticRegression', 'GridSearchCV', 'Pipeline'], exclude: [] },
    parity: { state: 'green', blockedBy: [], reason: 'The sklearn 1.9 fixture executes the full scaled multiclass regularization search and held-out accuracy gate.' },
});
