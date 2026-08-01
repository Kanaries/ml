import { freezeScenario } from './scenario';

export const digitsSvc = freezeScenario({
    id: 'digits-svc', title: 'Digits handwritten recognition', frozenAt: '2026-07-31',
    dataset: { name: 'Digits', source: 'sklearn.datasets.load_digits', protocol: 'stratified 80/20 split, random_state=42' },
    workflow: ['scale pixel features', 'fit RBF SVC', 'compare held-out predictions and accuracy'],
    algorithms: { include: ['StandardScaler', 'SVC'], exclude: [] },
    parity: { state: 'green', blockedBy: [], reason: 'The full Digits split executes StandardScaler and RBF SVC against frozen sklearn predictions and accuracy.' },
});
