import { freezeScenario } from './scenario';

export const featureSelection = freezeScenario({
    id: 'feature-selection', title: 'Feature selection pipeline', frozenAt: '2026-07-31',
    dataset: { name: 'informative plus noise classification', source: 'fixed make_classification fixture', protocol: '200 samples, 4 informative of 12 features, random_state=42' },
    workflow: ['SelectFromModel and RFE', 'transform train/test without leakage', 'fit LogisticRegression', 'compare masks and held-out accuracy'],
    algorithms: { include: ['SelectFromModel', 'RFE', 'LogisticRegression', 'Pipeline'], exclude: [] },
    parity: { state: 'green', blockedBy: [], reason: 'Both selectors execute inside leakage-safe scaled pipelines with support-mask overlap and held-out accuracy gates.' },
});
