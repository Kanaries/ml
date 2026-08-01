import { freezeScenario } from './scenario';

export const robustRegression = freezeScenario({
    id: 'robust-regression', title: 'Robust regression comparison', frozenAt: '2026-07-31',
    dataset: { name: 'linear regression with vertical outliers', source: 'fixed generated fixture', protocol: '200 samples, 20 outliers, random_state=42' },
    workflow: ['fit each robust regressor', 'compare coefficients, inlier masks where available, and median absolute error'],
    algorithms: { include: ['HuberRegressor', 'RANSACRegressor', 'TheilSenRegressor'], exclude: [] },
    parity: { state: 'pending', blockedBy: [], reason: 'Wave C robust regressors are implemented; the frozen cross-algorithm fixture remains an exit-gate task.' },
});
