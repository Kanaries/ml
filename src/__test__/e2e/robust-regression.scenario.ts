import { freezeScenario } from './scenario';

export const robustRegression = freezeScenario({
    id: 'robust-regression', title: 'Robust regression comparison', frozenAt: '2026-07-31',
    dataset: { name: 'linear regression with vertical outliers', source: 'fixed generated fixture', protocol: '200 samples, 20 outliers, random_state=42' },
    workflow: ['fit each robust regressor', 'compare coefficients, inlier masks where available, and median absolute error'],
    algorithms: { include: ['HuberRegressor', 'RANSACRegressor', 'TheilSenRegressor'], exclude: [] },
    parity: { state: 'green', blockedBy: [], reason: 'All three regressors are gated on coefficients, intercept, clean-target median absolute error, and RANSAC consensus agreement.' },
});
