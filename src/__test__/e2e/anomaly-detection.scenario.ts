import { freezeScenario } from './scenario';

export const anomalyDetection = freezeScenario({
    id: 'anomaly-detection', title: 'Anomaly detection comparison', frozenAt: '2026-07-31',
    dataset: { name: 'two Gaussian inlier clouds plus uniform outliers', source: 'fixed generated fixture', protocol: 'random_state=42; train on the full frozen sample' },
    workflow: ['fit each detector at contamination=0.1', 'compare decision scores, labels, and outlier recall'],
    algorithms: { include: ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'EllipticEnvelope'], exclude: [] },
    parity: { state: 'pending', blockedBy: [], reason: 'Wave A implementations are present; the cross-algorithm fixture remains an exit-gate task.' },
});
