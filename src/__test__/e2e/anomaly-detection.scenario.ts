import { freezeScenario } from './scenario';

export const anomalyDetection = freezeScenario({
    id: 'anomaly-detection', title: 'Anomaly detection comparison', frozenAt: '2026-07-31',
    dataset: { name: 'two Gaussian inlier clouds plus uniform outliers', source: 'fixed generated fixture', protocol: 'random_state=42; train on the full frozen sample' },
    workflow: ['fit each detector at contamination=0.1', 'compare decision scores, labels, and outlier recall'],
    algorithms: { include: ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'EllipticEnvelope'], exclude: [] },
    parity: { state: 'green', blockedBy: [], reason: 'All four detectors execute on the same frozen sample with normalized label conventions, sklearn agreement, and outlier-recall gates.' },
});
