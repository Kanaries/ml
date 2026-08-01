import { freezeScenario } from './scenario';

export const probabilityCalibration = freezeScenario({
    id: 'probability-calibration', title: 'Probability calibration', frozenAt: '2026-07-31',
    dataset: { name: 'binary classification', source: 'fixed make_classification fixture', protocol: '75/25 train/test split with internal deterministic 3-fold calibration, random_state=42' },
    workflow: ['fit base classifier', 'calibrate with sigmoid and isotonic modes', 'compare probabilities, Brier score, and labels'],
    algorithms: { include: ['CalibratedClassifierCV'], exclude: [] },
    parity: { state: 'green', blockedBy: [], reason: 'Sigmoid and isotonic modes are gated by held-out Brier score, probability correlation, and accuracy.' },
});
