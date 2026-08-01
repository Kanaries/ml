import { freezeScenario } from './scenario';

export const probabilityCalibration = freezeScenario({
    id: 'probability-calibration', title: 'Probability calibration', frozenAt: '2026-07-31',
    dataset: { name: 'binary classification', source: 'fixed make_classification fixture', protocol: 'train/calibration/test split, random_state=42' },
    workflow: ['fit base classifier', 'calibrate with sigmoid and isotonic modes', 'compare probabilities, Brier score, and labels'],
    algorithms: { include: ['CalibratedClassifierCV'], exclude: [] },
    parity: { state: 'pending', blockedBy: [], reason: 'Frozen fixture and expected probabilities still need to be encoded.' },
});
