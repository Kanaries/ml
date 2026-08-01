import { irisClassification } from './iris-classification.scenario';
import { digitsSvc } from './digits-svc.scenario';
import { textClassification } from './text-classification.scenario';
import { facesDecomposition } from './faces-decomposition.scenario';
import { anomalyDetection } from './anomaly-detection.scenario';
import { clusteringComparison } from './clustering-comparison.scenario';
import { manifoldComparison } from './manifold-comparison.scenario';
import { featureSelection } from './feature-selection.scenario';
import { probabilityCalibration } from './probability-calibration.scenario';
import { robustRegression } from './robust-regression.scenario';

const scenarios = [
    irisClassification, digitsSvc, textClassification, facesDecomposition, anomalyDetection,
    clusteringComparison, manifoldComparison, featureSelection, probabilityCalibration, robustRegression,
];

const expectedIds = [
    'iris-classification', 'digits-svc', 'text-classification', 'faces-decomposition',
    'anomaly-detection', 'clustering-comparison', 'manifold-comparison', 'feature-selection',
    'probability-calibration', 'robust-regression',
];

test('Phase 3 freezes exactly the ten named scenario specs at Wave A start', () => {
    expect(scenarios.map(scenario => scenario.id)).toEqual(expectedIds);
    expect(new Set(scenarios.map(scenario => scenario.id)).size).toBe(10);
    scenarios.forEach(scenario => expect(scenario.frozenAt).toBe('2026-07-31'));
});

test('every frozen scenario has an explicit dataset, workflow, inclusion set, and exclusion set', () => {
    scenarios.forEach(scenario => {
        expect(scenario.dataset.name.length).toBeGreaterThan(0);
        expect(scenario.dataset.source.length).toBeGreaterThan(0);
        expect(scenario.dataset.protocol.length).toBeGreaterThan(0);
        expect(scenario.workflow.length).toBeGreaterThan(0);
        expect(scenario.algorithms.include.length).toBeGreaterThan(0);
        const overlap = scenario.algorithms.include.filter(name => scenario.algorithms.exclude.includes(name));
        expect(overlap).toEqual([]);
        expect(scenario.parity.reason.length).toBeGreaterThan(0);
    });
});

test('implemented roadmap waves leave no scenario blocked on a missing estimator', () => {
    const blocked = scenarios.filter(scenario => scenario.parity.blockedBy.length > 0);
    expect(blocked.map(scenario => scenario.id)).toEqual([]);
    blocked.forEach(scenario => expect(scenario.parity.state).toBe('pending'));
});
