import type { BaseEstimator } from '../base';
import type { FeatureImportanceCapability } from '../base/capabilities';
import { createRandomGenerator } from '../utils';

export interface ForestMember extends BaseEstimator, FeatureImportanceCapability {
    fit(X: number[][], y: number[]): void;
    predict(X: number[][]): number[];
}

export interface ForestFitOptions<TMember extends ForestMember> {
    nEstimators: number;
    bootstrap: boolean;
    randomState?: number;
    /** Injects the tree family while the skeleton owns sampling and seeding. */
    createEstimator: (randomState: number) => TMember;
}

/** Shared bootstrap/training skeleton used by RandomForest and ExtraTrees. */
export function fitForest<TMember extends ForestMember>(
    X: number[][],
    y: number[],
    options: ForestFitOptions<TMember>,
): TMember[] {
    if (X.length === 0 || y.length === 0) throw new Error('X and y must be non-empty');
    if (X.length !== y.length) throw new Error('X and y must have the same length');
    if (!Number.isInteger(options.nEstimators) || options.nEstimators < 1) {
        throw new Error('nEstimators must be a positive integer');
    }

    const random = createRandomGenerator(options.randomState);
    const estimators: TMember[] = [];
    for (let i = 0; i < options.nEstimators; i++) {
        const estimator = options.createEstimator(Math.floor(random() * 1_000_000_000));
        const sampleX: number[][] = [];
        const sampleY: number[] = [];
        if (options.bootstrap) {
            for (let j = 0; j < X.length; j++) {
                const index = Math.floor(random() * X.length);
                sampleX.push(X[index]);
                sampleY.push(y[index]);
            }
        } else {
            sampleX.push(...X);
            sampleY.push(...y);
        }
        estimator.fit(sampleX, sampleY);
        estimators.push(estimator);
    }
    return estimators;
}

export function predictForestClassification(estimators: ForestMember[], X: number[][]): number[] {
    return X.map(row => {
        const votes = new Map<number, number>();
        for (const estimator of estimators) {
            const label = estimator.predict([row])[0];
            votes.set(label, (votes.get(label) ?? 0) + 1);
        }
        let bestLabel = 0;
        let bestCount = -1;
        for (const label of Array.from(votes.keys()).sort((a, b) => a - b)) {
            const count = votes.get(label)!;
            if (count > bestCount) {
                bestCount = count;
                bestLabel = label;
            }
        }
        return bestLabel;
    });
}

export function predictForestRegression(estimators: ForestMember[], X: number[][]): number[] {
    const predictions = estimators.map(estimator => estimator.predict(X));
    return X.map((_, row) => predictions.reduce((sum, values) => sum + values[row], 0) / estimators.length);
}
