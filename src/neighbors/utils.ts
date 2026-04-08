import { Distance } from '../metrics';

export type IWeightType = 'uniform' | 'distance';

export interface NeighborHit {
    index: number;
    distance: number;
}

function validateRectangularMatrix(X: number[][]): void {
    if (X.length === 0) {
        throw new Error('X must be non-empty');
    }
    const nFeatures = X[0].length;
    if (nFeatures === 0) {
        throw new Error('X must contain at least one feature');
    }
    for (let i = 1; i < X.length; i++) {
        if (X[i].length !== nFeatures) {
            throw new Error('all rows in X must have the same length');
        }
    }
}

export function validateFitData(X: number[][], y: number[]): void {
    if (X.length === 0 || y.length === 0) {
        throw new Error('X and Y must be non-empty');
    }
    if (X.length !== y.length) {
        throw new Error('X and Y must have the same length');
    }
    validateRectangularMatrix(X);
}

export function validatePredictData(X: number[][], nFeatures: number): void {
    validateRectangularMatrix(X);
    if (X[0].length !== nFeatures) {
        throw new Error('input feature size does not match fitted model');
    }
}

export function getWeights(dis: number[], weightType: IWeightType): number[] {
    const weights = dis.map(() => 1);
    if (weightType === 'uniform') return weights;
    else if (weightType === 'distance') {
        for (let i = 0; i < weights.length; i++) {
            weights[i] = 1 / dis[i];
        }
        return weights;
    } else {
        throw new Error(`Do not support weightType: ${weightType}. Use 'uniform' or 'distance' instead.`);
    }
}

export function getDistance(metric: Distance.IDistanceType) {
    return Distance.useDistance(metric);
}

export function getNeighborHits(
    trainX: number[][],
    sample: number[],
    metric: Distance.IDistanceType,
    p: number,
): NeighborHit[] {
    const distance = Distance.useDistance(metric);
    return trainX
        .map((row, index) => ({
            index,
            distance: distance(sample, row, p),
        }))
        .sort((a, b) => a.distance - b.distance || a.index - b.index);
}

export function getRadiusHits(
    trainX: number[][],
    sample: number[],
    radius: number,
    metric: Distance.IDistanceType,
    p: number,
): NeighborHit[] {
    return getNeighborHits(trainX, sample, metric, p).filter(hit => hit.distance <= radius);
}

export function weightedAverage(values: number[], weights: number[]): number {
    let numerator = 0;
    let denominator = 0;
    for (let i = 0; i < values.length; i++) {
        numerator += values[i] * weights[i];
        denominator += weights[i];
    }
    if (denominator === 0) {
        throw new Error('cannot compute a weighted average with zero total weight');
    }
    return numerator / denominator;
}

export function weightedMode(labels: number[], weights: number[]): number {
    const totals = new Map<number, number>();
    for (let i = 0; i < labels.length; i++) {
        totals.set(labels[i], (totals.get(labels[i]) || 0) + weights[i]);
    }
    let bestLabel = labels[0];
    let bestWeight = -Infinity;
    const orderedLabels = Array.from(totals.keys()).sort((a, b) => a - b);
    for (const label of orderedLabels) {
        const weight = totals.get(label) || 0;
        if (weight > bestWeight) {
            bestWeight = weight;
            bestLabel = label;
        }
    }
    return bestLabel;
}

export function resolveDistanceWeights(distances: number[]): number[] {
    const zeroDistanceCount = distances.filter(distance => distance === 0).length;
    if (zeroDistanceCount > 0) {
        return distances.map(distance => (distance === 0 ? 1 : 0));
    }
    return distances.map(distance => 1 / distance);
}

export function votes(voteClasses: number[], weights: number[]): number {
    const counter: Map<any, number> = new Map();
    for (let i = 0; i < voteClasses.length; i++) {
        if (!counter.has(voteClasses[i])) {
            counter.set(voteClasses[i], 0);
        }
        counter.set(voteClasses[i], counter.get(voteClasses[i]) + weights[i]);
    }
    let mode = 0;
    let modeCount = 0;
    for (let [m, mc] of counter) {
        if (mc > modeCount) {
            mode = m;
            modeCount = mc;
        }
    }
    return mode;
}
