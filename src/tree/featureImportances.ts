import type { IDTree } from './decisionTreeClassifier';

export function treeImportances(root: IDTree | null, nFeatures: number, normalize = true): number[] {
    const values = new Array(nFeatures).fill(0);
    const visit = (node: IDTree | null): void => {
        if (!node) return;
        if (node.splitIndex >= 0 && node.splitIndex < nFeatures) {
            if (node.weightedImpurityDecrease === undefined) {
                throw new Error('featureImportances are unavailable for this legacy tree model; refit the model');
            }
            values[node.splitIndex] += Math.max(0, node.weightedImpurityDecrease);
        }
        visit(node.leftChild);
        visit(node.rightChild);
    };
    visit(root);
    const total = values.reduce((sum, value) => sum + value, 0);
    return !normalize || total === 0 ? values : values.map(value => value / total);
}

export function normalizedTreeImportances(root: IDTree | null, nFeatures: number): number[] {
    return treeImportances(root, nFeatures, true);
}

export function averageImportances(rows: number[][]): number[] {
    return weightedImportances(rows);
}

export function weightedImportances(rows: number[][], weights?: number[]): number[] {
    if (rows.length === 0) return [];
    if (weights && weights.length !== rows.length) throw new Error('importance weights must match row count');
    const nFeatures = rows[0].length;
    const out = new Array(nFeatures).fill(0);
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        if (row.length !== nFeatures) throw new Error('feature-importance rows have inconsistent lengths');
        const weight = weights?.[i] ?? 1;
        for (let j = 0; j < nFeatures; j++) out[j] += weight * row[j];
    }
    const total = out.reduce((sum, value) => sum + value, 0);
    return total === 0 ? out : out.map(value => value / total);
}
