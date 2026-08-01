import { averageImportances, weightedImportances } from '../featureImportances';
import { DecisionTreeClassifier } from '../decisionTreeClassifier';
import { DecisionTreeRegressor } from '../decisionTreeRegressor';
import { loadModel, type SerializedModel } from '../../base';
import { RandomForestClassifier, RandomForestRegressor, XGBoostRegressor } from '../../ensemble';

const X = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]];
const y = [0, 0, 0, 1, 1, 1, 1, 1];

test.each([
    ['gini', [0.7333333333333333, 0.26666666666666666]],
    ['entropy', [0.7380646548280465, 0.26193534517195355]],
] as const)('DecisionTreeClassifier %s feature importances match sklearn 1.5', (criterion, expected) => {
    const tree = new DecisionTreeClassifier({ criterion, max_depth: 3, randomState: 0 });
    tree.fit(X, y);
    expect(tree.featureImportances[0]).toBeCloseTo(expected[0], 12);
    expect(tree.featureImportances[1]).toBeCloseTo(expected[1], 12);
});

test('DecisionTreeRegressor feature importances match sklearn 1.5', () => {
    const tree = new DecisionTreeRegressor({ max_depth: 3, randomState: 0 });
    tree.fit(X, [0, 0, 1, 3, 4, 5, 7, 8]);
    expect(tree.featureImportances[0]).toBeCloseTo(0.9545454545454546, 12);
    expect(tree.featureImportances[1]).toBeCloseTo(0.045454545454545456, 12);
});

test('RandomForest averages normalized per-tree importances like sklearn', () => {
    const forestX = Array.from({ length: 30 }, (_, i) => [i % 7, (i * i) % 11, i]);
    const classifier = new RandomForestClassifier({ nEstimators: 7, max_depth: 3, randomState: 2 });
    classifier.fit(forestX, forestX.map(row => row[0] + row[1] > 7 ? 1 : 0));
    const classifierExpected = [0.5061013879863517, 0.36156612534619076, 0.13233248666745742];
    classifier.featureImportances.forEach((value, i) => expect(value).toBeCloseTo(classifierExpected[i], 14));

    const regressor = new RandomForestRegressor({ nEstimators: 7, maxDepth: 3, randomState: 2 });
    regressor.fit(forestX, forestX.map((row, i) => 3 * row[0] - 2 * row[1] + (i % 3) * 5));
    const regressorExpected = [0.4900398899362749, 0.2817852217022072, 0.22817488836151795];
    regressor.featureImportances.forEach((value, i) => expect(value).toBeCloseTo(regressorExpected[i], 14));
});

test('raw and weighted ensemble aggregations normalize only once', () => {
    expect(averageImportances([[9, 1], [1, 1]])).toEqual([10 / 12, 2 / 12]);
    const weighted = weightedImportances([[0.9, 0.1], [0.2, 0.8]], [1, 3]);
    expect(weighted[0]).toBeCloseTo(1.5 / 4, 14);
    expect(weighted[1]).toBeCloseTo(2.5 / 4, 14);
});

test('feature importances throw before fitting', () => {
    expect(() => new DecisionTreeClassifier().featureImportances).toThrow(/fitted/);
});

test('legacy trees predict but reject unavailable importance metadata', () => {
    const tree = new DecisionTreeClassifier({ max_depth: 3, randomState: 0 });
    tree.fit(X, y);
    const legacy = JSON.parse(JSON.stringify(tree.toJSON())) as SerializedModel;
    const removeImportance = (value: unknown): void => {
        if (value === null || typeof value !== 'object') return;
        if (Array.isArray(value)) {
            value.forEach(removeImportance);
            return;
        }
        const record = value as Record<string, unknown>;
        delete record.weightedImpurityDecrease;
        Object.values(record).forEach(removeImportance);
    };
    removeImportance(legacy.state);
    const revived = loadModel(legacy) as DecisionTreeClassifier;
    expect(revived.predict(X)).toEqual(tree.predict(X));
    expect(() => revived.featureImportances).toThrow(/legacy tree model.*refit/);
});

test('legacy XGBoost models do not return a truncated importance vector', () => {
    const model = new XGBoostRegressor({ nEstimators: 2, maxDepth: 2, randomState: 0 });
    model.fit(X, y);
    const legacy = JSON.parse(JSON.stringify(model.toJSON())) as SerializedModel;
    delete (legacy.state as Record<string, unknown>).nFeatures;
    const revived = loadModel(legacy) as XGBoostRegressor;
    expect(revived.predict(X)).toEqual(model.predict(X));
    expect(() => revived.featureImportances).toThrow(/legacy XGBoost model.*refit/);
});
