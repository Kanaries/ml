import { DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor } from '..';
import { RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor } from '../../ensemble';

const trees = [DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor];

describe.each(trees)('%s constant feature fallback', Tree => {
    test('finds a separating feature after sampling a constant column', () => {
        const X = [[0, 0], [0, 1], [0, 2], [0, 3]], y = [0, 0, 1, 1];
        const model = new Tree({ max_features: 1, randomState: 0 });
        model.fit(X, y);
        expect(model.predict(X)).toEqual(y);
    });

    test('continues splitting when features become constant inside child nodes', () => {
        const X = [[0, 0], [0, 1], [1, 0], [1, 1]], y = [0, 1, 1, 2];
        for (let seed = 0; seed < 8; seed++) {
            const model = new Tree({ max_features: 1, randomState: seed });
            model.fit(X, y);
            expect(model.predict(X)).toEqual(y);
        }
    });

    test('keeps the feature budget when a sampled feature can split', () => {
        const model = new Tree({ max_features: 1, max_depth: 1, randomState: 0 });
        model.fit([[0, 0], [0, 1], [1, 2], [1, 3]], [0, 0, 0, 1]);
        // The first feature is useful, although the second offers a better split.
        expect(model.featureImportances).toEqual([1, 0]);
    });

    test('stops safely when every feature is constant', () => {
        const X = [[1, 1], [1, 1], [1, 1]], y = [0, 1, 0];
        const model = new Tree({ max_features: 1, randomState: 0 });
        model.fit(X, y);
        const predictions = model.predict(X);
        expect(predictions.every(Number.isFinite)).toBe(true);
        expect(new Set(predictions).size).toBe(1);
    });
});

test.each([RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor])(
    '%s learns signal surrounded by constant columns', Forest => {
        const X = Array.from({ length: 12 }, (_, i) => [...new Array(15).fill(0), i]);
        const y = X.map((_, i) => i < 6 ? 0 : 1);
        const model = new Forest({ nEstimators: 10, bootstrap: false, randomState: 0 });
        model.fit(X, y);
        expect(model.predict(X)).toEqual(y);
    },
);
