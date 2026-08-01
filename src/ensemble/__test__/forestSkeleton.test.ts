import { ExtraTreeClassifier } from '../../tree';
import { fitForest, predictForestClassification } from '../forest';

test('forest skeleton accepts an injected base-estimator family', () => {
    const X = [[0], [0.1], [0.2], [2], [2.1], [2.2]];
    const y = [0, 0, 0, 1, 1, 1];
    const estimators = fitForest(X, y, {
        nEstimators: 5,
        bootstrap: false,
        randomState: 7,
        createEstimator: randomState => new ExtraTreeClassifier({ max_depth: 3, randomState }),
    });

    expect(estimators).toHaveLength(5);
    expect(estimators.every(tree => tree instanceof ExtraTreeClassifier)).toBe(true);
    expect(predictForestClassification(estimators, X)).toEqual(y);
});
