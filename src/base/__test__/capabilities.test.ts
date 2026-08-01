import { PolynomialRegression } from '../../linear/polynomialRegression';
import { RidgeRegression } from '../../linear/ridgeRegression';
import { SVC } from '../../svm';
import { DecisionTreeClassifier } from '../../tree';
import {
    declaresEstimatorCapability,
    hasCoefficientCapability,
    hasFeatureImportanceCapability,
} from '../capabilities';

test('capabilities are opt-in rather than BaseEstimator placeholders', () => {
    expect(hasCoefficientCapability(new RidgeRegression())).toBe(true);
    expect(hasFeatureImportanceCapability(new DecisionTreeClassifier())).toBe(true);
    expect(declaresEstimatorCapability(new SVC(), 'coef')).toBe(false);
    expect(declaresEstimatorCapability(new PolynomialRegression(), 'coef')).toBe(false);
});

test('capability getters return defensive copies', () => {
    const ridge = new RidgeRegression({ alpha: 1 });
    ridge.fit([[0], [1], [2]], [0, 1, 2]);
    const first = ridge.coef;
    first[0] = 99;
    expect(ridge.coef[0]).not.toBe(99);

    const tree = new DecisionTreeClassifier({ max_depth: 2, randomState: 0 });
    tree.fit([[0], [1], [2], [3]], [0, 0, 1, 1]);
    const importance = tree.featureImportances;
    importance[0] = 0;
    expect(tree.featureImportances[0]).toBe(1);
});
