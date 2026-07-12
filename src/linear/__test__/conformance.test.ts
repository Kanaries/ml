import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { LogisticRegression } from '../logisticRegression';
import { LinearRegression } from '../linearRegression';
import { PolynomialRegression } from '../polynomialRegression';
import { RidgeRegression } from '../ridgeRegression';
import { LassoRegression } from '../lassoRegression';
import { Ridge } from '../ridge';
import { Lasso } from '../lasso';
import { ElasticNet } from '../elasticNet';
import { RidgeClassifier } from '../ridgeClassifier';
import { SGDClassifier } from '../sgdClassifier';
import { SGDRegressor } from '../sgdRegressor';
import { Perceptron } from '../perceptron';

runEstimatorConformance([
    {
        name: 'LogisticRegression',
        kind: 'classifier',
        dataset: 'binary',
        create: () => new LogisticRegression({ learningRate: 0.1, maxIter: 200 }),
    },
    {
        name: 'LinearRegression',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new LinearRegression(),
    },
    {
        name: 'PolynomialRegression',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new PolynomialRegression({ degree: 2 }),
    },
    {
        name: 'RidgeRegression',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new RidgeRegression({ alpha: 1, fitIntercept: true }),
    },
    {
        name: 'Ridge',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new Ridge({ alpha: 0.5, fitIntercept: true }),
    },
    {
        name: 'LassoRegression',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new LassoRegression({ alpha: 0.1, fitIntercept: true, maxIter: 500, tol: 1e-6 }),
    },
    {
        name: 'Lasso',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new Lasso({ alpha: 0.05, fitIntercept: true, maxIter: 500, tol: 1e-6 }),
    },
    {
        name: 'ElasticNet',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new ElasticNet({ alpha: 0.1, l1Ratio: 0.5, fitIntercept: true, maxIter: 500, tol: 1e-6 }),
    },
    {
        name: 'RidgeClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new RidgeClassifier({ alpha: 1, fitIntercept: true }),
    },
    {
        name: 'SGDClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new SGDClassifier({ loss: 'hinge', maxIter: 200, randomState: 42 }),
    },
    {
        name: 'SGDRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new SGDRegressor({ maxIter: 200, randomState: 42 }),
    },
    {
        name: 'Perceptron',
        kind: 'classifier',
        dataset: 'binary',
        create: () => new Perceptron({ maxIter: 200, randomState: 42 }),
    },
]);
