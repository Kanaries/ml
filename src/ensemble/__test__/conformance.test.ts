import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { AdaBoostClassifier } from '../adaBoostClassifier';
import { AdaBoostRegressor } from '../adaBoostRegressor';
import { BaggingClassifier } from '../baggingClassifier';
import { GradientBoostingClassifier } from '../gradientBoostingClassifier';
import { GradientBoostingRegressor } from '../gradientBoostingRegressor';
import { IsolationForest } from '../isolationForest';
import { RandomForestClassifier } from '../randomForestClassifier';
import { RandomForestRegressor } from '../randomForestRegressor';
import { XGBoostClassifier, XGBoostRegressor } from '../xgboost';
import { VotingClassifier, VotingRegressor } from '../voting';
import { StackingClassifier, StackingRegressor } from '../stacking';
import { LogisticRegression } from '../../linear/logisticRegression';
import { LinearRegression } from '../../linear/linearRegression';
import { RidgeRegression } from '../../linear/ridgeRegression';
import { GaussianNB } from '../../bayes/gaussianNB';
import { DecisionTreeClassifier } from '../../tree/decisionTreeClassifier';
import { DecisionTreeRegressor } from '../../tree/decisionTreeRegressor';

runEstimatorConformance([
    {
        name: 'RandomForestClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new RandomForestClassifier({ nEstimators: 5, max_depth: 5, randomState: 42 }),
    },
    {
        name: 'RandomForestRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new RandomForestRegressor({ nEstimators: 5, maxDepth: 5, randomState: 42 }),
    },
    {
        name: 'BaggingClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new BaggingClassifier({ nEstimators: 5, max_depth: 5, randomState: 42 }),
    },
    {
        // deterministic stump search — the estimator has no randomness at all
        name: 'AdaBoostClassifier',
        kind: 'classifier',
        dataset: 'multiclass', // SAMME multiclass support
        create: () => new AdaBoostClassifier({ nEstimators: 10 }),
    },
    {
        name: 'AdaBoostRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new AdaBoostRegressor({ n_estimators: 5, randomState: 42 }),
    },
    {
        name: 'GradientBoostingClassifier',
        kind: 'classifier',
        dataset: 'multiclass', // multinomial deviance multiclass support
        create: () => new GradientBoostingClassifier({ nEstimators: 5, maxDepth: 3, randomState: 42 }),
    },
    {
        name: 'GradientBoostingRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new GradientBoostingRegressor({ nEstimators: 5, maxDepth: 3, randomState: 42 }),
    },
    {
        name: 'IsolationForest',
        kind: 'outlier',
        dataset: 'blobs',
        create: () => new IsolationForest({ subsampling_size: 16, tree_num: 10, random_state: 42 }),
    },
    {
        name: 'XGBoostClassifier',
        kind: 'classifier',
        dataset: 'multiclass', // multi:softprob multiclass support
        create: () => new XGBoostClassifier({ nEstimators: 5, maxDepth: 3, randomState: 42 }),
    },
    {
        name: 'XGBoostRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new XGBoostRegressor({ nEstimators: 5, maxDepth: 3, randomState: 42 }),
    },
    {
        // this library's LogisticRegression is binary-only → binary dataset
        name: 'VotingClassifier',
        kind: 'classifier',
        dataset: 'binary',
        create: () => new VotingClassifier({
            estimators: [
                ['lr', new LogisticRegression({ learningRate: 0.2, maxIter: 200 })],
                ['nb', new GaussianNB()],
            ],
            voting: 'hard',
        }),
    },
    {
        name: 'VotingRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new VotingRegressor({
            estimators: [
                ['ridge', new RidgeRegression({ alpha: 0.1 })],
                ['lin', new LinearRegression()],
            ],
        }),
    },
    {
        // default finalEstimator (LogisticRegression) is binary-only → binary dataset
        name: 'StackingClassifier',
        kind: 'classifier',
        dataset: 'binary',
        create: () => new StackingClassifier({
            estimators: [
                ['nb', new GaussianNB()],
                ['dt', new DecisionTreeClassifier({ max_depth: 4, randomState: 42 })],
            ],
            cv: 3,
        }),
    },
    {
        name: 'StackingRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new StackingRegressor({
            estimators: [
                ['ridge', new RidgeRegression({ alpha: 0.1 })],
                ['dt', new DecisionTreeRegressor({ max_depth: 4, randomState: 42 })],
            ],
            cv: 3,
        }),
    },
]);
