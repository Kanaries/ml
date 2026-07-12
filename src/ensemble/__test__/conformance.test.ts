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
]);
