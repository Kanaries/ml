import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { DecisionTreeClassifier } from '../decisionTreeClassifier';
import { DecisionTreeRegressor } from '../decisionTreeRegressor';
import { ExtraTreeClassifier } from '../extraTreeClassifier';
import { ExtraTreeRegressor } from '../extraTreeRegressor';

runEstimatorConformance([
    {
        name: 'DecisionTreeClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new DecisionTreeClassifier({ max_depth: 5, randomState: 42 }),
    },
    {
        name: 'DecisionTreeRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new DecisionTreeRegressor({ max_depth: 5, randomState: 42 }),
    },
    {
        name: 'ExtraTreeClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        // random splits are fully driven by the seeded generator
        create: () => new ExtraTreeClassifier({ max_depth: 5, randomState: 42 }),
    },
    {
        name: 'ExtraTreeRegressor',
        kind: 'regressor',
        dataset: 'regression',
        // random splits are fully driven by the seeded generator
        create: () => new ExtraTreeRegressor({ max_depth: 5, randomState: 42 }),
    },
]);
