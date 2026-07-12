import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { LogisticRegression } from '../../linear/logisticRegression';
import { OneVsRestClassifier } from '../oneVsRest';
import { OneVsOneClassifier } from '../oneVsOne';

runEstimatorConformance([
    {
        name: 'OneVsRestClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new OneVsRestClassifier({ estimator: new LogisticRegression({ learningRate: 0.1, maxIter: 200 }) }),
    },
    {
        name: 'OneVsOneClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new OneVsOneClassifier({ estimator: new LogisticRegression({ learningRate: 0.1, maxIter: 200 }) }),
    },
]);
