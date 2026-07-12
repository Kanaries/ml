import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { LogisticRegression } from '../logisticRegression';

runEstimatorConformance([
    {
        name: 'LogisticRegression',
        kind: 'classifier',
        dataset: 'binary',
        create: () => new LogisticRegression({ learningRate: 0.1, maxIter: 200 }),
    },
]);
