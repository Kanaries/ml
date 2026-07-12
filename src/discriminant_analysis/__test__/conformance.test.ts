import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { LinearDiscriminantAnalysis } from '../lda';
import { QuadraticDiscriminantAnalysis } from '../qda';

runEstimatorConformance([
    {
        name: 'LinearDiscriminantAnalysis',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new LinearDiscriminantAnalysis({ solver: 'svd' }),
    },
    {
        name: 'QuadraticDiscriminantAnalysis',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new QuadraticDiscriminantAnalysis({ regParam: 0.01 }),
    },
]);
