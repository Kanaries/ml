import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { CalibratedClassifierCV } from '../calibratedClassifierCV';
import { GaussianNBProba } from './testEstimators';

runEstimatorConformance([
    {
        name: 'CalibratedClassifierCV',
        kind: 'classifier',
        dataset: 'multiclass',
        // GaussianNB is deterministic; StratifiedKFold without shuffle is too.
        create: () => new CalibratedClassifierCV({ estimator: new GaussianNBProba(), cv: 3 }),
    },
]);
