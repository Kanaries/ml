import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { SVC } from '../svc';
import { NuSVC } from '../nuSVC';
import { LinearSVC } from '../linearSVC';
import { LinearSVR } from '../linearSVR';

runEstimatorConformance([
    {
        name: 'SVC',
        kind: 'classifier',
        dataset: 'binary',
        create: () => new SVC({ kernel: 'rbf', C: 1, maxIter: 1000 }),
    },
    {
        // one-vs-one multiclass path
        name: 'SVC',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new SVC({ kernel: 'linear', C: 1, maxIter: 1000 }),
    },
    {
        name: 'NuSVC',
        kind: 'classifier',
        dataset: 'binary',
        create: () => new NuSVC({ nu: 0.5, kernel: 'rbf', maxIter: 1000 }),
    },
    {
        name: 'NuSVC',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new NuSVC({ nu: 0.5, kernel: 'rbf', maxIter: 1000 }),
    },
    {
        name: 'LinearSVC',
        kind: 'classifier',
        dataset: 'binary',
        create: () => new LinearSVC({ C: 1, maxIter: 100, randomState: 42 }),
    },
    {
        name: 'LinearSVC',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new LinearSVC({ C: 1, maxIter: 100, randomState: 42 }),
    },
    {
        name: 'LinearSVR',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new LinearSVR({ C: 10, maxIter: 200, randomState: 42 }),
    },
]);
