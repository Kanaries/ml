import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { BernoulliRBM } from '../bernoulliRBM';

runEstimatorConformance([
    {
        name: 'BernoulliRBM',
        kind: 'transformer',
        dataset: 'binaryFeatures',
        create: () => new BernoulliRBM({ nComponents: 4, learningRate: 0.1, batchSize: 5, nIter: 5, randomState: 42 }),
    },
]);
