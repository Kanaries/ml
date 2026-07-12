import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { BernoulliRBM } from '../bernoulliRBM';
import { MLPClassifier } from '../mlpClassifier';
import { MLPRegressor } from '../mlpRegressor';

runEstimatorConformance([
    {
        name: 'BernoulliRBM',
        kind: 'transformer',
        dataset: 'binaryFeatures',
        create: () => new BernoulliRBM({ nComponents: 4, learningRate: 0.1, batchSize: 5, nIter: 5, randomState: 42 }),
    },
    {
        name: 'MLPClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new MLPClassifier({ hiddenLayerSizes: [8], maxIter: 300, randomState: 42 }),
    },
    {
        name: 'MLPRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new MLPRegressor({ hiddenLayerSizes: [8], maxIter: 300, randomState: 42 }),
    },
]);
