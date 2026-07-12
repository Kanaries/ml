import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { BernoulliNB } from '../bernoulliNB';
import { CategoricalNB } from '../categoricalNB';
import { ComplementNB } from '../complementNB';
import { GaussianNB } from '../gaussianNB';
import { MultinomialNB } from '../multinomialNB';

runEstimatorConformance([
    {
        name: 'GaussianNB',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new GaussianNB(),
    },
    {
        name: 'MultinomialNB',
        kind: 'classifier',
        dataset: 'counts',
        create: () => new MultinomialNB({ alpha: 1.0 }),
    },
    {
        name: 'ComplementNB',
        kind: 'classifier',
        dataset: 'counts',
        create: () => new ComplementNB({ alpha: 1.0 }),
    },
    {
        name: 'BernoulliNB',
        kind: 'classifier',
        dataset: 'binaryFeatures',
        create: () => new BernoulliNB({ alpha: 1.0 }),
    },
    {
        name: 'CategoricalNB',
        kind: 'classifier',
        dataset: 'binaryFeatures',
        create: () => new CategoricalNB({ alpha: 1.0 }),
    },
]);
