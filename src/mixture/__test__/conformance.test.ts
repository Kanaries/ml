import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { blobsDataset } from '../../__test__/conformance/datasets';
import { loadModel } from '../../base/estimator';
import { GaussianMixture } from '../gaussianMixture';
import { BayesianGaussianMixture } from '../bayesianGaussianMixture';

runEstimatorConformance([
    {
        name: 'GaussianMixture',
        kind: 'cluster',
        dataset: 'blobs',
        create: () => new GaussianMixture({ nComponents: 3, randomState: 42 }),
    },
    {
        name: 'BayesianGaussianMixture',
        kind: 'cluster',
        dataset: 'blobs',
        create: () => new BayesianGaussianMixture({ nComponents: 3, randomState: 42 }),
    },
]);

// The cluster harness only exercises fitPredict; mixtures also expose
// out-of-sample inference, so check serialize → revive → parity for
// predict / predictProba / scoreSamples explicitly.
describe.each([
    ['GaussianMixture', () => new GaussianMixture({ nComponents: 3, randomState: 42 })],
    ['BayesianGaussianMixture', () => new BayesianGaussianMixture({ nComponents: 3, randomState: 42 })],
] as const)('%s serialized inference parity', (name, create) => {
    const { X } = blobsDataset();

    it('predict / predictProba / scoreSamples survive serialize → revive unchanged', () => {
        const est = create();
        est.fit(X);
        const revived = loadModel(JSON.stringify(est)) as GaussianMixture | BayesianGaussianMixture;
        expect(revived.constructor.name).toBe(name);
        expect(revived.predict(X)).toEqual(est.predict(X));
        expect(revived.predictProba(X)).toEqual(est.predictProba(X));
        expect(revived.scoreSamples(X)).toEqual(est.scoreSamples(X));
        // fitted accessors and flags survive too
        expect(revived.getMeans()).toEqual(est.getMeans());
        expect(revived.getWeights()).toEqual(est.getWeights());
        expect(revived.getCovariances()).toEqual(est.getCovariances());
        expect(revived.converged).toBe(est.converged);
        expect(revived.nIter).toBe(est.nIter);
    });
});

describe('GaussianMixture serialized aic/bic parity', () => {
    const { X } = blobsDataset();

    it('aic/bic of a revived model match the original', () => {
        const est = new GaussianMixture({ nComponents: 3, randomState: 42 }).fit(X);
        const revived = GaussianMixture.fromJSON(JSON.stringify(est));
        expect(revived.aic(X)).toBe(est.aic(X));
        expect(revived.bic(X)).toBe(est.bic(X));
        expect(revived.score(X)).toBe(est.score(X));
    });
});
