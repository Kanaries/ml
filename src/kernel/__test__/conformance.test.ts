import { getRegisteredEstimators, loadModel } from '../../base';
import { KernelDensity } from '../kernelDensity';
import { KernelRidge } from '../kernelRidge';
import { runEstimatorConformance } from '../../__test__/conformance/harness';

runEstimatorConformance([{ name: 'KernelRidge', kind: 'regressor', dataset: 'regression', create: () => new KernelRidge({ alpha: .5, kernel: 'rbf' }) }]);

test('KernelDensity registration, cloning, refit, and serialization conformance', () => {
    const X = [[-1], [0], [1]], model = new KernelDensity({ bandwidth: .5 }); expect(getRegisteredEstimators().has('KernelDensity')).toBe(true);
    model.fit(X); const expected = model.scoreSamples(X), revived = loadModel(JSON.stringify(model)) as KernelDensity; expect(revived.scoreSamples(X)).toEqual(expected);
    const clone = model.clone() as KernelDensity; expect(clone.getParams()).toEqual(model.getParams()); expect(() => clone.scoreSamples(X)).toThrow('not fitted'); clone.fit(X); expect(clone.scoreSamples(X)).toEqual(expected);
});
