import fs from 'fs';
import path from 'path';
import { KernelDensity } from '../kernelDensity';
import { KernelRidge } from '../kernelRidge';

const waveC = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8'));

test('KernelRidge matches sklearn RBF dual solution and predictions', () => {
    const fixture = waveC.kernel_ridge, model = new KernelRidge({ alpha: .5, kernel: 'rbf', gamma: .7 });
    model.fit(fixture.X, fixture.y);
    model.dualCoef.forEach((value, i) => expect(value).toBeCloseTo(fixture.dual_coef[i], 10));
    model.predict(fixture.query).forEach((value, i) => expect(value).toBeCloseTo(fixture.prediction[i], 10));
});

test('KernelRidge falls back to a least-squares solution for singular kernels', () => {
    const model = new KernelRidge({ alpha: 0, kernel: 'linear' }); model.fit([[1], [1]], [1, 1]);
    model.predict([[1], [2]]).forEach((value, i) => expect(value).toBeCloseTo(i + 1, 12));
});

test('KernelDensity matches sklearn normalized radial kernels', () => {
    const fixture = waveC.kernel_density;
    for (const [kernel, bandwidth] of [['gaussian', .6], ['epanechnikov', 1.5]] as const) {
        const model = new KernelDensity({ kernel, bandwidth }); model.fit(fixture.X);
        model.scoreSamples(fixture.query).forEach((value, i) => expect(value).toBeCloseTo(fixture[kernel][i], 7));
    }
    fixture.cosine_dimensions.forEach((dimension: number, i: number) => {
        const model = new KernelDensity({ kernel: 'cosine' }); model.fit([new Array(dimension).fill(0)]);
        expect(model.scoreSamples([new Array(dimension).fill(0)])[0]).toBeCloseTo(fixture.cosine_origin_scores[i], 10);
    });
});
