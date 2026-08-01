import fs from 'fs';
import path from 'path';
import { CCA, PLSRegression } from '../pls';

const waveC = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8'));
const closeMatrix = (actual: number[][], expected: number[][], digits = 7) => actual.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(expected[i][j], digits)));

test('PLSRegression NIPALS weights, scores, and prediction match sklearn', () => {
    const fixture = waveC.cross_decomposition, model = new PLSRegression({ nComponents: 2, tol: 1e-10 }); model.fit(fixture.X, fixture.Y);
    closeMatrix(model.xWeights, fixture.pls.x_weights, 7); closeMatrix(model.xLoadings, fixture.pls.x_loadings, 7);
    const rows = fixture.rows.map((i: number) => fixture.X[i]); closeMatrix(model.transform(rows), fixture.pls.transform, 7); closeMatrix(model.predict(rows) as number[][], fixture.pls.prediction, 7);
    expect(model.score(fixture.X, fixture.Y)).toBeCloseTo(fixture.pls.score, 12);
});

test('CCA mode-B canonical deflation matches sklearn', () => {
    const fixture = waveC.cross_decomposition, model = new CCA({ nComponents: 2, tol: 1e-10, maxIter: 1000 }); model.fit(fixture.X, fixture.Y);
    closeMatrix(model.xWeights, fixture.cca.x_weights, 6); closeMatrix(model.xLoadings, fixture.cca.x_loadings, 6);
    const rows = fixture.rows.map((i: number) => fixture.X[i]); closeMatrix(model.transform(rows), fixture.cca.transform, 6); closeMatrix(model.predict(rows) as number[][], fixture.cca.prediction, 6);
    expect(model.score(fixture.X, fixture.Y)).toBeCloseTo(fixture.cca.score, 12);
});
