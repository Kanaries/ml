import fs from 'fs';
import path from 'path';
import { EmpiricalCovariance } from '../empiricalCovariance';
import { GraphicalLasso } from '../graphicalLasso';
import { LedoitWolf, OAS, ShrunkCovariance } from '../shrunkCovariance';

const waveC = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8'));
const closeMatrix = (actual: number[][], expected: number[][], digits: number) => actual.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(expected[i][j], digits)));

test('EmpiricalCovariance and fixed shrinkage match sklearn MLE covariance', () => {
    const fixture = waveC.covariance, empirical = new EmpiricalCovariance(); empirical.fit(fixture.X);
    empirical.location.forEach((value, i) => expect(value).toBeCloseTo(fixture.empirical.location[i], 12));
    closeMatrix(empirical.covariance, fixture.empirical.covariance, 12); closeMatrix(empirical.precision, fixture.empirical.precision, 9);
    const shrunk = new ShrunkCovariance({ shrinkage: .2 }); shrunk.fit(fixture.X); closeMatrix(shrunk.covariance, fixture.shrunk.covariance, 12);
});

test('singular covariance score follows sklearn negative-infinity likelihood', () => {
    const model = new EmpiricalCovariance(); model.fit([[0, 0], [1, 1], [2, 2]]); expect(model.score([[1, 1]])).toBe(-Infinity);
});

test('LedoitWolf and OAS calculate sklearn shrinkage coefficients', () => {
    const fixture = waveC.covariance;
    const ledoit = new LedoitWolf(); ledoit.fit(fixture.X); expect(ledoit.shrinkageValue).toBeCloseTo(fixture.ledoit_wolf.shrinkage, 10); closeMatrix(ledoit.covariance, fixture.ledoit_wolf.covariance, 10);
    const oas = new OAS(); oas.fit(fixture.X); expect(oas.shrinkageValue).toBeCloseTo(fixture.oas.shrinkage, 10); closeMatrix(oas.covariance, fixture.oas.covariance, 10);
});

test('GraphicalLasso coordinate descent matches sklearn sparse precision solution', () => {
    const fixture = waveC.covariance, model = new GraphicalLasso({ alpha: .1, maxIter: 1000, tol: 1e-7, enetTol: 1e-8 }); model.fit(fixture.X);
    closeMatrix(model.precision, fixture.graphical_lasso.precision, 3); closeMatrix(model.covariance, fixture.graphical_lasso.covariance, 3);
    expect(model.nIter).toBeLessThanOrEqual(1000);
    const product = model.covariance.map(row => model.precision[0].map((_, j) => row.reduce((sum, value, k) => sum + value * model.precision[k][j], 0)));
    product.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(i === j ? 1 : 0, 6)));
});
