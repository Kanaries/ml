import { loadModel } from '../../base';
import { RidgeRegression } from '../../linear';
import { IterativeImputer } from '../iterativeImputer';
import fs from 'fs';
import path from 'path';

const waveB = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_b.json'), 'utf8'));

test('IterativeImputer matches the sklearn Ridge fixture for fit and transform', () => {
    const X = waveB.iterative_imputer.X.map((row: Array<number | null>) => row.map(value => value === null ? NaN : value));
    const model = new IterativeImputer({ maxIter: 20, tol: 1e-8 });
    const transformed = model.fitTransform(X);
    transformed.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(waveB.iterative_imputer.transform[i][j], 12)));
    expect(model.nIter).toBe(waveB.iterative_imputer.n_iter);
    expect(transformed.flat().every(Number.isFinite)).toBe(true);
    const revived = loadModel(JSON.stringify(model)) as IterativeImputer;
    revived.transform([[6, NaN]])[0].forEach((value, j) => expect(value).toBeCloseTo(waveB.iterative_imputer.future[0][j], 12));
});

test('IterativeImputer supports median initialization and bounds', () => {
    const model = new IterativeImputer({ maxIter: 0, initialStrategy: 'median', minValue: 0, maxValue: 10 });
    expect(model.fitTransform([[1, NaN], [3, 5], [100, 7]])).toEqual([[1, 6], [3, 5], [100, 7]]);
});

test('IterativeImputer convergence scale matches sklearn below unit magnitude', () => {
    const fixture = waveB.iterative_imputer.small_scale;
    const X = fixture.X.map((row: Array<number | null>) => row.map(value => value === null ? NaN : value));
    const model = new IterativeImputer({ maxIter: 20, tol: .03, randomState: 0 });
    const transformed = model.fitTransform(X);
    expect(model.nIter).toBe(fixture.n_iter);
    transformed.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(fixture.transform[i][j], 10)));
});

test('IterativeImputer runs to maxIter when sklearn convergence scale is zero', () => {
    const fixture = waveB.iterative_imputer.zero_scale;
    const X = fixture.X.map((row: Array<number | null>) => row.map(value => value === null ? NaN : value));
    const model = new IterativeImputer({ maxIter: 3, tol: 1e-3 });
    const transformed = model.fitTransform(X);
    expect(model.nIter).toBe(fixture.n_iter);
    expect(transformed).toEqual(fixture.transform);
});

test('IterativeImputer convergence uses sklearn matrix infinity norm', () => {
    const fixture = waveB.iterative_imputer.infinity_norm;
    const X = fixture.X.map((row: Array<number | null>) => row.map(value => value === null ? NaN : value));
    const model = new IterativeImputer({ maxIter: 20, tol: .05, randomState: 0 });
    const transformed = model.fitTransform(X);
    expect(model.nIter).toBe(fixture.n_iter);
    transformed.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(fixture.transform[i][j], 10)));
});

test('IterativeImputer descending order reverses feature-index ties like sklearn', () => {
    const fixture = waveB.iterative_imputer.descending_ties;
    const X = fixture.X.map((row: Array<number | null>) => row.map(value => value === null ? NaN : value));
    const model = new IterativeImputer({ maxIter: 2, tol: 0, imputationOrder: 'descending' });
    const transformed = model.fitTransform(X);
    expect(model.nIter).toBe(fixture.n_iter);
    transformed.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(fixture.transform[i][j], 10)));
});

test('IterativeImputer exposes nested estimator params', () => {
    const model = new IterativeImputer();
    model.fit([[0, 1], [1, NaN], [2, 5]]);
    model.setParams({ estimator__alpha: .25 });
    expect((model.getParams().estimator as RidgeRegression).getParams().alpha).toBe(.25);
    expect(() => model.transform([[3, NaN]])).toThrow('not fitted');
});
