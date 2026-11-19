import fixture from '../../../test_data/gaussian_process.json';
import { GaussianProcess } from '../../index';

function closeVector(actual: number[], expected: number[], digits: number): void {
    expect(actual).toHaveLength(expected.length);
    actual.forEach((value, i) => expect(value).toBeCloseTo(expected[i], digits));
}

test.each(fixture.regressions)('GPR matches sklearn fixed-kernel posterior and likelihood %#', row => {
    const model = new GaussianProcess.GaussianProcessRegressor(row.options);
    model.fit(fixture.X, fixture.target);
    const { mean, std } = model.predict(fixture.query, { returnStd: true });
    const { mean: covMean, cov } = model.predict(fixture.query, { returnCov: true });
    expect(mean).toEqual(covMean);
    expect(mean).toEqual(model.predict(fixture.query));
    closeVector(mean, row.mean, 11);
    closeVector(std, row.std, 11);
    expect(cov).toHaveLength(row.cov.length);
    cov.forEach((values, i) => closeVector(values, row.cov[i], 11));
    expect(model.logMarginalLikelihood()).toBeCloseTo(row.lml, 11);
});

test.each(fixture.classifiers)('GPC matches sklearn binary and multiclass probabilities %#', row => {
    const model = new GaussianProcess.GaussianProcessClassifier(row.options);
    model.fit(fixture.X, row.y);
    expect(model.getClasses()).toEqual(row.classes);
    expect(model.predict(fixture.query)).toEqual(row.predicted);
    const probabilities = model.predictProba(fixture.query);
    expect(probabilities).toHaveLength(row.probabilities.length);
    probabilities.forEach((values, i) => closeVector(values, row.probabilities[i], 6));
});
