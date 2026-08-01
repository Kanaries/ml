import fs from 'fs';
import path from 'path';
import { ARDRegression, BayesianRidge } from '../bayesianRidge';
import { GammaRegressor, PoissonRegressor, TweedieRegressor } from '../glm';
import { HuberRegressor, QuantileRegressor, RANSACRegressor, TheilSenRegressor } from '../robustRegressors';

const waveC = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8'));

test('BayesianRidge posterior mean, precisions, and predictive uncertainty match sklearn', () => {
    const fixture = waveC.linear_models, expected = fixture.bayesian_ridge, model = new BayesianRidge({ maxIter: 500, tol: 1e-8 }); model.fit(fixture.X, fixture.y);
    model.coef.forEach((value, i) => expect(value).toBeCloseTo(expected.coef[i], 8)); expect(model.intercept).toBeCloseTo(expected.intercept, 8); expect(model.alpha).toBeCloseTo(expected.alpha, 5); expect(model.lambda).toBeCloseTo(expected.lambda, 7);
    const rows = fixture.rows.map((i: number) => fixture.X[i]); model.predict(rows).forEach((value, i) => expect(value).toBeCloseTo(expected.prediction[i], 8)); model.predictStd(rows).forEach((value, i) => expect(value).toBeCloseTo(expected.std[i], 7));
});

test('ARDRegression prunes the same irrelevant feature as sklearn', () => {
    const fixture = waveC.linear_models, expected = fixture.ard, model = new ARDRegression({ maxIter: 500, tol: 1e-8 }); model.fit(fixture.X, fixture.y);
    model.coef.forEach((value, i) => expect(value).toBeCloseTo(expected.coef[i], 6)); expect(model.intercept).toBeCloseTo(expected.intercept, 6); expect(model.coef[3]).toBe(0);
});

test('Huber, RANSAC, Theil-Sen, and quantile robust fits track sklearn fixtures', () => {
    const fixture = waveC.linear_models.robust, X = fixture.X, y = fixture.y;
    const huber = new HuberRegressor({ epsilon: 1.35, alpha: .0001, maxIter: 500, tol: 1e-8 }); huber.fit(X, y); huber.coef.forEach((value, i) => expect(value).toBeCloseTo(fixture.huber.coef[i], 2)); expect(huber.intercept).toBeCloseTo(fixture.huber.intercept, 2);
    expect(huber.scale).toBeCloseTo(fixture.huber.scale, 2); expect(huber.outliers).toEqual(fixture.huber.outliers);
    const rows = fixture.rows.map((i: number) => X[i]);
    const ransac = new RANSACRegressor({ minSamples: 3, residualThreshold: .5, maxTrials: 100, randomState: 0 }); ransac.fit(X, y); expect(ransac.inlierMask).toEqual(fixture.ransac.inliers); const fitted = ransac.estimatorFitted as any; fitted.coef.forEach((value: number, i: number) => expect(value).toBeCloseTo(fixture.ransac.coef[i], 8)); expect(fitted.intercept).toBeCloseTo(fixture.ransac.intercept, 8); ransac.predict(rows).forEach((value, i) => expect(value).toBeCloseTo(fixture.ransac.prediction[i], 8));
    const theil = new TheilSenRegressor({ maxSubpopulation: 10000, maxIter: 500, tol: 1e-6, randomState: 0 }); theil.fit(X.slice(0, 12), y.slice(0, 12)); theil.coef.forEach((value, i) => expect(value).toBeCloseTo(fixture.theil_sen.coef[i], 2)); expect(theil.intercept).toBeCloseTo(fixture.theil_sen.intercept, 2);
    theil.predict(rows).forEach((value, i) => expect(value).toBeCloseTo(fixture.theil_sen.prediction[i], 2));
    const quantile = new QuantileRegressor({ quantile: .5, alpha: 0 }); quantile.fit(X, y); quantile.coef.forEach((value, i) => expect(value).toBeCloseTo(fixture.quantile.coef[i], 2)); expect(quantile.intercept).toBeCloseTo(fixture.quantile.intercept, 2); quantile.predict(rows).forEach((value, i) => expect(value).toBeCloseTo(fixture.quantile.prediction[i], 2));
    const regularized = new QuantileRegressor({ quantile: .5, alpha: .1 }); regularized.fit(X, y); regularized.coef.forEach((value, i) => expect(value).toBeCloseTo(fixture.quantile_regularized.coef[i], 2)); expect(regularized.intercept).toBeCloseTo(fixture.quantile_regularized.intercept, 2);
});

test('Poisson, Gamma, and Tweedie Newton solutions match sklearn', () => {
    const fixture = waveC.linear_models.glm, rows = fixture.rows.map((i: number) => fixture.X[i]);
    const cases = [
        { model: new PoissonRegressor({ alpha: .1, maxIter: 500, tol: 1e-10 }), y: fixture.poisson_y, expected: fixture.poisson },
        { model: new GammaRegressor({ alpha: .1, maxIter: 500, tol: 1e-10 }), y: fixture.gamma_y, expected: fixture.gamma },
        { model: new TweedieRegressor({ power: 1.5, alpha: .1, link: 'log', maxIter: 500, tol: 1e-10 }), y: fixture.gamma_y, expected: fixture.tweedie },
    ];
    for (const { model, y, expected } of cases) { model.fit(fixture.X, y); model.coef.forEach((value, i) => expect(value).toBeCloseTo(expected.coef[i], 6)); expect(model.intercept).toBeCloseTo(expected.intercept, 6); model.predict(rows).forEach((value, i) => expect(value).toBeCloseTo(expected.prediction[i], 6)); }
    const identity = new TweedieRegressor({ power: 1.5, alpha: .1, link: 'identity', maxIter: 500, tol: 1e-10 }); identity.fit(fixture.X, fixture.gamma_y);
    identity.coef.forEach((value, i) => expect(value).toBeCloseTo(fixture.tweedie_identity.coef[i], 6)); expect(identity.intercept).toBeCloseTo(fixture.tweedie_identity.intercept, 6); identity.predict(rows).forEach((value, i) => expect(value).toBeCloseTo(fixture.tweedie_identity.prediction[i], 6));
});
