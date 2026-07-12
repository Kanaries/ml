/**
 * Hand-rolled conformance checks for IsotonicRegression: the shared harness
 * assumes multi-feature `fit(X: number[][], y)`, while isotonic regression is
 * inherently 1-D, so the same contract points (registration, params
 * round-trip, clone, serialize -> revive -> equal state/behavior) are covered
 * here directly.
 */
import { getRegisteredEstimators, loadModel } from '../../base/estimator';
import { IsotonicRegression } from '../isotonicRegression';

const X = [1, 2, 3, 4, 5, 6, 7, 8];
const Y = [0.1, 0.4, 0.3, 0.7, 0.6, 0.6, 0.9, 1.0];
const QUERY = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 9];

describe('IsotonicRegression conformance', () => {
    it('is registered under its declared name', () => {
        expect(getRegisteredEstimators().get('IsotonicRegression')).toBe(IsotonicRegression);
    });

    it('getParams/setParams round-trips and validates keys', () => {
        const est = new IsotonicRegression({ yMin: 0, yMax: 1, increasing: 'auto', outOfBounds: 'nan' });
        const params = est.getParams();
        expect(params).toEqual({ yMin: 0, yMax: 1, increasing: 'auto', outOfBounds: 'nan' });
        est.setParams(params);
        expect(est.getParams()).toEqual(params);
        expect(() => est.setParams({ __definitely_not_a_param__: 1 })).toThrow(/Invalid parameter/);
    });

    it('clone copies params onto a fresh instance', () => {
        const est = new IsotonicRegression({ increasing: false, outOfBounds: 'raise' });
        const copy = est.clone();
        expect(copy).not.toBe(est);
        expect(copy.constructor).toBe(IsotonicRegression);
        expect(copy.getParams()).toEqual(est.getParams());
    });

    it('refitting a clone reproduces identical output', () => {
        const est = new IsotonicRegression();
        est.fit(X, Y);
        const copy = est.clone();
        copy.fit(X, Y);
        expect(copy.predict(QUERY)).toEqual(est.predict(QUERY));
    });

    it('survives serialize -> JSON text -> revive with identical state and behavior', () => {
        const est = new IsotonicRegression({ yMin: 0, yMax: 1 });
        est.fit(X, Y);
        const revived = loadModel(JSON.stringify(est)) as IsotonicRegression;
        expect(revived.constructor).toBe(IsotonicRegression);
        expect(revived.getParams()).toEqual(est.getParams());
        expect(JSON.parse(JSON.stringify(revived))).toEqual(JSON.parse(JSON.stringify(est)));
        expect(revived.predict(QUERY)).toEqual(est.predict(QUERY));
    });

    it('a revived model refits identically and stays serializable', () => {
        const est = new IsotonicRegression();
        est.fit(X, Y);
        const revived = loadModel(JSON.stringify(est)) as IsotonicRegression;
        revived.fit(X, Y);
        expect(revived.predict(QUERY)).toEqual(est.predict(QUERY));
        expect(() => JSON.stringify(revived)).not.toThrow();
    });

    it('setParams after fit resets to a working unfitted estimator', () => {
        const est = new IsotonicRegression();
        est.fit(X, Y);
        est.setParams({});
        expect(() => est.predict(QUERY)).toThrow(/must be fitted/);
        est.fit(X, Y);
        expect(est.predict(QUERY)).toHaveLength(QUERY.length);
    });
});
