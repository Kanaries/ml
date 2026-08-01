import fs from 'fs';
import path from 'path';
import { loadModel } from '../../base';
import { LinearRegression } from '../../linear';
import { FunctionTransformer } from '../../utils/preprocessingExtra';
import { TransformedTargetRegressor } from '../transformedTargetRegressor';

const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8')).advanced.transformed_target;

describe('TransformedTargetRegressor', () => {
    test('fits in transformed space and inverts predictions', () => {
        const model = new TransformedTargetRegressor({
            regressor: new LinearRegression(),
            transformer: new FunctionTransformer({ func: 'log1p', inverseFunc: 'expm1' }),
        });
        model.fit([[0], [1], [2]], [1, 3, 7]);
        expect(model.predict([[3]])[0]).toBeCloseTo(fixture.prediction[0], 10);
        const revived = loadModel(JSON.stringify(model)) as TransformedTargetRegressor;
        expect(revived.predict([[3]])[0]).toBeCloseTo(fixture.prediction[0], 10);
    });
    test('supports nested estimator parameters', () => {
        const model = new TransformedTargetRegressor();
        expect(() => model.setParams({ regressor__missing: 1 })).toThrow(/Invalid parameter/);
        expect(() => model.setParams({ func: 'log1p' })).toThrow(/set together/);
        expect(() => model.setParams({ inverseFunc: 'expm1' })).toThrow(/set together/);
        expect(model.clone()).not.toBe(model);
    });
    test('supports sklearn-style func/inverseFunc configuration', () => {
        expect(() => new TransformedTargetRegressor({ func: 'log1p' })).toThrow(/provided together/);
        expect(() => new TransformedTargetRegressor({ inverseFunc: 'expm1' })).toThrow(/provided together/);
        const model = new TransformedTargetRegressor({ regressor: new LinearRegression(), func: 'log1p', inverseFunc: 'expm1' });
        model.fit([[0], [1], [2]], [1, 3, 7]); expect(model.predict([[3]])[0]).toBeCloseTo(fixture.prediction[0], 10); expect(model.score([[0], [1], [2]], [1, 3, 7])).toBeCloseTo(1, 12);
    });
});
