import { LinearRegression } from '../../linear/linearRegression';
import { LogisticRegression } from '../../linear/logisticRegression';
import { RidgeRegression } from '../../linear/ridgeRegression';
import { loadModel, type SerializedModel } from '../estimator';

function renameStateKey(model: SerializedModel, currentKey: string, legacyKey: string): SerializedModel {
    const payload = JSON.parse(JSON.stringify(model)) as SerializedModel;
    const state = payload.state as Record<string, unknown>;
    state[legacyKey] = state[currentKey];
    delete state[currentKey];
    return payload;
}

describe('format-v1 fitted-state migrations', () => {
    it('revives LogisticRegression models serialized with the legacy weights field', () => {
        const model = new LogisticRegression({ learningRate: 0.2, maxIter: 200 });
        const X = [[-2], [-1], [1], [2]];
        const y = [0, 0, 1, 1];
        model.fit(X, y);
        const legacy = renameStateKey(model.toJSON(), 'coefState', 'weights');
        const revived = loadModel(legacy) as LogisticRegression;
        expect(revived.predict(X)).toEqual(model.predict(X));
        expect(revived.coef).toEqual(model.coef);
    });

    it.each([
        ['RidgeRegression', () => new RidgeRegression({ alpha: 1 })],
        ['LinearRegression', () => new LinearRegression()],
    ] as const)('revives %s models serialized with the legacy coef field', (_name, create) => {
        const model = create();
        const X = [[0], [1], [2], [3]];
        const y = [1, 3, 5, 7];
        model.fit(X, y);
        const legacy = renameStateKey(model.toJSON(), 'coefState', 'coef');
        const revived = loadModel(legacy) as RidgeRegression | LinearRegression;
        expect(revived.predict(X)).toEqual(model.predict(X));
        expect(revived.coef).toEqual(model.coef);
    });
});
