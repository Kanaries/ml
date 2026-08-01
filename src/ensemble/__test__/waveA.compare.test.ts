import { BaggingRegressor } from '../baggingRegressor';
import { ExtraTreesClassifier } from '../extraTreesClassifier';
import { ExtraTreesRegressor } from '../extraTreesRegressor';

function r2Score(y: number[], predictions: number[]): number {
    const mean = y.reduce((sum, value) => sum + value, 0) / y.length;
    const residual = y.reduce((sum, value, i) => sum + (value - predictions[i]) ** 2, 0);
    const total = y.reduce((sum, value) => sum + (value - mean) ** 2, 0);
    return 1 - residual / total;
}

const X = Array.from({ length: 80 }, (_, i) => {
    const x0 = -2 + 4 * i / 79;
    return [x0, Math.sin(i * 1.7), (i % 7) / 6];
});
const regressionTarget = X.map(row => 2 * row[0] - .5 * row[1] + row[2] ** 2);
const classificationTarget = X.map(row => row[0] + .3 * row[1] > 0 ? 1 : 0);

test('BaggingRegressor learns the sklearn-style nonlinear regression fixture deterministically', () => {
    const create = () => new BaggingRegressor({ nEstimators: 30, maxSamples: .8, randomState: 42 });
    const first = create();
    const second = create();
    first.fit(X, regressionTarget);
    second.fit(X, regressionTarget);
    const predictions = first.predict(X);
    expect(predictions).toEqual(second.predict(X));
    expect(r2Score(regressionTarget, predictions)).toBeGreaterThan(.95);
});

test('ExtraTreesClassifier fits the parity fixture and exposes normalized importances', () => {
    const model = new ExtraTreesClassifier({ nEstimators: 40, max_features: 'all', randomState: 42 });
    model.fit(X, classificationTarget);
    const predictions = model.predict(X);
    const accuracy = predictions.filter((value, i) => value === classificationTarget[i]).length / X.length;
    expect(accuracy).toBeGreaterThan(.98);
    expect(model.featureImportances).toHaveLength(3);
    expect(model.featureImportances.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
    expect(model.getParams().max_features).toBe('all');
});

test('ExtraTreesRegressor fits the parity fixture and exposes normalized importances', () => {
    const model = new ExtraTreesRegressor({ nEstimators: 40, max_features: 'all', randomState: 42 });
    model.fit(X, regressionTarget);
    expect(r2Score(regressionTarget, model.predict(X))).toBeGreaterThan(.98);
    expect(model.featureImportances).toHaveLength(3);
    expect(model.featureImportances.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
});

test('ExtraTreesClassifier averages leaf probabilities instead of hard tree votes', () => {
    const shallowX = [
        [-0.5094835390425134, -0.8898288746269689], [0.6461189348680143, -0.6790589044606862],
        [-0.9429941307222416, -0.9023398430108464], [0.37427340482759597, 0.4131198343011735],
        [-0.6949403078248169, 0.13825965313078803], [-0.27000308713876, 0.058124397935461536],
        [0.8967634727216918, -0.09631315907120064], [-0.7352559303261861, 0.553592588336759],
        [0.23063566929719936, 0.2936998990305699], [0.21420853418690022, 0.20284022875394703],
        [-0.8642690944152597, 0.22934475283077438], [0.5972668580685434, 0.264086709603748],
        [0.5053340694916753, -0.8502901828384867], [-0.8270884862421904, -0.8761739738994967],
        [0.1440353543907733, 0.8022079447304904], [0.7089286332101827, 0.9635406415569974],
        [0.22756293344084444, 0.6502283854877842], [0.38847763034373295, -0.856462027743889],
        [-0.5572857629109973, -0.3018050578439655], [-0.43759699579104505, -0.6926970097168321],
    ];
    const shallowY = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0];
    const query = [[0.37427340482759597, 0.4131198343011735]];
    const model = new ExtraTreesClassifier({ nEstimators: 3, max_depth: 1, max_features: 'all', randomState: 2 });
    model.fit(shallowX, shallowY);
    const probabilities = model.predictProba(query)[0];
    expect(probabilities[0]).toBeCloseTo(0.5100585688820983, 12);
    expect(probabilities[1]).toBeCloseTo(0.4899414311179017, 12);
    expect(model.predict(query)).toEqual([0]);
});
