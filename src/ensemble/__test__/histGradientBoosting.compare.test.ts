import { readFileSync } from 'node:fs';
import path from 'node:path';
import { Ensemble, loadModel } from '../../index';

type Task = 'binary' | 'multiclass' | 'regression';

interface Reference {
    task: Task;
    weighted: boolean;
    prediction: number[];
    classes?: number[];
    probability?: number[][];
    decision?: number[] | number[][];
}

const fixture = JSON.parse(readFileSync(path.join(__dirname, '../../../test_data/hist_gradient_boosting.json'), 'utf8'));

function restoreNaN(rows: (number | null)[][]): number[][] {
    return rows.map(row => row.map(value => value === null ? NaN : value));
}

const X = restoreNaN(fixture.trainX);
const query = restoreNaN(fixture.testX);
const options = {
    learningRate: fixture.options.learning_rate,
    maxIter: fixture.options.max_iter,
    maxBins: fixture.options.max_bins,
    maxLeafNodes: fixture.options.max_leaf_nodes,
    minSamplesLeaf: fixture.options.min_samples_leaf,
    l2Regularization: fixture.options.l2_regularization,
    maxDepth: fixture.options.max_depth,
};

function expectClose(actual: number[], expected: number[]): void {
    expect(actual).toHaveLength(expected.length);
    for (let i = 0; i < actual.length; i++) expect(actual[i]).toBeCloseTo(expected[i], 6);
}

test.each<Reference>(fixture.cases)('sklearn $task with NaN features, weighted=$weighted', reference => {
    const y = fixture.targets[reference.task];
    const weights = reference.weighted ? fixture.weights : undefined;
    if (reference.task === 'regression') {
        const model = new Ensemble.HistGradientBoostingRegressor(options);
        model.fit(X, y, weights);
        expectClose(model.predict(query), reference.prediction);
        return;
    }
    const model = new Ensemble.HistGradientBoostingClassifier(options);
    model.fit(X, y, weights);
    expect(model.getClasses()).toEqual(reference.classes);
    expect(model.predict(query)).toEqual(reference.prediction);
    const probabilities = model.predictProba(query);
    expect(probabilities).toHaveLength(query.length);
    probabilities.forEach((p, i) => expectClose(p, reference.probability![i]));
    expectClose(model.decisionFunction(query).flat(), reference.decision!.flat());
    const revived = loadModel(JSON.stringify(model)) as typeof model;
    expect(revived.predictProba(query)).toEqual(probabilities);
});

test('invalid refits preserve the fitted state and reject invalid input at prediction', () => {
    const model = new Ensemble.HistGradientBoostingClassifier({ minSamplesLeaf: 1, maxIter: 3 });
    model.fit([[0], [1], [NaN]], [1, 2, 1]);
    const before = model.predictProba([[NaN]]);
    expect(() => model.fit([[0], [1]], [1, 2], [0, 0])).toThrow(/sampleWeight/);
    expect(() => model.fit([[0], [1, 2]], [1, 2])).toThrow(/feature/);
    expect(() => model.predict([[Infinity]])).toThrow(/finite/);
    expect(model.predictProba([[NaN]])).toEqual(before);
});
