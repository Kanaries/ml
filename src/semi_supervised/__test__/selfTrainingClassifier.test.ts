import { ClassifierBase, loadModel, Params } from '../../base';
import { GaussianNB } from '../../bayes';
import { SelfTrainingClassifier } from '../selfTrainingClassifier';
import fs from 'fs';
import path from 'path';

const waveB = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_b.json'), 'utf8'));

class FixedConfidenceClassifier extends ClassifierBase {
    public getParams(): Params { return {}; }
    public fit(_X: number[][], _y: number[]): void {}
    public predict(X: number[][]): number[] { return X.map(() => 1); }
    public predictProba(X: number[][]): number[][] { return X.map(() => [.2, .8]); }
}

test('SelfTrainingClassifier labels confident unlabeled samples and preserves seeds', () => {
    const { X, y, transduction, labeled_iteration: labeledIteration, n_iter: nIter } = waveB.self_training;
    const model = new SelfTrainingClassifier({ estimator: new GaussianNB(), threshold: .8, maxIter: 10 });
    model.fit(X, y);
    expect(model.transduction).toEqual(transduction);
    expect(model.labeledIteration).toEqual(labeledIteration);
    expect(model.nIter).toBe(nIter);
    expect(model.predict([[-2.5], [2.5]])).toEqual([0, 1]);
    const revived = loadModel(JSON.stringify(model)) as SelfTrainingClassifier;
    expect(revived.predict(X)).toEqual(model.predict(X));
});

test('kBest criterion adds at most kBest pseudo-labels per iteration', () => {
    const X = [[-3], [-2], [-1], [0], [1], [2], [3]];
    const model = new SelfTrainingClassifier({ estimator: new GaussianNB(), criterion: 'kBest', kBest: 1, maxIter: 1 });
    model.fit(X, [0, 0, -1, -1, -1, 1, 1]);
    expect(model.labeledIteration.filter(iteration => iteration === 1)).toHaveLength(1);
    expect(model.terminationCondition).toBe('maxIter');
});

test('nested estimator params use estimator__ addressing', () => {
    const model = new SelfTrainingClassifier({ estimator: new GaussianNB() });
    model.fit([[-1], [1]], [0, 1]);
    model.setParams({ estimator__varSmoothing: 1e-6 });
    expect((model.getParams().estimator as GaussianNB).getParams().varSmoothing).toBe(1e-6);
    expect(() => model.predict([[0]])).toThrow('not fitted');
});

test('fully labeled input matches sklearn zero-iteration semantics', () => {
    const model = new SelfTrainingClassifier({ estimator: new GaussianNB() });
    model.fit([[0], [1]], [0, 1]);
    expect(model.nIter).toBe(0);
    expect(model.terminationCondition).toBe('allLabeled');
    expect(model.labeledIteration).toEqual([0, 0]);
});

test('threshold selection is strict, matching sklearn confidence > threshold', () => {
    const model = new SelfTrainingClassifier({ estimator: new FixedConfidenceClassifier(), threshold: .8 });
    model.fit([[0], [1], [2]], [0, 1, -1]);
    expect(model.transduction).toEqual([0, 1, -1]);
    expect(model.terminationCondition).toBe('noChange');
});

test('no-change on the final iteration reports sklearn max_iter termination', () => {
    const fixture = waveB.self_training.max_iter_no_change;
    const model = new SelfTrainingClassifier({ estimator: new GaussianNB(), threshold: .99, maxIter: 1 });
    model.fit(fixture.X, fixture.y);
    expect(model.nIter).toBe(fixture.n_iter);
    expect(model.terminationCondition).toBe(fixture.termination === 'max_iter' ? 'maxIter' : fixture.termination);
});
