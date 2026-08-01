import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { DecisionTreeClassifier } from '../../tree';
import { RFE } from '../rfe';
import { RFECV } from '../rfecv';
import { SelectFromModel } from '../selectFromModel';
import { BaseEstimator } from '../../base';
import type { Params } from '../../base';

const estimator = () => new DecisionTreeClassifier({ max_depth: 4, randomState: 0 });

runEstimatorConformance([
    { name: 'SelectFromModel', kind: 'transformer', dataset: 'binary', create: () => new SelectFromModel({ estimator: estimator() }) },
    { name: 'RFE', kind: 'transformer', dataset: 'binary', create: () => new RFE({ estimator: estimator(), nFeaturesToSelect: 2 }) },
    { name: 'RFECV', kind: 'transformer', dataset: 'binary', create: () => new RFECV({ estimator: estimator(), minFeaturesToSelect: 1, cv: 3 }) },
]);

test('selectors expose support and ranking', () => {
    const X = [[0, 1, 7], [1, 1, 6], [2, 0, 7], [8, 1, 6], [9, 0, 7], [10, 1, 6]];
    const y = [0, 0, 0, 1, 1, 1];
    const rfe = new RFE({ estimator: estimator(), nFeaturesToSelect: 1 });
    rfe.fit(X, y);
    expect(rfe.getSupport(true)).toEqual([0]);
    expect(rfe.ranking[0]).toBe(1);
    expect(rfe.transform(X)[0]).toEqual([0]);
});

test('RFE uses sklearn fraction and fixed-step semantics', () => {
    const X = Array.from({ length: 12 }, (_, i) => [i, i % 2, i % 3, i % 4, i % 5, 1]);
    const y = X.map(row => row[0] > 5 ? 1 : 0);
    const rfe = new RFE({ estimator: estimator(), nFeaturesToSelect: .5, step: .5 });
    rfe.fit(X, y);
    expect((rfe.getSupport(true) as number[])).toHaveLength(3);
    expect(() => rfe.transform([[1]])).toThrow(/feature size/);

    const rfecv = new RFECV({ estimator: estimator(), minFeaturesToSelect: 1, step: 2, cv: 3 });
    rfecv.fit(X, y);
    expect(rfecv.cvResults.nFeatures).toEqual([1, 2, 4, 6]);
    expect(rfecv.gridScores).toHaveLength(4);

    const smallFraction = new RFE({ estimator: estimator(), nFeaturesToSelect: 1, step: .05 });
    expect(() => smallFraction.fit(X, y)).not.toThrow();
    expect(smallFraction.getSupport(true)).toHaveLength(1);
});

test('SelectFromModel supports scaled thresholds and pure maxFeatures top-k', () => {
    const X = [[0, 0, 1], [1, 0, 1], [2, 1, 1], [8, 0, 1], [9, 1, 1], [10, 0, 1]];
    const y = [0, 0, 0, 1, 1, 1];
    const top = new SelectFromModel({ estimator: estimator(), maxFeatures: 2 });
    top.fit(X, y);
    expect(top.getSupport(true)).toHaveLength(2);
    expect(top.fittedEstimator).toBeInstanceOf(DecisionTreeClassifier);
    const scaled = new SelectFromModel({ estimator: estimator(), threshold: '1.5*mean' });
    scaled.fit(X, y);
    expect((scaled.getSupport(true) as number[]).length).toBeLessThanOrEqual(2);
});

class FixedCoefClassifier extends BaseEstimator {
    public getParams(): Params { return {}; }
    public fit(X: number[][], _y: number[]): void { this.width = X[0].length; }
    public predict(X: number[][]): number[] { return new Array(X.length).fill(0); }
    public score(): number { return 1; }
    private width = 0;
    public get coef(): number[][] { return this.width === 2 ? [[.6, 1], [.6, 0]] : [[1], [0]]; }
}

class MalformedImportanceClassifier extends FixedCoefClassifier {
    public get coef(): number[][] { return [[1, 2, 3], [0, 0, 0]]; }
}

test('RFE uses squared multiclass coefficients like sklearn', () => {
    const selector = new RFE({ estimator: new FixedCoefClassifier(), nFeaturesToSelect: 1 });
    selector.fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 0, 1, 1]);
    expect(selector.getSupport(true)).toEqual([1]);
});

test('RFE rejects estimator importance vectors with the wrong width', () => {
    const selector = new RFE({ estimator: new MalformedImportanceClassifier(), nFeaturesToSelect: 1 });
    expect(() => selector.fit([[0, 0], [1, 1]], [0, 1])).toThrow(/importance length/);
});
