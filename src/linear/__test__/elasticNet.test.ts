import fs from 'fs';
import path from 'path';
import { ElasticNet } from '../elasticNet';
import { Lasso } from '../lasso';
import { Ridge } from '../ridge';

test('ElasticNet initializes', () => {
    expect(new ElasticNet()).toBeDefined();
});

test('ElasticNet with l1Ratio=1 behaves like lasso on the same data', () => {
    const X = [[0], [1], [2], [3], [4]];
    const y = [1, 3, 5, 7, 9];

    const elastic = new ElasticNet({ alpha: 0.1, l1Ratio: 1, maxIter: 20000, tol: 1e-12 });
    const lasso = new Lasso({ alpha: 0.1, maxIter: 20000, tol: 1e-12 });

    elastic.fit(X, y);
    lasso.fit(X, y);

    const elasticPred = elastic.predict([[5], [6]]);
    const lassoPred = lasso.predict([[5], [6]]);
    expect(elasticPred[0]).toBeCloseTo(lassoPred[0], 6);
    expect(elasticPred[1]).toBeCloseTo(lassoPred[1], 6);
});

test('ElasticNet with l1Ratio=0 behaves like ridge with alpha scaled by n', () => {
    const X = [[0], [1], [2], [3], [4]];
    const y = [1, 3, 5, 7, 9];

    const elastic = new ElasticNet({ alpha: 0.1, l1Ratio: 0, maxIter: 20000, tol: 1e-12 });
    // ElasticNet's objective scales the residual by 1/(2n); Ridge does not,
    // so the equivalent Ridge penalty is n * alpha.
    const ridge = new Ridge({ alpha: 0.1 * X.length });

    elastic.fit(X, y);
    ridge.fit(X, y);

    const elasticPred = elastic.predict([[5], [6]]);
    const ridgePred = ridge.predict([[5], [6]]);
    expect(elasticPred[0]).toBeCloseTo(ridgePred[0], 6);
    expect(elasticPred[1]).toBeCloseTo(ridgePred[1], 6);
});

test('ElasticNet validates constructor parameters', () => {
    expect(() => new ElasticNet({ alpha: -1 })).toThrow('alpha must be a finite number >= 0');
    expect(() => new ElasticNet({ l1Ratio: -0.1 })).toThrow('l1Ratio must be a finite number between 0 and 1');
    expect(() => new ElasticNet({ l1Ratio: 1.1 })).toThrow('l1Ratio must be a finite number between 0 and 1');
});

test('ElasticNet compare with sklearn', () => {
    const p = path.join(__dirname, '../../../test_data/elastic_net.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const reg = new ElasticNet({ alpha: 0.2, l1Ratio: 0.4, maxIter: 50000, tol: 1e-12 });
    reg.fit(data.trainX, data.trainY);
    const pred = reg.predict(data.testX);
    for (let i = 0; i < pred.length; i++) {
        expect(pred[i]).toBeCloseTo(data.expected[i], 5);
    }
});

describe('l1Ratio=0 matches sklearn ElasticNet(l1_ratio=0) scaling', () => {
    // sklearn ElasticNet objective: 1/(2n)||y-Xw||^2 + 0.5*alpha*||w||^2 (at l1_ratio=0)
    // hand-derived on X=[[0..4]], y=2x+1, alpha=0.1: w = rho/(z+alpha) = 4/2.1
    test('coefficient matches the coordinate-descent limit', () => {
        const X = [[0], [1], [2], [3], [4]];
        const Y = [1, 3, 5, 7, 9];
        const en = new ElasticNet({ alpha: 0.1, l1Ratio: 0 });
        en.fit(X, Y);
        const pred = en.predict([[5]])[0];
        const w = 4 / 2.1;
        const intercept = 5 - w * 2;
        expect(pred).toBeCloseTo(intercept + 5 * w, 6);
    });

    test('is continuous as l1Ratio -> 0', () => {
        const X = [[0], [1], [2], [3], [4]];
        const Y = [1, 3, 5, 7, 9];
        const at0 = new ElasticNet({ alpha: 0.1, l1Ratio: 0 });
        const near0 = new ElasticNet({ alpha: 0.1, l1Ratio: 1e-9, maxIter: 5000, tol: 1e-10 });
        at0.fit(X, Y);
        near0.fit(X, Y);
        expect(at0.predict([[5]])[0]).toBeCloseTo(near0.predict([[5]])[0], 4);
    });
});
