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

test('ElasticNet with l1Ratio=0 behaves like ridge on the same data', () => {
    const X = [[0], [1], [2], [3], [4]];
    const y = [1, 3, 5, 7, 9];

    const elastic = new ElasticNet({ alpha: 0.1, l1Ratio: 0, maxIter: 20000, tol: 1e-12 });
    const ridge = new Ridge({ alpha: 0.1 });

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
