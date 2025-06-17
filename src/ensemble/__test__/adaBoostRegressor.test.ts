import { AdaBoostRegressor } from '../adaBoostRegressor';

 test('init', () => {
    const reg = new AdaBoostRegressor();
    expect(reg).toBeDefined();
});

test('fit simple linear', () => {
    const X: number[][] = [];
    const Y: number[] = [];
    for (let i = 0; i < 20; i++) {
        X.push([i]);
        Y.push(2 * i + 3);
    }
    const reg = new AdaBoostRegressor({ n_estimators: 10 });
    reg.fit(X, Y);
    const pred = reg.predict([[30], [40]]);
    expect(pred.length).toBe(2);
});
