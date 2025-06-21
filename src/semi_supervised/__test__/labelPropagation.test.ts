import { LabelPropagation } from '../labelPropagation';

test('init', () => {
    const lp = new LabelPropagation();
    expect(lp).toBeDefined();
});

test('simple fit', () => {
    const X = [[0], [1], [2], [3]];
    const Y = [0, 0, -1, 1];
    const lp = new LabelPropagation({ maxIter: 100 });
    lp.fit(X, Y);
    const pred = lp.predict(X);
    expect(pred.length).toBe(4);
});
