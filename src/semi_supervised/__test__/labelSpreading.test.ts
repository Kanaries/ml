import { LabelSpreading } from '../labelSpreading';

test('init', () => {
    const ls = new LabelSpreading();
    expect(ls).toBeDefined();
});

test('simple fit', () => {
    const X = [[0], [1], [2], [3]];
    const Y = [0, 0, -1, 1];
    const ls = new LabelSpreading({ maxIter: 100 });
    ls.fit(X, Y);
    const pred = ls.predict(X);
    expect(pred.length).toBe(4);
});
