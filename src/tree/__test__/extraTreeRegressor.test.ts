import { ExtraTreeRegressor } from '../extraTreeRegressor';

test('extra tree init', () => {
    const tree = new ExtraTreeRegressor();
    expect(tree).toBeDefined();
});

test('basic prediction', () => {
    const X = [[0, 0], [2, 2], [3, 6]];
    const Y = [0.5, 2.5, 3.6];
    const tree = new ExtraTreeRegressor();
    tree.fit(X, Y);
    const ans = tree.predict([[1, 1]]);
    expect(ans.length).toBe(1);
});
