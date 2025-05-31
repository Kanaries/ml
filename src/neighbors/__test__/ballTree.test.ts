import { BallTree } from '../ballTree';

test('init', () => {
    const tree = new BallTree();
    expect(tree).toBeDefined();
});

test('basic query', () => {
    const X = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ];
    const tree = new BallTree(X, 1);
    const { indices } = tree.query([[0, 0]], 2);
    expect(indices[0][0]).toBe(0);
    expect([1, 2]).toContain(indices[0][1]);
});
