import { SparsePCA } from '../sparsePCA';

test('basic sparse pca', () => {
    const X = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 1],
        [2, 4, 5],
        [3, 6, 0]
    ];
    const spca = new SparsePCA({ nComponents: 2, alpha: 1 });
    const T = spca.fitTransform(X);
    expect(T.length).toBe(5);
    const comps = spca.getComponents();
    let zeroCount = 0;
    let total = 0;
    for (const row of comps) {
        for (const v of row) {
            if (v === 0) zeroCount += 1;
            total += 1;
        }
    }
    expect(zeroCount).toBeGreaterThan(0);
    expect(total).toBeGreaterThan(0);
});
