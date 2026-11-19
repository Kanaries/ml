import { binData, fitBinning, fitHistTree, HistTreeOptions, predictHistTree } from '../histTree';

const stump: HistTreeOptions = { maxDepth: null, maxLeafNodes: 2, minSamplesLeaf: 1, l2Regularization: 0 };

function fit(X: number[][], y: number[], options: HistTreeOptions = stump) {
    const thresholds = fitBinning(X, 255);
    const mean = y.reduce((a, b) => a + b) / y.length;
    const tree = fitHistTree(binData(X, thresholds), thresholds, y.map(v => mean - v), y.map(() => 1), options);
    return { tree, predict(test: number[][]) {
        const bins = binData(test, thresholds);
        return test.map((_, i) => mean + predictHistTree(tree, bins, i));
    } };
}

test('bins numeric values at inclusive quantile cut points and reserves a separate NaN bin', () => {
    const X = [[1, NaN], [2, NaN], [3, NaN], [4, NaN], [5, NaN], [6, NaN], [NaN, NaN]];
    const cuts = fitBinning(X, 3);
    expect(cuts).toEqual([[2.5, 4.5], []]);
    const binned = binData([[2.5, NaN], [2.51, 1], [NaN, 2], [Number.MAX_VALUE, 3]], cuts);
    expect(Array.from(binned[0])).toEqual([0, 1, 255, 2]);
    expect(Array.from(binned[1])).toEqual([255, 0, 0, 0]);
});

test('non-dividing bin counts follow the sklearn percentile grid at empirical mass boundaries', () => {
    const X = Array.from({ length: 126 }, (_, i) => [i]);
    expect(fitBinning(X, 12)[0]).toEqual([10, 21, 31, 42, 52, 62.5, 73, 84, 94, 105, 115]);
});

test('weighted quantiles skip zero weights, average at mass boundaries, and match repeated observations', () => {
    const X = [[0], [1], [2], [3], [4], [NaN]];
    expect(fitBinning(X, 2, [2, 1, 0, 1, 2, 7])).toEqual([[2]]);
    expect(fitBinning(X, 3, [2, 1, 0, 1, 2, 7])).toEqual(fitBinning([[0], [0], [1], [3], [4], [4]], 3));
    expect(fitBinning([[0], [0], [0], [0], [0], [1], [2], [3], [4]], 4)).toEqual([[0, 2]]);
});

test('adjacent floating-point values occupy different bins and invalid input is rejected', () => {
    const a = 1 + Number.EPSILON, b = 1 + 2 * Number.EPSILON;
    const X = [[a], [b], [Number.MAX_VALUE]];
    expect(Array.from(binData(X, fitBinning(X, 255))[0])).toEqual([0, 1, 2]);
    expect(() => fitBinning([[0], [Infinity]], 2)).toThrow(/finite/);
    expect(() => fitBinning([[0]], 256)).toThrow(/maxBins/);
    expect(() => binData([[0, 1]], [[]])).toThrow(/feature/);
});

test('a feature can split solely on missingness even when every observed value is identical', () => {
    const X = [[8], [8], [NaN], [NaN]];
    expect(fit(X, [0, 0, 5, 5]).predict([[NaN], [-99], [999]])).toEqual([5, 0, 0]);
});

test('learns either missing direction and routes unseen missing values according to child counts', () => {
    const X = [[0], [1], [2], [3], [NaN]];
    const queries = [[NaN], [0], [3]];
    fit(X, [2, 2, 0, 0, 2]).predict(queries).forEach((p, i) => expect(p).toBeCloseTo([2, 2, 0][i], 12));
    fit(X, [2, 2, 0, 0, 0]).predict(queries).forEach((p, i) => expect(p).toBeCloseTo([0, 2, 0][i], 12));
    expect(fit([[0], [0], [1]], [2, 2, 5]).predict([[NaN]])).toEqual([2]);
    expect(fit([[0], [1]], [2, 5]).predict([[NaN]])).toEqual([5]);
});

test('grows the most beneficial leaf first and honors tree and leaf regularization', () => {
    const X = Array.from({ length: 8 }, (_, i) => [i]);
    const y = [0, 2, 0, 2, 5, 5, 10, 10];
    const grown = fit(X, y, { ...stump, maxLeafNodes: 3 });
    expect(grown.tree).toHaveLength(5);
    expect(grown.predict(X)).toEqual([1, 1, 1, 1, 5, 5, 10, 10]);
    expect(fit(X, y, { ...stump, maxLeafNodes: null, maxDepth: 1 }).tree).toHaveLength(3);
    expect(fit(X, y, { ...stump, minSamplesLeaf: 5 }).tree).toEqual([{ value: 0 }]);
    expect(fit([[0], [1]], [0, 2], { ...stump, l2Regularization: 1 }).predict([[0], [1]])).toEqual([0.5, 1.5]);
});

test('tiny Hessians cannot create numerically unstable leaves', () => {
    const X = [[0], [1]];
    const cuts = fitBinning(X, 255);
    expect(fitHistTree(binData(X, cuts), cuts, [-1, 1], [1e-10, 1], stump)).toEqual([{ value: 0 }]);
});
