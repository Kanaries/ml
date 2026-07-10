import { TSNE } from '../tsne';
import fs from 'fs';
import path from 'path';

// t-SNE embeddings are not comparable coordinate-wise (sign/rotation/
// local-optimum indeterminacy), so this compares STRUCTURE against the
// sklearn-labeled blobs: finite output plus kNN label purity.

function dist(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return Math.sqrt(s);
}

function knnLabelPurity(Y: number[][], labels: number[], k: number): number {
    let agree = 0;
    for (let i = 0; i < Y.length; i++) {
        const order = Y.map((_, j) => j)
            .filter(j => j !== i)
            .sort((a, b) => dist(Y[i], Y[a]) - dist(Y[i], Y[b]))
            .slice(0, k);
        agree += order.filter(j => labels[j] === labels[i]).length / k;
    }
    return agree / Y.length;
}

// TSNE initializes Y with Math.random; pin it for a deterministic, flake-free test
beforeEach(() => {
    let seed = 12345;
    jest.spyOn(Math, 'random').mockImplementation(() => {
        seed = (seed * 16807) % 2147483647;
        return (seed - 1) / 2147483646;
    });
});

afterEach(() => {
    jest.restoreAllMocks();
});

test('tsne embedding preserves sklearn blob structure', () => {
    const p = path.join(__dirname, '../../../test_data/tsne.json');
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    const tsne = new TSNE({ nComponents: 2, perplexity: 20, learningRate: 200, nIter: 250 });
    const Y = tsne.fitTransform(data.X);

    expect(Y.length).toBe(data.X.length);
    expect(Y[0].length).toBe(2);
    for (const row of Y) for (const v of row) expect(Number.isFinite(v)).toBe(true);

    // each point's 5 nearest neighbors in the embedding overwhelmingly share its label
    expect(knnLabelPurity(Y, data.labels, 5)).toBeGreaterThan(0.9);
    // sanity: sklearn's own embedding passes the same structural bar on this fixture
    expect(knnLabelPurity(data.expected, data.labels, 5)).toBeGreaterThan(0.9);
});
