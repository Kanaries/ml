import { TSNE } from '../tsne';

// deterministic pseudo-random for reproducible embeddings
function mockRandom() {
    let seed = 42;
    return jest.spyOn(Math, 'random').mockImplementation(() => {
        seed = (seed * 16807) % 2147483647;
        return (seed - 1) / 2147483646;
    });
}

afterEach(() => {
    jest.restoreAllMocks();
});

function pairwiseMeanDist(Y: number[][], idxA: number[], idxB: number[]): number {
    let s = 0;
    let c = 0;
    for (const i of idxA) {
        for (const j of idxB) {
            if (i === j) continue;
            s += Math.hypot(...Y[i].map((v, d) => v - Y[j][d]));
            c++;
        }
    }
    return s / c;
}

test('tsne separates two well-separated clusters (gradient descends KL)', () => {
    mockRandom();
    const X: number[][] = [];
    for (let i = 0; i < 10; i++) X.push([i * 0.05, (i * 7 % 10) * 0.05]);
    for (let i = 0; i < 10; i++) X.push([10 + i * 0.05, 10 + (i * 3 % 10) * 0.05]);
    const a = Array.from({ length: 10 }, (_, i) => i);
    const b = Array.from({ length: 10 }, (_, i) => 10 + i);

    const tsne = new TSNE({ nComponents: 2, perplexity: 5, learningRate: 100, nIter: 400 });
    const Y = tsne.fitTransform(X);

    const intra = (pairwiseMeanDist(Y, a, a) + pairwiseMeanDist(Y, b, b)) / 2;
    const inter = pairwiseMeanDist(Y, a, b);
    // with a correct KL gradient the clusters separate cleanly
    expect(intra / inter).toBeLessThan(0.5);
});
