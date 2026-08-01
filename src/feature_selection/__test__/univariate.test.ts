import { chi2, fClassif, mutualInfoClassif, mutualInfoRegression } from '../univariate';

const X = [[0, 1, 0], [1, 0, 2], [0, 2, 1], [3, 0, 0], [4, 1, 0], [5, 0, 1]];
const y = [0, 0, 0, 1, 1, 1];

test('chi2 and fClassif match sklearn 1.5 fixtures', () => {
    const [chiScores, chiP] = chi2(X, y);
    const [fScores, fP] = fClassif(X, y);
    [9.307692307692308, 1, 1].forEach((v, i) => expect(chiScores[i]).toBeCloseTo(v, 12));
    [0.0022819372533154484, 0.31731050786291115, 0.31731050786291115].forEach((v, i) => expect(chiP[i]).toBeCloseTo(v, 10));
    [30.25, 1, 1].forEach((v, i) => expect(fScores[i]).toBeCloseTo(v, 12));
    [0.005328128424646905, 0.3739009663000589, 0.3739009663000589].forEach((v, i) => expect(fP[i]).toBeCloseTo(v, 10));
});

test('chi2 preserves sklearn NaN semantics for an all-zero feature', () => {
    const [scores, pValues] = chi2([[0], [0]], [0, 1]);
    expect(scores[0]).toBeNaN();
    expect(pValues[0]).toBeNaN();
});

test('discrete mutualInfoClassif matches sklearn 1.5', () => {
    const scores = mutualInfoClassif(X, y, { discreteFeatures: true, randomState: 0 });
    [0.693147180559945, 0.14384103622589034, 0.14384103622589034]
        .forEach((v, i) => expect(scores[i]).toBeCloseTo(v, 12));
});

test('k-NN mutual information ranks signal ahead of noise', () => {
    const signal = Array.from({ length: 30 }, (_, i) => i / 29);
    const noise = [1.764, .4, .979, 2.241, 1.868, -.977, .95, -.151, -.103, .411, .144, 1.454, .761, .122, .444, .334, 1.494, -.205, .313, -.854, -2.553, .654, .864, -.742, 2.27, -1.454, .046, -.187, 1.533, 1.469];
    const matrix = signal.map((value, i) => [value, 1 - value, noise[i]]);
    const cls = mutualInfoClassif(matrix, signal.map(v => v > .5 ? 1 : 0), { randomState: 0 });
    const reg = mutualInfoRegression(matrix, signal.map(v => v * v), { randomState: 0 });
    [0.6423136932469544, 0.6600914710247322, 0].forEach((value, i) => expect(cls[i]).toBeCloseTo(value, 1));
    [1.6113892473225078, 1.5891670251002852, 0].forEach((value, i) => expect(reg[i]).toBeCloseTo(value, 1));
    expect(cls[0]).toBeGreaterThan(cls[2]);
    expect(reg[0]).toBeGreaterThan(reg[2]);
});

test('continuous mutual information is scale invariant', () => {
    const y = Array.from({ length: 30 }, (_, i) => i / 29);
    const base = mutualInfoRegression(y.map(value => [value]), y, { randomState: 0 })[0];
    for (const scale of [1e3, 1e9]) {
        expect(mutualInfoRegression(y.map(value => [value * scale]), y, { randomState: 0 })[0])
            .toBeCloseTo(base, 12);
    }
});

test('continuous-discrete mutual information removes singleton labels like sklearn', () => {
    const scores = mutualInfoClassif([[0], [.1], [1], [3], [10]], [0, 0, 1, 1, 2], { randomState: 0 });
    expect(scores[0]).toBeCloseTo(0.4583333333333331, 8);
});
