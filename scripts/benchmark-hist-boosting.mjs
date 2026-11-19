// Run each method in a fresh Node process after "yarn build".
// node scripts/benchmark-hist-boosting.mjs hist
// node scripts/benchmark-hist-boosting.mjs exact
import { Ensemble } from '../build/index.mjs';

const method = process.argv[2];
if (!['hist', 'exact'].includes(method)) throw new Error('Pass either hist or exact');
const nTrain = 8_000, nTest = 2_000, width = 12;
let seed = 915853;
function random() {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
}

const X = [], y = [];
for (let i = 0; i < nTrain + nTest; i++) {
    const row = Array.from({ length: width }, () => random() < 0.08 ? NaN : 2 * random() - 1);
    const signal = (Number.isNaN(row[0]) ? -0.7 : 2.4 * row[0])
        + (Number.isNaN(row[1]) ? 0.9 : 1.3 * row[1])
        + (row[2] > 0.2 ? 0.7 : -0.4) + (random() - 0.5) * 0.55;
    X.push(row);
    y.push(Number(signal > 0));
}
const train = X.slice(0, nTrain), test = X.slice(nTrain);
const trainLabels = y.slice(0, nTrain), testLabels = y.slice(nTrain);

const started = performance.now();
let model;
if (method === 'hist') {
    model = new Ensemble.HistGradientBoostingClassifier({
        maxIter: 25, learningRate: 0.12, maxDepth: 4, maxLeafNodes: 16,
        maxBins: 63, minSamplesLeaf: 10, l2Regularization: 1,
    });
} else {
    // Fit the imputer on training data only. Its cost is part of the exact-boosting pipeline.
    const sums = new Float64Array(width), counts = new Uint32Array(width);
    for (const row of train) {
        for (let j = 0; j < width; j++) if (!Number.isNaN(row[j])) {
            sums[j] += row[j];
            counts[j]++;
        }
    }
    const means = sums.map((sum, j) => sum / counts[j]);
    for (const row of X) {
        for (let j = 0; j < width; j++) if (Number.isNaN(row[j])) row[j] = means[j];
    }
    model = new Ensemble.XGBoostClassifier({ nEstimators: 25, learningRate: 0.12, maxDepth: 4, lambda: 1, randomState: 42 });
}
model.fit(train, trainLabels);
const fitMs = Math.round(performance.now() - started);
const prediction = model.predict(test);
const correct = prediction.filter((label, i) => label === testLabels[i]).length;
console.log(JSON.stringify({
    method, nTrain, nTest, width, fitMs, accuracy: correct / nTest,
    peakProcessRssMiB: Math.round(process.resourceUsage().maxRSS / 1024),
    node: process.version, platform: process.platform, arch: process.arch,
}));
