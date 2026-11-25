// Shared by the Node and browser runners. Input decoding and scoring are untimed.
import { Clusters } from '../../build/index.mjs';

export const datasets = {
    blobs: [2, 3], rare: [8, 4], digits: [64, 32],
    'glove-25-angular': [25, 32], 'sift-128-euclidean': [128, 32],
};

function initialCenters(train, k, seed, rare) {
    const size = rare ? train.length - Math.floor(train.length / 100) : train.length;
    const ids = Array.from({ length: size }, (_, i) => i);
    let state = seed + 1;
    for (let i = 0; i < k; i++) {
        state = (Math.imul(1664525, state) + 1013904223) >>> 0;
        const j = i + Math.floor((state / 2 ** 32) * (size - i));
        [ids[i], ids[j]] = [ids[j], ids[i]];
    }
    const centers = ids.slice(0, k).map(i => train[i]);
    if (rare) centers[k - 1] = train[train.length - 1];
    return centers;
}

function readout(rows, centers) {
    let loss = 0;
    const labels = new Int32Array(rows.length);
    for (let i = 0; i < rows.length; i++) {
        const x = rows[i];
        let best = Infinity;
        for (let c = 0; c < centers.length; c++) {
            let dist = 0;
            for (let j = 0; j < x.length; j++) dist += (x[j] - centers[c][j]) ** 2;
            if (dist < best) { best = dist; labels[i] = c; }
        }
        loss += best;
    }
    return { mse: loss / rows.length, labels };
}

function fit(method, train, test, init, seed, epochs) {
    const k = init.length;
    let model, passes = 0;
    const start = performance.now();
    if (method === 'Lloyd') {
        model = new Clusters.KMeans({ n_clusters: k, random_state: seed,
            ...(epochs && { initCenters: init, max_iter: epochs, tol: 0 }) });
        const assign = model.assignLabels.bind(model);
        model.assignLabels = (...args) => { passes++; return assign(...args); };
    } else if (method === 'MiniBatch') {
        model = new Clusters.MiniBatchKMeans({ nClusters: k, randomState: seed,
            ...(epochs && { nInit: 1, maxIter: epochs, maxNoImprovement: null }) });
        if (epochs) model.initCenters = () => init.map(row => row.slice());
        const step = model.miniBatchStep.bind(model);
        model.miniBatchStep = (...args) => { passes += args[2].length / train.length; return step(...args); };
    } else {
        model = new Clusters.EMeans({ nClusters: k, randomState: seed,
            ...(epochs && { epochs, initCenters: init }), ...(method === 'NoRevival' && { revivalThreshold: 0 }) });
        passes = epochs || 5;
    }
    model.fitPredict(train);
    const centers = model.getCentroids();
    // Lloyd may exit at its iteration cap with labels from before the last center move.
    if (method === 'Lloyd') readout(train, centers);
    const ms = performance.now() - start;
    return { seed, method, epochs: epochs || 'default', passes, ms, mse: readout(test, centers).mse };
}

export function benchmark(name, train, test) {
    const [, k] = datasets[name];
    const small = train.slice(0, 128), warm = small.slice(0, k);
    for (const method of ['Lloyd', 'MiniBatch', 'EMeans']) fit(method, small, small, warm, 42, 2);
    const runs = [];
    for (let seed = 0; seed < 5; seed++) {
        const init = initialCenters(train, k, seed, name === 'rare');
        const methods = ['Lloyd', 'MiniBatch', 'EMeans'];
        // Rotate ordering to distribute JIT warmup and thermal effects.
        methods.push(...methods.splice(0, seed % methods.length));
        for (const method of methods) {
            for (const budget of [1, 3, 5, 25, null]) {
                runs.push(fit(method, train, test, init, seed, budget));
            }
        }
        if (name === 'rare') runs.push(fit('NoRevival', train, test, init, seed, 5));
    }
    return runs;
}

export function decode(buffer, width) {
    const flat = new Float32Array(buffer);
    if (!flat.length || flat.length % width) throw new Error('Invalid benchmark input length');
    return Array.from({ length: flat.length / width }, (_, i) => Array.from(flat.subarray(i * width, (i + 1) * width)));
}
