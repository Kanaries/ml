import { createRandomGenerator } from '../utils/random';

export type ScoreResult = [scores: number[], pValues: number[]];

function validateXY(X: number[][], y: number[]): number {
    if (X.length === 0 || X.length !== y.length || X[0].length === 0) {
        throw new Error('X and y must be non-empty and have the same number of samples');
    }
    const p = X[0].length;
    if (X.some(row => row.length !== p)) throw new Error('all rows in X must have the same length');
    return p;
}

// Lanczos log-gamma plus Numerical Recipes continued fractions for survival probabilities.
function logGamma(z: number): number {
    const c = [0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406, 12.507343278686905, -0.13857109526572012, 9.984369578019572e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
    z -= 1;
    let x = c[0];
    for (let i = 1; i < c.length; i++) x += c[i] / (z + i);
    const t = z + 7.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function gammaQ(a: number, x: number): number {
    if (x <= 0) return 1;
    if (x < a + 1) {
        let sum = 1 / a;
        let term = sum;
        for (let n = 1; n < 200; n++) {
            term *= x / (a + n);
            sum += term;
            if (Math.abs(term) < Math.abs(sum) * 1e-14) break;
        }
        return Math.max(0, 1 - sum * Math.exp(-x + a * Math.log(x) - logGamma(a)));
    }
    let b = x + 1 - a;
    let c = 1 / 1e-300;
    let d = 1 / b;
    let h = d;
    for (let i = 1; i < 200; i++) {
        const an = -i * (i - a);
        b += 2;
        d = an * d + b;
        if (Math.abs(d) < 1e-300) d = 1e-300;
        c = b + an / c;
        if (Math.abs(c) < 1e-300) c = 1e-300;
        d = 1 / d;
        const delta = d * c;
        h *= delta;
        if (Math.abs(delta - 1) < 1e-14) break;
    }
    return Math.max(0, Math.min(1, Math.exp(-x + a * Math.log(x) - logGamma(a)) * h));
}

function betaContinuedFraction(a: number, b: number, x: number): number {
    let qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < 1e-300) d = 1e-300;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 200; m++) {
        const m2 = 2 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300;
        c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300;
        d = 1 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300;
        c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300;
        d = 1 / d;
        const delta = d * c; h *= delta;
        if (Math.abs(delta - 1) < 1e-14) break;
    }
    return h;
}

function regularizedBeta(x: number, a: number, b: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const factor = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2)
        ? factor * betaContinuedFraction(a, b, x) / a
        : 1 - factor * betaContinuedFraction(b, a, 1 - x) / b;
}

export function chi2(X: number[][], y: number[]): ScoreResult {
    const p = validateXY(X, y);
    if (X.some(row => row.some(value => value < 0 || !Number.isFinite(value)))) throw new Error('chi2 requires finite non-negative features');
    const classes = Array.from(new Set(y)).sort((a, b) => a - b);
    const classIndex = new Map(classes.map((value, i) => [value, i]));
    const counts = new Array(classes.length).fill(0);
    y.forEach(value => counts[classIndex.get(value)!]++);
    const scores = new Array(p).fill(0);
    for (let j = 0; j < p; j++) {
        const observed = new Array(classes.length).fill(0);
        for (let i = 0; i < X.length; i++) observed[classIndex.get(y[i])!] += X[i][j];
        const total = observed.reduce((a, b) => a + b, 0);
        if (total === 0) {
            scores[j] = NaN;
            continue;
        }
        for (let c = 0; c < classes.length; c++) {
            const expected = total * counts[c] / X.length;
            if (expected > 0) scores[j] += (observed[c] - expected) ** 2 / expected;
        }
    }
    return [scores, scores.map(score => Number.isNaN(score) ? NaN : gammaQ((classes.length - 1) / 2, score / 2))];
}

export function fClassif(X: number[][], y: number[]): ScoreResult {
    const p = validateXY(X, y);
    const classes = Array.from(new Set(y));
    const scores = new Array(p).fill(0);
    const dfBetween = classes.length - 1;
    const dfWithin = X.length - classes.length;
    for (let j = 0; j < p; j++) {
        const grand = X.reduce((sum, row) => sum + row[j], 0) / X.length;
        let between = 0, within = 0;
        for (const label of classes) {
            const values = X.filter((_, i) => y[i] === label).map(row => row[j]);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            between += values.length * (mean - grand) ** 2;
            within += values.reduce((sum, value) => sum + (value - mean) ** 2, 0);
        }
        scores[j] = within === 0 ? (between === 0 ? NaN : Infinity) : (between / dfBetween) / (within / dfWithin);
    }
    const pValues = scores.map(score => Number.isNaN(score) ? NaN : score === Infinity ? 0 : regularizedBeta(dfWithin / (dfWithin + dfBetween * score), dfWithin / 2, dfBetween / 2));
    return [scores, pValues];
}

function digamma(x: number): number {
    let result = 0;
    while (x < 8) { result -= 1 / x; x += 1; }
    const inv = 1 / x, inv2 = inv * inv;
    return result + Math.log(x) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 / 252));
}

function discreteMI(x: number[], y: number[]): number {
    const px = new Map<number, number>(), py = new Map<number, number>(), joint = new Map<string, number>();
    for (let i = 0; i < x.length; i++) {
        px.set(x[i], (px.get(x[i]) ?? 0) + 1);
        py.set(y[i], (py.get(y[i]) ?? 0) + 1);
        const key = `${x[i]}\u0000${y[i]}`;
        joint.set(key, (joint.get(key) ?? 0) + 1);
    }
    let mi = 0;
    for (const [key, count] of joint) {
        const [xs, ys] = key.split('\u0000').map(Number);
        mi += count / x.length * Math.log(count * x.length / (px.get(xs)! * py.get(ys)!));
    }
    return Math.max(0, mi);
}

function continuousContinuousMI(x: number[], y: number[], k: number): number {
    const n = x.length;
    const kk = Math.min(k, n - 1);
    let total = 0;
    for (let i = 0; i < n; i++) {
        const distances = Array.from({ length: n }, (_, j) => j === i ? Infinity : Math.max(Math.abs(x[i] - x[j]), Math.abs(y[i] - y[j]))).sort((a, b) => a - b);
        const radius = distances[kk - 1];
        let nx = 0, ny = 0;
        for (let j = 0; j < n; j++) if (j !== i) {
            if (Math.abs(x[i] - x[j]) < radius) nx++;
            if (Math.abs(y[i] - y[j]) < radius) ny++;
        }
        total += digamma(nx + 1) + digamma(ny + 1);
    }
    return Math.max(0, digamma(n) + digamma(kk) - total / n);
}

function continuousDiscreteMI(x: number[], y: number[], k: number): number {
    const counts = new Map<number, number>();
    y.forEach(label => counts.set(label, (counts.get(label) ?? 0) + 1));
    // Ross' estimator removes singleton-label samples before evaluating psi(n).
    const kept = y.map((label, i) => ({ label, i })).filter(entry => counts.get(entry.label)! > 1);
    const xx = kept.map(entry => x[entry.i]);
    const yy = kept.map(entry => entry.label);
    const n = xx.length;
    if (n < 2) return 0;
    const byClass = new Map<number, number[]>();
    yy.forEach((label, i) => { if (!byClass.has(label)) byClass.set(label, []); byClass.get(label)!.push(i); });
    let sumK = 0, sumLabel = 0, sumNeighbors = 0;
    for (let i = 0; i < n; i++) {
        const members = byClass.get(yy[i])!;
        const ki = Math.min(k, members.length - 1);
        const d = members.filter(j => j !== i).map(j => Math.abs(xx[i] - xx[j])).sort((a, b) => a - b);
        const radius = d[ki - 1];
        let count = 0;
        // sklearn uses nextafter(radius, 0) before its radius query, so the
        // kth boundary point is excluded; the radius query includes self.
        for (let j = 0; j < n; j++) if (Math.abs(xx[i] - xx[j]) < radius) count++;
        sumK += digamma(ki);
        sumLabel += digamma(members.length);
        sumNeighbors += digamma(count);
    }
    return Math.max(0, digamma(n) + (sumK - sumLabel - sumNeighbors) / n);
}

function gaussianGenerator(random: () => number): () => number {
    let spare: number | undefined;
    return () => {
        if (spare !== undefined) { const value = spare; spare = undefined; return value; }
        let u = 0, v = 0;
        while (u === 0) u = random();
        while (v === 0) v = random();
        const magnitude = Math.sqrt(-2 * Math.log(u));
        spare = magnitude * Math.sin(2 * Math.PI * v);
        return magnitude * Math.cos(2 * Math.PI * v);
    };
}

function scaleAndJitter(values: number[], normal: () => number): number[] {
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    const scale = Math.sqrt(variance) || 1;
    const standardized = values.map(value => value / scale); // sklearn StandardScaler(with_mean=false)
    const noiseScale = 1e-10 * Math.max(1, standardized.reduce((sum, value) => sum + Math.abs(value), 0) / standardized.length);
    return standardized.map(value => value + noiseScale * normal());
}

export interface MutualInfoOptions {
    discreteFeatures?: boolean | boolean[];
    nNeighbors?: number;
    randomState?: number;
}

function mutualInfo(X: number[][], y: number[], classification: boolean, options: MutualInfoOptions): number[] {
    const p = validateXY(X, y);
    const { discreteFeatures = false, nNeighbors = 3, randomState = 0 } = options;
    if (!Number.isInteger(nNeighbors) || nNeighbors < 1) throw new Error('nNeighbors must be a positive integer');
    const discrete = Array.isArray(discreteFeatures) ? discreteFeatures : new Array(p).fill(discreteFeatures);
    if (discrete.length !== p) throw new Error('discreteFeatures must match the feature count');
    const random = createRandomGenerator(randomState);
    const normal = gaussianGenerator(random);
    const continuousTarget = classification ? undefined : scaleAndJitter(y, normal);
    return Array.from({ length: p }, (_, j) => {
        const values = X.map(row => row[j]);
        if (discrete[j] && classification) return discreteMI(values, y);
        const jittered = discrete[j] ? values : scaleAndJitter(values, normal);
        if (classification) return continuousDiscreteMI(jittered, y, nNeighbors);
        if (discrete[j]) return continuousDiscreteMI(continuousTarget!, values, nNeighbors);
        return continuousContinuousMI(jittered, continuousTarget!, nNeighbors);
    });
}

export function mutualInfoClassif(X: number[][], y: number[], options: MutualInfoOptions = {}): number[] {
    return mutualInfo(X, y, true, options);
}

export function mutualInfoRegression(X: number[][], y: number[], options: MutualInfoOptions = {}): number[] {
    return mutualInfo(X, y, false, options);
}
