// CPU recipe from Faiss v1.16.0, contrib/e_means/base.py and e_means.py.
// https://github.com/facebookresearch/faiss/tree/v1.16.0/contrib/e_means
function nearestPowerOfTwo(x: number): number {
    return 2 ** Math.max(0, Math.round(Math.log2(x)));
}

export function resolveEMeansRecipe(n: number, d: number, k: number, epochs: number, requestedBatch?: number) {
    const quality = d <= 16;
    const coefficient = quality ? 0.00439175 : 0.035134;
    const raw = coefficient * k ** 0.134749 * n ** 0.876473 * epochs ** 0.813344;
    const cap = quality ? Math.floor(n / 2) : nearestPowerOfTwo(n / 2);
    const batch = requestedBatch ?? Math.max(1, Math.min(nearestPowerOfTwo(raw), cap));
    const batchSize = Math.min(n, batch);
    const step = 0.8244 * k ** -0.4617 * batchSize ** 0.9003 * n ** -0.4498 * epochs ** -0.1057;
    return { batchSize, alpha0: Math.max(0.0001, Math.min(1, step)), totalSteps: epochs * Math.ceil(n / batchSize) };
}

export function squaredDistance(row: number[], center: number[]): number {
    let sum = 0;
    for (let j = 0; j < row.length; j++) {
        const delta = row[j] - center[j];
        sum += delta * delta;
    }
    return sum;
}

export function nearestCenter(row: number[], centers: number[][]): number {
    let best = 0;
    let distance = squaredDistance(row, centers[0]);
    for (let c = 1; c < centers.length; c++) {
        const current = squaredDistance(row, centers[c]);
        if (current < distance) {
            distance = current;
            best = c;
        }
    }
    if (!Number.isFinite(distance)) throw new RangeError('EMeans: squared distance exceeds numeric range');
    return best;
}

function revive(centers: number[][], mass: Float64Array, fraction: number, random: () => number) {
    const threshold = fraction * (mass.reduce((sum, m) => sum + m, 0) / mass.length);
    const targets: number[] = [];
    for (let c = 0; c < mass.length; c++) if (mass[c] < threshold) targets.push(c);
    if (!targets.length) return;
    const donors: { c: number; key: number }[] = [];
    for (let c = 0; c < mass.length; c++) {
        if (mass[c] >= threshold && mass[c] > 0) {
            const u = Math.max(Number.MIN_VALUE, random());
            donors.push({ c, key: Math.log(mass[c]) - Math.log(-Math.log(u)) });
        }
    }
    donors.sort((a, b) => b.key - a.key);
    for (let i = 0; i < Math.min(targets.length, donors.length); i++) {
        const donor = donors[i].c, child = targets[i];
        mass[donor] /= 2;
        mass[child] = mass[donor];
        const source = centers[donor], target = centers[child];
        let separated = false;
        for (let j = 0; j < source.length; j++) {
            const offset = j % 2 === 0 ? 1e-4 : -1e-4;
            target[j] = source[j] * (1 + offset);
            source[j] *= 1 - offset;
            separated ||= source[j] !== target[j];
        }
        if (!separated) target[0] += 2 ** -149; // FP32 perturbation used for a zero donor
    }
}

// Transient workspace. Only the closing centers and exact readout survive fit.
export class EMeansTraining {
    readonly mass: Float64Array;
    private readonly counts: Uint32Array;
    private readonly sums: Float64Array[];
    private readonly closingCounts: Float64Array;
    private readonly closingSums: Float64Array[];
    private steps = 0;

    constructor(readonly centers: number[][], private recipe: ReturnType<typeof resolveEMeansRecipe>,
                private threshold: number, private random: () => number) {
        const k = centers.length, d = centers[0].length;
        this.mass = new Float64Array(k).fill(2 ** -16);
        this.counts = new Uint32Array(k);
        this.sums = centers.map(() => new Float64Array(d));
        this.closingCounts = new Float64Array(k);
        this.closingSums = centers.map(() => new Float64Array(d));
    }

    update(X: number[][], order: ArrayLike<number>, begin: number, end: number, closing: boolean) {
        // Assign the whole logical batch before moving any center.
        for (let i = begin; i < end; i++) {
            const row = X[order[i]], c = nearestCenter(row, this.centers);
            this.counts[c]++;
            const sum = this.sums[c];
            for (let j = 0; j < row.length; j++) sum[j] += row[j];
        }
        const cosine = (1 + Math.cos(Math.PI * this.steps / this.recipe.totalSteps)) / 2;
        const alpha = this.recipe.alpha0 * (0.1 + 0.9 * cosine);
        for (let c = 0; c < this.centers.length; c++) {
            const count = this.counts[c];
            // At alpha=1 Faiss replaces occupied mass and preserves unused centers.
            if (alpha !== 1) this.mass[c] *= 1 - alpha;
            if (count === 0) continue;
            const added = alpha * count;
            this.mass[c] = alpha === 1 ? added : this.mass[c] + added;
            const weight = added / this.mass[c];
            const center = this.centers[c], sum = this.sums[c];
            if (closing) this.closingCounts[c] += count;
            for (let j = 0; j < center.length; j++) {
                const mean = sum[j] / count;
                center[j] = weight === 1 ? mean : center[j] + weight * (mean - center[j]);
                if (closing) this.closingSums[c][j] += sum[j];
                sum[j] = 0;
            }
            this.counts[c] = 0;
        }
        this.steps++;
        if (this.threshold > 0 && this.steps % 5 === 0) revive(this.centers, this.mass, this.threshold, this.random);
    }

    finalize(): number[][] {
        for (let c = 0; c < this.centers.length; c++) {
            if (!this.closingCounts[c]) continue;
            for (let j = 0; j < this.centers[c].length; j++) {
                this.centers[c][j] = this.closingSums[c][j] / this.closingCounts[c];
            }
        }
        return this.centers;
    }
}
