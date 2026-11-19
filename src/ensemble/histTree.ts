/** Numeric bins occupy 0..254. NaN has its own bin even when every training value is present. */
const MISSING_BIN = 255;
const MIN_CHILD_HESSIAN = 1e-3;

function validateRows(X: number[][], width: number): void {
    for (const row of X) {
        if (!Array.isArray(row) || row.length !== width) throw new Error('X must match the fitted feature count');
        for (const value of row) {
            if (typeof value !== 'number' || (!Number.isFinite(value) && !Number.isNaN(value))) {
                throw new Error('X must contain only finite numbers or NaN');
            }
        }
    }
}

function midpoint(a: number, b: number): number {
    if (a === b) return a;
    // Dividing before adding avoids overflow. Keep the upper endpoint in the next bin even for adjacent floats.
    const average = a / 2 + b / 2;
    return Math.max(a, average < b ? average : a);
}

/** Per-feature thresholds for the averaged inverse empirical CDF, ignoring NaN and zero-weight rows. */
export function fitBinning(X: number[][], maxBins: number, sampleWeight?: number[]): number[][] {
    if (!Number.isInteger(maxBins) || maxBins < 2 || maxBins > 255) throw new Error('maxBins must be in [2, 255]');
    if (!X.length || !X[0]?.length) throw new Error('X must contain at least one sample and feature');
    const width = X[0].length;
    validateRows(X, width);
    if (sampleWeight && (sampleWeight.length !== X.length || sampleWeight.some(w => !Number.isFinite(w) || w < 0))) {
        throw new Error('sampleWeight must contain one finite non-negative weight per sample');
    }
    return Array.from({ length: width }, (_, feature) => {
        const order: number[] = [];
        for (let i = 0; i < X.length; i++) {
            if (!Number.isNaN(X[i][feature]) && (!sampleWeight || sampleWeight[i] > 0)) order.push(i);
        }
        order.sort((a, b) => X[a][feature] - X[b][feature]);
        const distinct: number[] = [];
        for (const row of order) {
            const value = X[row][feature];
            if (value !== distinct[distinct.length - 1]) {
                distinct.push(value);
                if (distinct.length > maxBins) break;
            }
        }
        if (distinct.length <= maxBins) return distinct.slice(1).map((value, i) => midpoint(distinct[i], value));

        const total = sampleWeight ? order.reduce((sum, row) => sum + sampleWeight[row], 0) : order.length;
        const cuts: number[] = [];
        let index = 0;
        let cumulative = sampleWeight ? sampleWeight[order[0]] : 1;
        for (let bin = 1; bin < maxBins; bin++) {
            // Match the percentile grid (0..100) used for sklearn's numeric quantile bins.
            const target = total * ((bin * (100 / maxBins)) / 100);
            while (cumulative < target && index < order.length - 1) {
                index++;
                cumulative += sampleWeight ? sampleWeight[order[index]] : 1;
            }
            const value = X[order[index]][feature];
            const boundary = sampleWeight ? cumulative - target <= Number.EPSILON : cumulative === target;
            const cut = boundary && index + 1 < order.length ? midpoint(value, X[order[index + 1]][feature]) : value;
            if (cut !== cuts[cuts.length - 1]) cuts.push(cut);
        }
        return cuts;
    });
}

/** Bin once per dataset. Keeping each feature contiguous makes histogram accumulation cache-friendly. */
export function binData(X: number[][], thresholds: number[][]): Uint8Array[] {
    validateRows(X, thresholds.length);
    return thresholds.map((cuts, feature) => {
        const column = new Uint8Array(X.length);
        for (let i = 0; i < X.length; i++) {
            const value = X[i][feature];
            if (Number.isNaN(value)) {
                column[i] = MISSING_BIN;
                continue;
            }
            let lo = 0;
            let hi = cuts.length;
            while (lo < hi) {
                const mid = (lo + hi) >>> 1;
                if (value <= cuts[mid]) hi = mid;
                else lo = mid + 1;
            }
            column[i] = lo;
        }
        return column;
    });
}

export interface HistTreeOptions {
    maxLeafNodes: number | null;
    maxDepth: number | null;
    minSamplesLeaf: number;
    l2Regularization: number;
}

export type HistNode = { value: number } | {
    feature: number;
    splitBin: number;
    missingLeft: boolean;
    left: number;
    right: number;
};

interface Split {
    feature: number;
    splitBin: number;
    missingLeft: boolean;
    gain: number;
    leftValue: number;
    rightValue: number;
}

interface Candidate {
    node: number;
    start: number;
    end: number;
    depth: number;
    split: Split;
}

function higherPriority(a: Candidate, b: Candidate): boolean {
    return a.split.gain > b.split.gain || (a.split.gain === b.split.gain && a.node < b.node);
}

/** Keep the best available leaf at the root; use construction order for ties. */
function pushCandidate(heap: Candidate[], candidate: Candidate): void {
    let child = heap.length;
    heap.push(candidate);
    while (child > 0) {
        const parent = (child - 1) >>> 1;
        if (!higherPriority(candidate, heap[parent])) break;
        heap[child] = heap[parent];
        child = parent;
    }
    heap[child] = candidate;
}

function popCandidate(heap: Candidate[]): Candidate {
    const best = heap[0];
    const last = heap.pop()!;
    if (!heap.length) return best;
    let parent = 0;
    while (2 * parent + 1 < heap.length) {
        let child = 2 * parent + 1;
        if (child + 1 < heap.length && higherPriority(heap[child + 1], heap[child])) child++;
        if (!higherPriority(heap[child], last)) break;
        heap[parent] = heap[child];
        parent = child;
    }
    heap[parent] = last;
    return best;
}

function goesLeft(bin: number, splitBin: number, missingLeft: boolean): boolean {
    return bin === MISSING_BIN ? missingLeft : bin <= splitBin;
}

/** Gradient and Hessian are derivatives with respect to the current raw prediction. */
export function fitHistTree(
    bins: Uint8Array[], thresholds: number[][], gradients: ArrayLike<number>, hessians: ArrayLike<number>,
    options: HistTreeOptions,
): HistNode[] {
    const n = gradients.length;
    const rows = new Uint32Array(n);
    for (let i = 0; i < n; i++) rows[i] = i;
    const hasMissing = bins.map(column => column.includes(MISSING_BIN));
    const nodes: HistNode[] = [{ value: 0 }];
    const heap: Candidate[] = [];
    // Reuse a single histogram across all features and leaves. Only boundaries and leaf values survive fitting.
    const counts = new Uint32Array(256);
    const gradientSums = new Float64Array(256);
    const hessianSums = new Float64Array(256);

    function enqueue(node: number, start: number, end: number, depth: number, value: number): void {
        const size = end - start;
        if (size < 2 * options.minSamplesLeaf || (options.maxDepth !== null && depth >= options.maxDepth)) return;
        let totalGradient = 0;
        let totalHessian = 0;
        for (let pos = start; pos < end; pos++) {
            totalGradient += gradients[rows[pos]];
            totalHessian += hessians[rows[pos]];
        }
        if (totalHessian < 2 * MIN_CHILD_HESSIAN) return;
        let best: Split | undefined;
        function consider(feature: number, splitBin: number, leftCount: number, leftG: number, leftH: number, missingLeft: boolean): void {
            const rightH = totalHessian - leftH;
            if (leftCount < options.minSamplesLeaf || size - leftCount < options.minSamplesLeaf ||
                leftH < MIN_CHILD_HESSIAN || rightH < MIN_CHILD_HESSIAN) return;
            const rightG = totalGradient - leftG;
            const leftValue = -leftG / (leftH + options.l2Regularization);
            const rightValue = -rightG / (rightH + options.l2Regularization);
            const gain = totalGradient * value - leftG * leftValue - rightG * rightValue;
            if (gain > (best?.gain ?? 0)) {
                best = { feature, splitBin, gain, leftValue, rightValue,
                    missingLeft: hasMissing[feature] ? missingLeft : leftCount > size - leftCount };
            }
        }
        for (let feature = 0; feature < bins.length; feature++) {
            const column = bins[feature];
            counts.fill(0);
            gradientSums.fill(0);
            hessianSums.fill(0);
            for (let pos = start; pos < end; pos++) {
                const i = rows[pos], bin = column[i];
                counts[bin]++;
                gradientSums[bin] += gradients[i];
                hessianSums[bin] += hessians[i];
            }
            const lastBin = thresholds[feature].length;
            let leftCount = 0, leftG = 0, leftH = 0;
            // A cut after the last finite bin separates missing values from all observed values.
            for (let bin = 0; bin <= lastBin; bin++) {
                leftCount += counts[bin];
                leftG += gradientSums[bin];
                leftH += hessianSums[bin];
                consider(feature, bin, leftCount, leftG, leftH, false);
            }
            if (counts[MISSING_BIN] > 0) {
                leftCount = size; leftG = totalGradient; leftH = totalHessian;
                for (let bin = lastBin; bin > 0; bin--) {
                    leftCount -= counts[bin];
                    leftG -= gradientSums[bin];
                    leftH -= hessianSums[bin];
                    consider(feature, bin - 1, leftCount, leftG, leftH, true);
                }
            }
        }
        if (best) pushCandidate(heap, { node, start, end, depth, split: best });
    }

    enqueue(0, 0, n, 0, 0);
    let leaves = 1;
    const limit = options.maxLeafNodes ?? Infinity;
    while (heap.length && leaves < limit) {
        const { node, start, end, depth, split } = popCandidate(heap);
        const column = bins[split.feature];
        let middle = start;
        for (let pos = start; pos < end; pos++) {
            if (goesLeft(column[rows[pos]], split.splitBin, split.missingLeft)) {
                const tmp = rows[middle];
                rows[middle++] = rows[pos];
                rows[pos] = tmp;
            }
        }
        const left = nodes.length, right = left + 1;
        nodes[node] = { feature: split.feature, splitBin: split.splitBin, missingLeft: split.missingLeft, left, right };
        nodes.push({ value: split.leftValue }, { value: split.rightValue });
        leaves++;
        if (leaves < limit) {
            enqueue(left, start, middle, depth + 1, split.leftValue);
            enqueue(right, middle, end, depth + 1, split.rightValue);
        }
    }
    return nodes;
}

export function predictHistTree(tree: HistNode[], bins: Uint8Array[], row: number): number {
    let node = tree[0];
    while ('feature' in node) {
        node = tree[goesLeft(bins[node.feature][row], node.splitBin, node.missingLeft) ? node.left : node.right];
    }
    return node.value;
}
