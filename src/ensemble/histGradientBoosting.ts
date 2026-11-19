import { ClassifierBase, RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { argmaxRow, softmaxRows } from '../discriminant_analysis/common';
import { binData, fitBinning, fitHistTree, HistNode, predictHistTree } from './histTree';

interface HistGradientBoostingOptions {
    learningRate?: number;
    maxIter?: number;
    maxLeafNodes?: number | null;
    maxDepth?: number | null;
    minSamplesLeaf?: number;
    l2Regularization?: number;
    maxBins?: number;
}

export interface HistGradientBoostingClassifierProps extends HistGradientBoostingOptions {
    loss?: 'log_loss';
}

export interface HistGradientBoostingRegressorProps extends HistGradientBoostingOptions {
    loss?: 'squared_error';
}

type Options = Required<HistGradientBoostingOptions>;

function resolveOptions({
    learningRate = 0.1, maxIter = 100, maxLeafNodes = 31, maxDepth = null,
    minSamplesLeaf = 20, l2Regularization = 0, maxBins = 255,
}: HistGradientBoostingOptions): Options {
    if (!Number.isFinite(learningRate) || learningRate <= 0) throw new Error('learningRate must be finite and positive');
    if (!Number.isSafeInteger(maxIter) || maxIter < 1) throw new Error('maxIter must be a positive integer');
    if (maxLeafNodes !== null && (!Number.isSafeInteger(maxLeafNodes) || maxLeafNodes < 2)) {
        throw new Error('maxLeafNodes must be null or an integer of at least 2');
    }
    if (maxDepth !== null && (!Number.isSafeInteger(maxDepth) || maxDepth < 1)) {
        throw new Error('maxDepth must be null or a positive integer');
    }
    if (!Number.isSafeInteger(minSamplesLeaf) || minSamplesLeaf < 1) {
        throw new Error('minSamplesLeaf must be a positive integer');
    }
    if (!Number.isFinite(l2Regularization) || l2Regularization < 0) {
        throw new Error('l2Regularization must be finite and nonnegative');
    }
    if (!Number.isInteger(maxBins) || maxBins < 2 || maxBins > 255) throw new Error('maxBins must be an integer in [2, 255]');
    return { learningRate, maxIter, maxLeafNodes, maxDepth, minSamplesLeaf, l2Regularization, maxBins };
}

function validateTargets(X: number[][], y: number[], sampleWeight?: number[]): number {
    if (!y.length || y.length !== X.length || !y.every(Number.isFinite)) {
        throw new Error('y must contain one finite target per row of a non-empty X');
    }
    if (sampleWeight === undefined) return y.length;
    if (sampleWeight.length !== y.length || !sampleWeight.every(w => Number.isFinite(w) && w >= 0)) {
        throw new Error('sampleWeight must contain one finite nonnegative weight per row');
    }
    const total = sampleWeight.reduce((sum, weight) => sum + weight, 0);
    if (!(total > 0 && Number.isFinite(total))) throw new Error('sampleWeight must have a positive finite sum');
    return total;
}

interface FittedEnsemble {
    thresholds: number[][];
    initial: number[];
    // Each output has an independent sequence of Newton trees. Binary log loss uses one output.
    trees: HistNode[][][];
}

function fitTreeAndUpdate(
    bins: Uint8Array[], thresholds: number[][], gradient: Float64Array, hessian: Float64Array,
    options: Options, raw: Float64Array,
): HistNode[] {
    const tree = fitHistTree(bins, thresholds, gradient, hessian, options);
    for (let i = 0; i < raw.length; i++) {
        raw[i] += options.learningRate * predictHistTree(tree, bins, i);
        if (!Number.isFinite(raw[i])) throw new Error('Non-finite prediction during histogram boosting');
    }
    return tree;
}

function predictRaw(X: number[][], model: FittedEnsemble | undefined, learningRate: number): number[][] {
    if (!model) throw new Error('model is not fitted');
    if (!X.length) return [];
    const bins = binData(X, model.thresholds);
    return X.map((_, row) => model.initial.map((initial, output) => {
        let margin = initial;
        for (const tree of model.trees[output]) margin += learningRate * predictHistTree(tree, bins, row);
        return margin;
    }));
}

/** Best-first histogram boosting for numeric regression with missing feature values. */
export class HistGradientBoostingRegressor extends RegressorBase {
    private options: Options;
    private model?: FittedEnsemble;

    constructor(props: HistGradientBoostingRegressorProps = {}) {
        super();
        if (props.loss !== undefined && props.loss !== 'squared_error') throw new Error('loss must be squared_error');
        this.options = resolveOptions(props);
    }

    public getParams(): Params { return { ...this.options, loss: 'squared_error' }; }
    public get nIter(): number { return this.model?.trees[0].length ?? 0; }

    public fit(X: number[][], y: number[], sampleWeight?: number[]): void {
        const totalWeight = validateTargets(X, y, sampleWeight);
        const thresholds = fitBinning(X, this.options.maxBins, sampleWeight);
        const bins = binData(X, thresholds);
        const initial = y.reduce((sum, value, i) => sum + value * ((sampleWeight?.[i] ?? 1) / totalWeight), 0);
        const raw = new Float64Array(y.length).fill(initial);
        const gradient = new Float64Array(y.length);
        const hessian = sampleWeight ? Float64Array.from(sampleWeight) : new Float64Array(y.length).fill(1);
        const trees: HistNode[][] = [];
        for (let round = 0; round < this.options.maxIter; round++) {
            for (let i = 0; i < y.length; i++) gradient[i] = (raw[i] - y[i]) * hessian[i];
            trees.push(fitTreeAndUpdate(bins, thresholds, gradient, hessian, this.options, raw));
        }
        this.model = { initial: [initial], thresholds, trees: [trees] };
    }

    public predict(X: number[][]): number[] {
        return predictRaw(X, this.model, this.options.learningRate).map(row => row[0]);
    }
}
registerEstimator('HistGradientBoostingRegressor', HistGradientBoostingRegressor);

function sigmoid(value: number): number {
    const exp = Math.exp(-Math.abs(value));
    return value >= 0 ? 1 / (1 + exp) : exp / (1 + exp);
}

function initialMargins(counts: number[], totalWeight: number): number[] {
    if (counts.length === 2) {
        const epsilon = 10 * Number.EPSILON;
        const p = Math.max(epsilon, Math.min(1 - epsilon, counts[1] / totalWeight));
        return [Math.log(p) - Math.log1p(-p)];
    }
    const logPriors = counts.map(count => Math.log(Math.max(Number.EPSILON, count / totalWeight)));
    const mean = logPriors.reduce((sum, value) => sum + value, 0) / logPriors.length;
    return logPriors.map(value => value - mean);
}

/**
 * All outputs' derivatives are evaluated at the same pre-round margins.
 * Multinomial Newton trees use the diagonal of the softmax Hessian.
 */
function classificationDerivatives(
    raw: Float64Array[], encoded: number[], sampleWeight: number[] | undefined,
    gradients: Float64Array[], hessians: Float64Array[],
): void {
    if (raw.length === 1) {
        for (let i = 0; i < encoded.length; i++) {
            const p = sigmoid(raw[0][i]);
            const q = sigmoid(-raw[0][i]);
            const weight = sampleWeight?.[i] ?? 1;
            gradients[0][i] = (encoded[i] === 1 ? -q : p) * weight;
            hessians[0][i] = p * q * weight;
        }
        return;
    }
    const exps = new Float64Array(raw.length);
    for (let i = 0; i < encoded.length; i++) {
        let max = -Infinity;
        for (const output of raw) max = Math.max(max, output[i]);
        let sum = 0;
        for (let k = 0; k < raw.length; k++) {
            exps[k] = Math.exp(raw[k][i] - max);
            sum += exps[k];
        }
        const weight = sampleWeight?.[i] ?? 1;
        for (let k = 0; k < raw.length; k++) {
            const p = exps[k] / sum;
            gradients[k][i] = (p - Number(encoded[i] === k)) * weight;
            hessians[k][i] = p * (1 - p) * weight;
        }
    }
}

/** Binary and multiclass log-loss boosting with learned missing-value routing. */
export class HistGradientBoostingClassifier extends ClassifierBase {
    private options: Options;
    private classes: number[] = [];
    private model?: FittedEnsemble;

    constructor(props: HistGradientBoostingClassifierProps = {}) {
        super();
        if (props.loss !== undefined && props.loss !== 'log_loss') throw new Error('loss must be log_loss');
        this.options = resolveOptions(props);
    }

    public getParams(): Params { return { ...this.options, loss: 'log_loss' }; }
    public getClasses(): number[] { return this.classes.slice(); }
    public get nIter(): number { return this.model?.trees[0].length ?? 0; }

    public fit(X: number[][], y: number[], sampleWeight?: number[]): void {
        const totalWeight = validateTargets(X, y, sampleWeight);
        const classes = Array.from(new Set(y)).sort((a, b) => a - b);
        if (classes.length < 2) throw new Error('HistGradientBoostingClassifier requires at least two classes');
        const lookup = new Map(classes.map((label, index) => [label, index]));
        const encoded = y.map(label => lookup.get(label)!);
        const counts = new Array<number>(classes.length).fill(0);
        for (let i = 0; i < y.length; i++) counts[encoded[i]] += sampleWeight?.[i] ?? 1;

        const thresholds = fitBinning(X, this.options.maxBins, sampleWeight);
        const bins = binData(X, thresholds);
        const initial = initialMargins(counts, totalWeight);
        const raw = initial.map(value => new Float64Array(y.length).fill(value));
        const gradients = initial.map(() => new Float64Array(y.length));
        const hessians = initial.map(() => new Float64Array(y.length));
        const trees: HistNode[][][] = initial.map(() => []);
        for (let round = 0; round < this.options.maxIter; round++) {
            classificationDerivatives(raw, encoded, sampleWeight, gradients, hessians);
            for (let k = 0; k < initial.length; k++) {
                trees[k].push(fitTreeAndUpdate(bins, thresholds, gradients[k], hessians[k], this.options, raw[k]));
            }
        }
        this.model = { initial, thresholds, trees };
        this.classes = classes;
    }

    public decisionFunction(X: number[][]): number[] | number[][] {
        const raw = predictRaw(X, this.model, this.options.learningRate);
        return this.classes.length === 2 ? raw.map(row => row[0]) : raw;
    }

    public predictProba(X: number[][]): number[][] {
        const raw = predictRaw(X, this.model, this.options.learningRate);
        return this.classes.length === 2 ? raw.map(([score]) => [sigmoid(-score), sigmoid(score)]) : softmaxRows(raw);
    }

    public predict(X: number[][]): number[] {
        const raw = predictRaw(X, this.model, this.options.learningRate);
        return raw.map(row => this.classes[row.length === 1 ? Number(row[0] > 0) : argmaxRow(row)]);
    }
}
registerEstimator('HistGradientBoostingClassifier', HistGradientBoostingClassifier);
