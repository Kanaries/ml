import { ClassifierBase, RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { createRandomGenerator } from '../utils/random';

export type DummyClassifierStrategy = 'prior' | 'mostFrequent' | 'stratified' | 'uniform' | 'constant';
export type DummyRegressorStrategy = 'mean' | 'median' | 'quantile' | 'constant';

export interface DummyClassifierProps {
    /** sklearn strategies; default 'prior'. */
    strategy?: DummyClassifierStrategy;
    /** The class to always predict (strategy 'constant' only). */
    constant?: number;
    /** Seed for the 'stratified' / 'uniform' strategies. */
    randomState?: number;
}

export interface DummyRegressorProps {
    /** sklearn strategies; default 'mean'. */
    strategy?: DummyRegressorStrategy;
    /** Quantile in [0, 1] (strategy 'quantile' only; 0.5 = median). */
    quantile?: number;
    /** The value to always predict (strategy 'constant' only). */
    constant?: number;
}

/** Index of the first maximum (numpy argmax tie-breaking). */
function argmax(values: number[]): number {
    let best = 0;
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}

/** Linear-interpolation quantile (numpy default), q in [0, 1]. */
function quantileLinear(values: number[], q: number): number {
    const sorted = values.slice().sort((a, b) => a - b);
    const pos = q * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}

/**
 * Classifier that ignores the features, mirroring `sklearn.dummy.DummyClassifier`.
 * Useful as a baseline: any real model should beat it.
 *
 * Strategies:
 *  - 'prior' (default): always predicts the most frequent class;
 *    `predictProba` returns the empirical class prior for every row.
 *  - 'mostFrequent': same predictions; `predictProba` returns the one-hot
 *    vector of the mode.
 *  - 'stratified': samples predictions from the empirical class distribution;
 *    `predictProba` rows are random one-hot draws.
 *  - 'uniform': samples classes uniformly; `predictProba` is 1/nClasses.
 *  - 'constant': always predicts `constant` (must be one of the training
 *    classes); `predictProba` is its one-hot vector.
 *
 * Randomness: the RNG for 'stratified'/'uniform' is needed at PREDICT time,
 * so `randomState` is stored and a fresh seeded generator is derived at the
 * start of every `predict`/`predictProba` call — repeated calls on the same
 * fitted model are therefore reproducible (and serialization-safe), unlike
 * sklearn where consecutive calls consume one shared RNG stream.
 */
export class DummyClassifier extends ClassifierBase {
    private strategy: DummyClassifierStrategy;
    private constant?: number;
    private randomState?: number;

    private classes: number[] = [];
    private classPrior: number[] = [];
    private fitted = false;

    constructor(props: DummyClassifierProps = {}) {
        super();
        const { strategy = 'prior', constant, randomState } = props;
        const strategies: DummyClassifierStrategy[] = ['prior', 'mostFrequent', 'stratified', 'uniform', 'constant'];
        if (!strategies.includes(strategy)) {
            throw new Error(`Unknown strategy "${strategy}". Valid strategies: ${strategies.join(', ')}.`);
        }
        this.strategy = strategy;
        this.constant = constant;
        this.randomState = randomState;
    }

    public getParams(): Params {
        return {
            strategy: this.strategy,
            constant: this.constant,
            randomState: this.randomState,
        };
    }

    public fit(X: number[][], y: number[], sampleWeight?: number[]): void {
        if (!Array.isArray(y) || y.length === 0 || X.length !== y.length) {
            throw new Error('X and y must be non-empty arrays of the same length');
        }
        if (sampleWeight !== undefined && sampleWeight.length !== y.length) {
            throw new Error('sampleWeight must have the same length as y');
        }
        const classes = Array.from(new Set(y)).sort((a, b) => a - b);
        const counts = new Map<number, number>(classes.map((c) => [c, 0]));
        let total = 0;
        for (let i = 0; i < y.length; i++) {
            const w = sampleWeight ? sampleWeight[i] : 1;
            counts.set(y[i], counts.get(y[i])! + w);
            total += w;
        }
        if (this.strategy === 'constant') {
            if (this.constant === undefined) {
                throw new Error("strategy 'constant' requires the constant param");
            }
            if (!classes.includes(this.constant)) {
                throw new Error(`constant=${this.constant} is not present in the training labels`);
            }
        }
        this.classes = classes;
        this.classPrior = classes.map((c) => counts.get(c)! / total);
        this.fitted = true;
    }

    private ensureFitted(caller: string): void {
        if (!this.fitted) {
            throw new Error(`DummyClassifier must be fitted before calling ${caller}`);
        }
    }

    /** Categorical draw from the class prior. */
    private drawFromPrior(rand: () => number): number {
        const r = rand();
        let cum = 0;
        for (let k = 0; k < this.classPrior.length; k++) {
            cum += this.classPrior[k];
            if (r < cum) return k;
        }
        return this.classPrior.length - 1;
    }

    public predict(X: number[][]): number[] {
        this.ensureFitted('predict');
        const n = X.length;
        switch (this.strategy) {
            case 'prior':
            case 'mostFrequent': {
                const mode = this.classes[argmax(this.classPrior)];
                return new Array<number>(n).fill(mode);
            }
            case 'constant':
                return new Array<number>(n).fill(this.constant!);
            case 'stratified': {
                const rand = createRandomGenerator(this.randomState);
                return Array.from({ length: n }, () => this.classes[this.drawFromPrior(rand)]);
            }
            case 'uniform': {
                const rand = createRandomGenerator(this.randomState);
                return Array.from({ length: n }, () => this.classes[Math.floor(rand() * this.classes.length)]);
            }
        }
    }

    public predictProba(X: number[][]): number[][] {
        this.ensureFitted('predictProba');
        const n = X.length;
        const K = this.classes.length;
        const oneHot = (index: number): number[] => {
            const row = new Array<number>(K).fill(0);
            row[index] = 1;
            return row;
        };
        switch (this.strategy) {
            case 'prior':
                return Array.from({ length: n }, () => this.classPrior.slice());
            case 'mostFrequent': {
                const mode = argmax(this.classPrior);
                return Array.from({ length: n }, () => oneHot(mode));
            }
            case 'constant': {
                const index = this.classes.indexOf(this.constant!);
                return Array.from({ length: n }, () => oneHot(index));
            }
            case 'uniform':
                return Array.from({ length: n }, () => new Array<number>(K).fill(1 / K));
            case 'stratified': {
                const rand = createRandomGenerator(this.randomState);
                return Array.from({ length: n }, () => oneHot(this.drawFromPrior(rand)));
            }
        }
    }
}
registerEstimator('DummyClassifier', DummyClassifier);

/**
 * Regressor that ignores the features, mirroring `sklearn.dummy.DummyRegressor`.
 *
 * Strategies: 'mean' (default), 'median' (proper even/odd handling — the
 * average of the two middle values for even-length y), 'quantile' (linear
 * interpolation, numpy's default), 'constant'.
 *
 * Omitted vs sklearn: `sampleWeight` is only supported for strategy 'mean'
 * (sklearn also implements weighted percentiles for 'median'/'quantile';
 * passing weights with those strategies throws here).
 */
export class DummyRegressor extends RegressorBase {
    private strategy: DummyRegressorStrategy;
    private quantile?: number;
    private constant?: number;

    private value = 0;
    private fitted = false;

    constructor(props: DummyRegressorProps = {}) {
        super();
        const { strategy = 'mean', quantile, constant } = props;
        const strategies: DummyRegressorStrategy[] = ['mean', 'median', 'quantile', 'constant'];
        if (!strategies.includes(strategy)) {
            throw new Error(`Unknown strategy "${strategy}". Valid strategies: ${strategies.join(', ')}.`);
        }
        if (strategy === 'quantile') {
            if (quantile === undefined || !Number.isFinite(quantile) || quantile < 0 || quantile > 1) {
                throw new Error("strategy 'quantile' requires the quantile param, a number in [0, 1]");
            }
        }
        this.strategy = strategy;
        this.quantile = quantile;
        this.constant = constant;
    }

    public getParams(): Params {
        return {
            strategy: this.strategy,
            quantile: this.quantile,
            constant: this.constant,
        };
    }

    public fit(X: number[][], y: number[], sampleWeight?: number[]): void {
        if (!Array.isArray(y) || y.length === 0 || X.length !== y.length) {
            throw new Error('X and y must be non-empty arrays of the same length');
        }
        if (sampleWeight !== undefined && sampleWeight.length !== y.length) {
            throw new Error('sampleWeight must have the same length as y');
        }
        switch (this.strategy) {
            case 'mean': {
                let sum = 0;
                let total = 0;
                for (let i = 0; i < y.length; i++) {
                    const w = sampleWeight ? sampleWeight[i] : 1;
                    sum += w * y[i];
                    total += w;
                }
                this.value = sum / total;
                break;
            }
            case 'median':
            case 'quantile': {
                if (sampleWeight !== undefined) {
                    throw new Error(`sampleWeight is not supported for strategy '${this.strategy}'`);
                }
                this.value = quantileLinear(y, this.strategy === 'median' ? 0.5 : this.quantile!);
                break;
            }
            case 'constant': {
                if (this.constant === undefined) {
                    throw new Error("strategy 'constant' requires the constant param");
                }
                this.value = this.constant;
                break;
            }
        }
        this.fitted = true;
    }

    public predict(X: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('DummyRegressor must be fitted before calling predict');
        }
        return new Array<number>(X.length).fill(this.value);
    }
}
registerEstimator('DummyRegressor', DummyRegressor);
