/**
 * Voting meta-estimators, mirroring sklearn's `VotingClassifier` and
 * `VotingRegressor`.
 *
 * Both take a named `[name, estimator]` list (the Pipeline pattern) so that
 * members are addressable in `setParams` as `name__param`, e.g.
 * `votingClf.setParams({ lr__maxIter: 50 })`.
 */
import { ClassifierBase, RegressorBase } from '../base';
import { BaseEstimator, Params, registerEstimator } from '../base/estimator';

/** Structural contract for a voting/stacking classifier member. */
export interface ClassifierLike extends BaseEstimator {
    fit(X: number[][], y: number[], sampleWeight?: number[]): void;
    predict(X: number[][]): number[];
    /** Columns ordered by the member's sorted classes (base-class contract). */
    predictProba?(X: number[][]): number[][];
    decisionFunction?(X: number[][]): number[] | number[][];
}

/** Structural contract for a voting/stacking regressor member. */
export interface RegressorLike extends BaseEstimator {
    fit(X: number[][], y: number[], sampleWeight?: number[]): void;
    predict(X: number[][]): number[];
}

export type NamedEstimator<T extends BaseEstimator = BaseEstimator> = [name: string, estimator: T];

/** Validate a named-estimator list (Pipeline-style rules). */
export function validateNamedEstimators(estimators: NamedEstimator[], owner: string): void {
    if (!Array.isArray(estimators) || estimators.length === 0) {
        throw new Error(`${owner} requires a non-empty estimators array`);
    }
    const names = new Set<string>();
    for (const entry of estimators) {
        if (!Array.isArray(entry) || entry.length !== 2 || typeof entry[0] !== 'string') {
            throw new Error(`Each ${owner} entry must be a [name, estimator] pair`);
        }
        const [name, est] = entry;
        if (names.has(name)) throw new Error(`Duplicate estimator name "${name}"`);
        if (name.includes('__')) throw new Error(`Estimator name "${name}" must not contain "__"`);
        names.add(name);
        if (!(est instanceof BaseEstimator)) {
            throw new Error(`Estimator "${name}" is not an estimator`);
        }
    }
}

/** Split a params object into own keys and nested `name__param` groups. */
export function splitNestedParams(params: Params): { own: Params; nested: Map<string, Params> } {
    const own: Params = {};
    const nested = new Map<string, Params>();
    for (const key of Object.keys(params)) {
        const idx = key.indexOf('__');
        if (idx > 0) {
            const name = key.slice(0, idx);
            const subKey = key.slice(idx + 2);
            if (!nested.has(name)) nested.set(name, {});
            nested.get(name)![subKey] = params[key];
        } else {
            own[key] = params[key];
        }
    }
    return { own, nested };
}

export function validateVotingWeights(weights: number[] | undefined, nEstimators: number): void {
    if (weights === undefined) return;
    if (!Array.isArray(weights) || weights.length !== nEstimators) {
        throw new Error(`weights must have one entry per estimator (${nEstimators}), got ${weights?.length}`);
    }
    let total = 0;
    for (const w of weights) {
        if (!Number.isFinite(w) || w < 0) throw new Error('weights must be finite numbers >= 0');
        total += w;
    }
    if (total <= 0) throw new Error('weights must not all be zero');
}

function validateXY(X: number[][], y: number[]): void {
    if (!Array.isArray(X) || X.length === 0 || !Array.isArray(y) || y.length === 0) {
        throw new Error('X and y must be non-empty');
    }
    if (X.length !== y.length) {
        throw new Error('X and y must have the same length');
    }
}

export function sortedUniqueLabels(y: number[]): number[] {
    return Array.from(new Set(y)).sort((a, b) => a - b);
}

/**
 * Index of the maximum value; on an exact tie the FIRST (lowest-index)
 * maximum wins — the same rule as numpy's `argmax`, which is what sklearn's
 * voting uses. Since classes are stored in ascending sorted order, ties
 * therefore resolve to the smallest class label.
 */
export function argmaxFirst(values: number[]): number {
    let best = 0;
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}

export interface VotingClassifierProps {
    /** Named members; addressable as `name__param` in setParams. */
    estimators: NamedEstimator<ClassifierLike>[];
    /** 'hard': weighted majority of labels; 'soft': argmax of weighted-average predictProba. */
    voting?: 'hard' | 'soft';
    /** Per-estimator weights (default: uniform). */
    weights?: number[];
}

/**
 * Soft/hard voting ensemble following sklearn's `VotingClassifier`.
 *
 * Tie-breaking rule (both modes): the class with the highest weighted
 * vote/probability wins; on an exact tie the smallest class label (first in
 * ascending sorted class order) is returned, matching sklearn/np.argmax.
 */
export class VotingClassifier extends ClassifierBase {
    private estimators: NamedEstimator<ClassifierLike>[];
    private voting: 'hard' | 'soft';
    private weights?: number[];
    private classes: number[];
    private fitted: boolean;

    constructor(props: VotingClassifierProps) {
        super();
        const { estimators, voting = 'hard', weights } = props ?? {};
        validateNamedEstimators(estimators, 'VotingClassifier');
        if (voting !== 'hard' && voting !== 'soft') {
            throw new Error(`voting must be 'hard' or 'soft', got "${voting}"`);
        }
        validateVotingWeights(weights, estimators.length);
        this.estimators = estimators.map(([name, est]) => [name, est]);
        this.voting = voting;
        this.weights = weights;
        this.classes = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            estimators: this.estimators.map(([name, est]) => [name, est]),
            voting: this.voting,
            weights: this.weights,
        };
    }

    /** Supports both own params and nested `name__param` addressing. */
    public setParams(params: Params): this {
        const { own, nested } = splitNestedParams(params);
        if (Object.keys(own).length > 0) super.setParams(own);
        for (const [name, subParams] of nested) {
            this.getEstimator(name).setParams(subParams);
        }
        return this;
    }

    public getEstimator(name: string): BaseEstimator {
        const entry = this.estimators.find(([n]) => n === name);
        if (!entry) {
            throw new Error(`Unknown estimator "${name}". Estimators: ${this.estimators.map(([n]) => n).join(', ')}`);
        }
        return entry[1];
    }

    public get namedEstimators(): Record<string, BaseEstimator> {
        return Object.fromEntries(this.estimators);
    }

    public getClasses(): number[] {
        return this.classes.slice();
    }

    public fit(X: number[][], y: number[], sampleWeight?: number[]): void {
        validateXY(X, y);
        if (this.voting === 'soft') {
            for (const [name, est] of this.estimators) {
                if (typeof est.predictProba !== 'function') {
                    throw new Error(`voting='soft' requires every estimator to implement predictProba; ` +
                        `"${name}" (${est.constructor.name}) does not`);
                }
            }
        }
        this.classes = sortedUniqueLabels(y);
        for (const [, est] of this.estimators) {
            est.fit(X, y, sampleWeight);
        }
        this.fitted = true;
    }

    public predict(X: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('VotingClassifier must be fitted before calling predict');
        }
        if (this.voting === 'soft') {
            return this.averagedProba(X).map(row => this.classes[argmaxFirst(row)]);
        }
        // hard voting: weighted label counts, argmax with first-win ties
        const classIndex = new Map<number, number>(this.classes.map((c, i) => [c, i]));
        const memberPreds = this.estimators.map(([, est]) => est.predict(X));
        return X.map((_, i) => {
            const counts = new Array(this.classes.length).fill(0);
            for (let m = 0; m < memberPreds.length; m++) {
                const label = memberPreds[m][i];
                const idx = classIndex.get(label);
                if (idx === undefined) {
                    throw new Error(`Estimator "${this.estimators[m][0]}" predicted unknown label ${label}`);
                }
                counts[idx] += this.weights ? this.weights[m] : 1;
            }
            return this.classes[argmaxFirst(counts)];
        });
    }

    /** Weighted-average class probabilities. Only available with voting='soft'. */
    public predictProba(X: number[][]): number[][] {
        if (this.voting !== 'soft') {
            throw new Error("predictProba is not available when voting='hard'");
        }
        if (!this.fitted) {
            throw new Error('VotingClassifier must be fitted before calling predictProba');
        }
        return this.averagedProba(X);
    }

    private averagedProba(X: number[][]): number[][] {
        const nClasses = this.classes.length;
        const totalWeight = this.weights
            ? this.weights.reduce((a, b) => a + b, 0)
            : this.estimators.length;
        const avg: number[][] = X.map(() => new Array(nClasses).fill(0));
        for (let m = 0; m < this.estimators.length; m++) {
            const [name, est] = this.estimators[m];
            const proba = est.predictProba!(X);
            const w = this.weights ? this.weights[m] : 1;
            for (let i = 0; i < X.length; i++) {
                if (proba[i].length !== nClasses) {
                    throw new Error(`Estimator "${name}" returned ${proba[i].length} probability columns, ` +
                        `expected ${nClasses}`);
                }
                for (let c = 0; c < nClasses; c++) {
                    avg[i][c] += (w * proba[i][c]) / totalWeight;
                }
            }
        }
        return avg;
    }
}
registerEstimator('VotingClassifier', VotingClassifier);

export interface VotingRegressorProps {
    /** Named members; addressable as `name__param` in setParams. */
    estimators: NamedEstimator<RegressorLike>[];
    /** Per-estimator weights (default: uniform). */
    weights?: number[];
}

/** Weighted-mean prediction ensemble following sklearn's `VotingRegressor`. */
export class VotingRegressor extends RegressorBase {
    private estimators: NamedEstimator<RegressorLike>[];
    private weights?: number[];
    private fitted: boolean;

    constructor(props: VotingRegressorProps) {
        super();
        const { estimators, weights } = props ?? {};
        validateNamedEstimators(estimators, 'VotingRegressor');
        validateVotingWeights(weights, estimators.length);
        this.estimators = estimators.map(([name, est]) => [name, est]);
        this.weights = weights;
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            estimators: this.estimators.map(([name, est]) => [name, est]),
            weights: this.weights,
        };
    }

    /** Supports both own params and nested `name__param` addressing. */
    public setParams(params: Params): this {
        const { own, nested } = splitNestedParams(params);
        if (Object.keys(own).length > 0) super.setParams(own);
        for (const [name, subParams] of nested) {
            this.getEstimator(name).setParams(subParams);
        }
        return this;
    }

    public getEstimator(name: string): BaseEstimator {
        const entry = this.estimators.find(([n]) => n === name);
        if (!entry) {
            throw new Error(`Unknown estimator "${name}". Estimators: ${this.estimators.map(([n]) => n).join(', ')}`);
        }
        return entry[1];
    }

    public get namedEstimators(): Record<string, BaseEstimator> {
        return Object.fromEntries(this.estimators);
    }

    public fit(X: number[][], y: number[], sampleWeight?: number[]): void {
        validateXY(X, y);
        for (const [, est] of this.estimators) {
            est.fit(X, y, sampleWeight);
        }
        this.fitted = true;
    }

    public predict(X: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('VotingRegressor must be fitted before calling predict');
        }
        const totalWeight = this.weights
            ? this.weights.reduce((a, b) => a + b, 0)
            : this.estimators.length;
        const memberPreds = this.estimators.map(([, est]) => est.predict(X));
        return X.map((_, i) => {
            let sum = 0;
            for (let m = 0; m < memberPreds.length; m++) {
                sum += (this.weights ? this.weights[m] : 1) * memberPreds[m][i];
            }
            return sum / totalWeight;
        });
    }
}
registerEstimator('VotingRegressor', VotingRegressor);
