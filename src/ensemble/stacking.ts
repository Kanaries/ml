/**
 * Stacked generalization meta-estimators, mirroring sklearn's
 * `StackingClassifier` and `StackingRegressor`.
 *
 * The final estimator is trained on OUT-OF-FOLD predictions of the base
 * estimators: the data is split with (Stratified)KFold, each base estimator
 * is `clone()`d per fold, fit on the fold's training part, and its
 * predictions on the held-out part fill that fold's rows of the meta-feature
 * matrix. The base estimators themselves are then refit on the full data and
 * used to build meta-features at inference time.
 */
import { ClassifierBase, RegressorBase } from '../base';
import { BaseEstimator, Params, registerEstimator } from '../base/estimator';
import { KFold, StratifiedKFold, FoldIndices } from '../utils/modelSelection';
import { LogisticRegression } from '../linear/logisticRegression';
import { RidgeRegression } from '../linear/ridgeRegression';
import {
    ClassifierLike,
    RegressorLike,
    NamedEstimator,
    splitNestedParams,
    validateNamedEstimators,
    sortedUniqueLabels,
} from './voting';

export type StackMethod = 'auto' | 'predictProba' | 'decisionFunction' | 'predict';
type ResolvedStackMethod = Exclude<StackMethod, 'auto'>;

function validateXY(X: number[][], y: number[]): void {
    if (!Array.isArray(X) || X.length === 0 || !Array.isArray(y) || y.length === 0) {
        throw new Error('X and y must be non-empty');
    }
    if (X.length !== y.length) {
        throw new Error('X and y must have the same length');
    }
}

function validateCv(cv: number): void {
    if (!Number.isInteger(cv) || cv < 2) {
        throw new Error('cv must be an integer >= 2');
    }
}

/**
 * Resolve the meta-feature method for one base estimator.
 * 'auto' = predictProba if available, else decisionFunction, else predict.
 */
function resolveStackMethod(name: string, est: ClassifierLike, requested: StackMethod): ResolvedStackMethod {
    if (requested === 'auto') {
        if (typeof est.predictProba === 'function') return 'predictProba';
        if (typeof est.decisionFunction === 'function') return 'decisionFunction';
        return 'predict';
    }
    if (typeof (est as unknown as Record<string, unknown>)[requested] !== 'function') {
        throw new Error(`stackMethod='${requested}' but estimator "${name}" (${est.constructor.name}) ` +
            `does not implement ${requested}()`);
    }
    return requested;
}

/**
 * One base estimator's meta-feature block for X, as a [nSamples][width]
 * matrix.
 *  - predictProba: nClasses columns; for binary (2 columns) the FIRST column
 *    is dropped (sklearn behavior — the two columns are collinear).
 *  - decisionFunction: 1 column (binary number[]) or nClasses columns.
 *  - predict: 1 column of raw predictions/labels.
 */
function memberMetaBlock(est: ClassifierLike, method: ResolvedStackMethod, X: number[][]): number[][] {
    if (method === 'predictProba') {
        const proba = est.predictProba!(X);
        if (proba.length > 0 && proba[0].length === 2) {
            return proba.map(row => [row[1]]);
        }
        return proba.map(row => row.slice());
    }
    if (method === 'decisionFunction') {
        const out = est.decisionFunction!(X);
        if (out.length > 0 && Array.isArray(out[0])) {
            return (out as number[][]).map(row => row.slice());
        }
        return (out as number[]).map(v => [v]);
    }
    return est.predict(X).map(v => [v]);
}

/**
 * Build the out-of-fold meta-feature matrix. For every fold, every base
 * estimator is cloned (unfitted copy), fit on the fold's train part, and its
 * meta block on the held-out part fills the held-out rows. Every sample's
 * meta-features therefore come from a model that never saw that sample.
 */
function outOfFoldMetaFeatures(
    estimators: NamedEstimator<ClassifierLike>[],
    methods: ResolvedStackMethod[],
    folds: FoldIndices[],
    X: number[][],
    y: number[],
    sampleWeight?: number[],
): number[][][] {
    const oof: (number[] | undefined)[][] = estimators.map(() => new Array(X.length).fill(undefined));
    for (const fold of folds) {
        const trainX = fold.trainIndices.map(i => X[i]);
        const trainY = fold.trainIndices.map(i => y[i]);
        const trainW = sampleWeight ? fold.trainIndices.map(i => sampleWeight[i]) : undefined;
        const testX = fold.testIndices.map(i => X[i]);
        for (let m = 0; m < estimators.length; m++) {
            const foldEst = estimators[m][1].clone();
            foldEst.fit(trainX, trainY, trainW);
            const block = memberMetaBlock(foldEst, methods[m], testX);
            for (let k = 0; k < fold.testIndices.length; k++) {
                oof[m][fold.testIndices[k]] = block[k];
            }
        }
    }
    return oof.map((rows, m) => rows.map((row, i) => {
        if (row === undefined) {
            throw new Error(`Sample ${i} was not covered by any CV test fold (estimator "${estimators[m][0]}")`);
        }
        return row;
    }));
}

/** Stitch per-member blocks (and optionally the original X) into meta rows. */
function concatBlocks(blocks: number[][][], X: number[][], passthrough: boolean): number[][] {
    return X.map((row, i) => {
        const out: number[] = [];
        for (const block of blocks) out.push(...block[i]);
        if (passthrough) out.push(...row);
        return out;
    });
}

export interface StackingClassifierProps {
    /** Named base estimators; addressable as `name__param` in setParams. */
    estimators: NamedEstimator<ClassifierLike>[];
    /**
     * Meta-estimator fit on the out-of-fold meta-features. Defaults to
     * `LogisticRegression()` (binary-only in this library — pass a
     * multiclass-capable classifier for >2 classes). Addressable as
     * `finalEstimator__param`.
     */
    finalEstimator?: ClassifierLike;
    /** StratifiedKFold split count for the out-of-fold predictions. */
    cv?: number;
    /** How base estimators produce meta-features (see resolveStackMethod). */
    stackMethod?: StackMethod;
    /** Append the original features after the meta-features. */
    passthrough?: boolean;
}

export class StackingClassifier extends ClassifierBase {
    private estimators: NamedEstimator<ClassifierLike>[];
    private finalEstimator: ClassifierLike;
    private cv: number;
    private stackMethod: StackMethod;
    private passthrough: boolean;
    private classes: number[];
    private resolvedMethods: ResolvedStackMethod[];
    private fitted: boolean;

    constructor(props: StackingClassifierProps) {
        super();
        const { estimators, finalEstimator, cv = 5, stackMethod = 'auto', passthrough = false } = props ?? {};
        validateNamedEstimators(estimators, 'StackingClassifier');
        validateCv(cv);
        if (!['auto', 'predictProba', 'decisionFunction', 'predict'].includes(stackMethod)) {
            throw new Error(`Unknown stackMethod "${stackMethod}"`);
        }
        if (finalEstimator !== undefined && !(finalEstimator instanceof BaseEstimator)) {
            throw new Error('finalEstimator must be an estimator');
        }
        this.estimators = estimators.map(([name, est]) => [name, est]);
        this.finalEstimator = finalEstimator ?? new LogisticRegression();
        this.cv = cv;
        this.stackMethod = stackMethod;
        this.passthrough = passthrough;
        this.classes = [];
        this.resolvedMethods = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            estimators: this.estimators.map(([name, est]) => [name, est]),
            finalEstimator: this.finalEstimator,
            cv: this.cv,
            stackMethod: this.stackMethod,
            passthrough: this.passthrough,
        };
    }

    /** Supports own params plus `name__param` / `finalEstimator__param`. */
    public setParams(params: Params): this {
        const { own, nested } = splitNestedParams(params);
        if (Object.keys(own).length > 0) super.setParams(own);
        for (const [name, subParams] of nested) {
            this.getEstimator(name).setParams(subParams);
        }
        return this;
    }

    public getEstimator(name: string): BaseEstimator {
        if (name === 'finalEstimator') return this.finalEstimator;
        const entry = this.estimators.find(([n]) => n === name);
        if (!entry) {
            throw new Error(`Unknown estimator "${name}". ` +
                `Estimators: ${this.estimators.map(([n]) => n).join(', ')}, finalEstimator`);
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
        this.classes = sortedUniqueLabels(y);
        this.resolvedMethods = this.estimators.map(([name, est]) => resolveStackMethod(name, est, this.stackMethod));
        // out-of-fold meta-features (stratified so every train part sees all classes)
        const folds = new StratifiedKFold({ nSplits: this.cv }).split(X, y);
        const oofBlocks = outOfFoldMetaFeatures(this.estimators, this.resolvedMethods, folds, X, y, sampleWeight);
        const metaX = concatBlocks(oofBlocks, X, this.passthrough);
        // refit the base estimators on the full data for inference
        for (const [, est] of this.estimators) {
            est.fit(X, y, sampleWeight);
        }
        this.finalEstimator.fit(metaX, y, sampleWeight);
        this.fitted = true;
    }

    /** Meta-features for X from the full-data base estimators. */
    public transform(X: number[][]): number[][] {
        this.assertFitted('transform');
        const blocks = this.estimators.map(([, est], m) => memberMetaBlock(est, this.resolvedMethods[m], X));
        return concatBlocks(blocks, X, this.passthrough);
    }

    public predict(X: number[][]): number[] {
        this.assertFitted('predict');
        return this.finalEstimator.predict(this.transform(X));
    }

    public predictProba(X: number[][]): number[][] {
        this.assertFitted('predictProba');
        if (typeof this.finalEstimator.predictProba !== 'function') {
            throw new Error(`finalEstimator (${this.finalEstimator.constructor.name}) does not implement predictProba()`);
        }
        return this.finalEstimator.predictProba(this.transform(X));
    }

    public decisionFunction(X: number[][]): number[] | number[][] {
        this.assertFitted('decisionFunction');
        if (typeof this.finalEstimator.decisionFunction !== 'function') {
            throw new Error(`finalEstimator (${this.finalEstimator.constructor.name}) does not implement decisionFunction()`);
        }
        return this.finalEstimator.decisionFunction(this.transform(X));
    }

    private assertFitted(method: string): void {
        if (!this.fitted) {
            throw new Error(`StackingClassifier must be fitted before calling ${method}`);
        }
    }
}
registerEstimator('StackingClassifier', StackingClassifier);

export interface StackingRegressorProps {
    /** Named base estimators; addressable as `name__param` in setParams. */
    estimators: NamedEstimator<RegressorLike>[];
    /**
     * Meta-estimator fit on the out-of-fold predictions. Defaults to
     * `RidgeRegression()`. Addressable as `finalEstimator__param`.
     */
    finalEstimator?: RegressorLike;
    /** KFold split count for the out-of-fold predictions. */
    cv?: number;
    /** Append the original features after the meta-features. */
    passthrough?: boolean;
}

export class StackingRegressor extends RegressorBase {
    private estimators: NamedEstimator<RegressorLike>[];
    private finalEstimator: RegressorLike;
    private cv: number;
    private passthrough: boolean;
    private fitted: boolean;

    constructor(props: StackingRegressorProps) {
        super();
        const { estimators, finalEstimator, cv = 5, passthrough = false } = props ?? {};
        validateNamedEstimators(estimators, 'StackingRegressor');
        validateCv(cv);
        if (finalEstimator !== undefined && !(finalEstimator instanceof BaseEstimator)) {
            throw new Error('finalEstimator must be an estimator');
        }
        this.estimators = estimators.map(([name, est]) => [name, est]);
        this.finalEstimator = finalEstimator ?? new RidgeRegression();
        this.cv = cv;
        this.passthrough = passthrough;
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            estimators: this.estimators.map(([name, est]) => [name, est]),
            finalEstimator: this.finalEstimator,
            cv: this.cv,
            passthrough: this.passthrough,
        };
    }

    /** Supports own params plus `name__param` / `finalEstimator__param`. */
    public setParams(params: Params): this {
        const { own, nested } = splitNestedParams(params);
        if (Object.keys(own).length > 0) super.setParams(own);
        for (const [name, subParams] of nested) {
            this.getEstimator(name).setParams(subParams);
        }
        return this;
    }

    public getEstimator(name: string): BaseEstimator {
        if (name === 'finalEstimator') return this.finalEstimator;
        const entry = this.estimators.find(([n]) => n === name);
        if (!entry) {
            throw new Error(`Unknown estimator "${name}". ` +
                `Estimators: ${this.estimators.map(([n]) => n).join(', ')}, finalEstimator`);
        }
        return entry[1];
    }

    public get namedEstimators(): Record<string, BaseEstimator> {
        return Object.fromEntries(this.estimators);
    }

    public fit(X: number[][], y: number[], sampleWeight?: number[]): void {
        validateXY(X, y);
        const folds = new KFold({ nSplits: this.cv }).split(X, y);
        const methods = this.estimators.map(() => 'predict' as ResolvedStackMethod);
        const oofBlocks = outOfFoldMetaFeatures(
            this.estimators as NamedEstimator<ClassifierLike>[], methods, folds, X, y, sampleWeight);
        const metaX = concatBlocks(oofBlocks, X, this.passthrough);
        for (const [, est] of this.estimators) {
            est.fit(X, y, sampleWeight);
        }
        this.finalEstimator.fit(metaX, y, sampleWeight);
        this.fitted = true;
    }

    /** Meta-features for X from the full-data base estimators. */
    public transform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('StackingRegressor must be fitted before calling transform');
        }
        const blocks = this.estimators.map(([, est]) => est.predict(X).map(v => [v]));
        return concatBlocks(blocks, X, this.passthrough);
    }

    public predict(X: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('StackingRegressor must be fitted before calling predict');
        }
        return this.finalEstimator.predict(this.transform(X));
    }
}
registerEstimator('StackingRegressor', StackingRegressor);
