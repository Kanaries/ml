import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { createRandomGenerator } from '../utils/random';
import {
    ClassificationLossName,
    SGDLearningRate,
    SGDPenalty,
    getClassificationLoss,
    plainSGD,
    validatePenalty,
    validateSGDData,
    validateSchedule,
} from './sgdBase';

export interface SGDClassifierProps {
    loss?: ClassificationLossName;
    penalty?: SGDPenalty;
    alpha?: number;
    l1Ratio?: number;
    fitIntercept?: boolean;
    maxIter?: number;
    tol?: number | null;
    shuffle?: boolean;
    randomState?: number;
    learningRate?: SGDLearningRate;
    eta0?: number;
    powerT?: number;
    nIterNoChange?: number;
}

/**
 * Linear classifier trained with plain (non-averaged) stochastic gradient
 * descent, mirroring scikit-learn's `SGDClassifier`.
 *
 * Binary problems fit a single weight vector; multiclass problems are handled
 * one-vs-rest (one binary SGD per class, prediction by argmax of the decision
 * values). `predictProba` is available only for `loss='logLoss'` and
 * `loss='modifiedHuber'` (the sklearn rule); other losses throw.
 */
export class SGDClassifier extends ClassifierBase {
    protected loss: ClassificationLossName;
    protected penalty: SGDPenalty;
    protected alpha: number;
    protected l1Ratio: number;
    protected fitIntercept: boolean;
    protected maxIter: number;
    protected tol: number | null;
    protected shuffle: boolean;
    protected randomState?: number;
    protected learningRate: SGDLearningRate;
    protected eta0: number;
    protected powerT: number;
    protected nIterNoChange: number;

    protected classes: number[];
    /** One weight row per class (a single row for binary problems). */
    protected coef: number[][];
    /** One intercept per weight row. */
    protected intercept: number[];
    /** Epochs run by the slowest-converging binary problem. */
    protected nIter: number;

    constructor(props: SGDClassifierProps = {}) {
        super();
        this.loss = props.loss ?? 'hinge';
        this.penalty = validatePenalty(props.penalty === undefined ? 'l2' : props.penalty);
        this.alpha = props.alpha ?? 1e-4;
        this.l1Ratio = props.l1Ratio ?? 0.15;
        this.fitIntercept = props.fitIntercept ?? true;
        this.maxIter = props.maxIter ?? 1000;
        this.tol = props.tol === undefined ? 1e-3 : props.tol;
        this.shuffle = props.shuffle ?? true;
        this.randomState = props.randomState;
        this.learningRate = props.learningRate ?? 'optimal';
        this.eta0 = props.eta0 ?? 0;
        this.powerT = props.powerT ?? 0.5;
        this.nIterNoChange = props.nIterNoChange ?? 5;
        this.classes = [];
        this.coef = [];
        this.intercept = [];
        this.nIter = 0;
    }

    public getParams(): Params {
        return {
            loss: this.loss,
            penalty: this.penalty,
            alpha: this.alpha,
            l1Ratio: this.l1Ratio,
            fitIntercept: this.fitIntercept,
            maxIter: this.maxIter,
            tol: this.tol,
            shuffle: this.shuffle,
            randomState: this.randomState,
            learningRate: this.learningRate,
            eta0: this.eta0,
            powerT: this.powerT,
            nIterNoChange: this.nIterNoChange,
        };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        validateSGDData(trainX, trainY);
        getClassificationLoss(this.loss); // validate the loss name up front
        validateSchedule(this.learningRate, this.eta0, this.alpha);
        this.classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        if (this.classes.length < 2) {
            throw new Error(`SGDClassifier needs at least 2 classes, got ${this.classes.length}`);
        }
        const rand = createRandomGenerator(this.randomState);
        this.coef = [];
        this.intercept = [];
        this.nIter = 0;
        // Binary: one weight vector for the positive (larger) class.
        // Multiclass: one-vs-rest, one weight vector per class.
        const targets = this.classes.length === 2 ? [this.classes[1]] : this.classes;
        for (const positive of targets) {
            const yBin = trainY.map(v => (v === positive ? 1 : -1));
            const { weights, intercept, nIter } = plainSGD({
                X: trainX,
                y: yBin,
                loss: getClassificationLoss(this.loss),
                penalty: this.penalty,
                alpha: this.alpha,
                l1Ratio: this.l1Ratio,
                fitIntercept: this.fitIntercept,
                maxIter: this.maxIter,
                tol: this.tol,
                shuffle: this.shuffle,
                rand,
                learningRate: this.learningRate,
                eta0: this.eta0,
                powerT: this.powerT,
                nIterNoChange: this.nIterNoChange,
            });
            this.coef.push(weights);
            this.intercept.push(intercept);
            this.nIter = Math.max(this.nIter, nIter);
        }
    }

    private assertFitted(): void {
        if (this.coef.length === 0) {
            throw new Error(`${this.constructor.name} must be fitted before calling predict`);
        }
    }

    private scoreRow(x: number[], k: number): number {
        const w = this.coef[k];
        let s = this.intercept[k];
        for (let j = 0; j < w.length; j++) s += w[j] * x[j];
        return s;
    }

    /**
     * Signed distances to the decision boundary. Returns a 1-D array for
     * binary problems, an [nSamples][nClasses] array for multiclass.
     */
    public decisionFunction(testX: number[][]): number[] | number[][] {
        this.assertFitted();
        if (this.classes.length === 2) {
            return testX.map(x => this.scoreRow(x, 0));
        }
        return testX.map(x => this.coef.map((_, k) => this.scoreRow(x, k)));
    }

    public predict(testX: number[][]): number[] {
        this.assertFitted();
        if (this.classes.length === 2) {
            return testX.map(x => (this.scoreRow(x, 0) > 0 ? this.classes[1] : this.classes[0]));
        }
        return testX.map(x => {
            let best = 0;
            let bestScore = -Infinity;
            for (let k = 0; k < this.coef.length; k++) {
                const s = this.scoreRow(x, k);
                if (s > bestScore) {
                    bestScore = s;
                    best = k;
                }
            }
            return this.classes[best];
        });
    }

    /**
     * Class-membership probabilities, columns ordered by sorted `classes`.
     * Only supported for `loss='logLoss'` (sigmoid of the decision value,
     * one-vs-rest normalized for multiclass) and `loss='modifiedHuber'`
     * ((clip(d, -1, 1) + 1) / 2, normalized; uniform when every class gives
     * zero) — the same rule scikit-learn applies.
     */
    public predictProba(testX: number[][]): number[][] {
        this.assertFitted();
        if (this.loss !== 'logLoss' && this.loss !== 'modifiedHuber') {
            throw new Error('predictProba is only supported for loss=\'logLoss\' or loss=\'modifiedHuber\' ' +
                `(got loss='${this.loss}')`);
        }
        const toProb = this.loss === 'logLoss'
            ? (s: number) => 1 / (1 + Math.exp(-s))
            : (s: number) => (Math.min(1, Math.max(-1, s)) + 1) / 2;
        if (this.classes.length === 2) {
            return testX.map(x => {
                const p = toProb(this.scoreRow(x, 0));
                return [1 - p, p];
            });
        }
        const nClasses = this.classes.length;
        return testX.map(x => {
            const probs = this.coef.map((_, k) => toProb(this.scoreRow(x, k)));
            let sum = 0;
            for (const p of probs) sum += p;
            if (sum === 0) {
                // every one-vs-rest vote is zero (possible with modifiedHuber)
                return new Array<number>(nClasses).fill(1 / nClasses);
            }
            return probs.map(p => p / sum);
        });
    }

    public getClasses(): number[] {
        return this.classes.slice();
    }

    /** Fitted weights, one row per class (a single row for binary problems). */
    public getCoef(): number[][] {
        return this.coef.map(row => row.slice());
    }

    public getIntercept(): number[] {
        return this.intercept.slice();
    }

    /** Epochs run by the slowest-converging one-vs-rest problem. */
    public getNIter(): number {
        return this.nIter;
    }
}
registerEstimator('SGDClassifier', SGDClassifier);
