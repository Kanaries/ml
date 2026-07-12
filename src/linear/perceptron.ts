import { registerEstimator, Params } from '../base/estimator';
import { SGDClassifier } from './sgdClassifier';
import { SGDPenalty } from './sgdBase';

export interface PerceptronProps {
    penalty?: SGDPenalty;
    alpha?: number;
    l1Ratio?: number;
    fitIntercept?: boolean;
    maxIter?: number;
    tol?: number | null;
    shuffle?: boolean;
    randomState?: number;
    eta0?: number;
}

/**
 * The classic perceptron, mirroring scikit-learn's `Perceptron`: an
 * `SGDClassifier` with `loss='perceptron'`, a constant learning rate `eta0`
 * (default 1) and no penalty by default. Updates only on misclassified
 * samples. Multiclass is handled one-vs-rest; `predictProba` is unavailable
 * (the perceptron has no probability model).
 */
export class Perceptron extends SGDClassifier {
    constructor(props: PerceptronProps = {}) {
        super({
            loss: 'perceptron',
            penalty: props.penalty === undefined ? null : props.penalty,
            alpha: props.alpha ?? 1e-4,
            l1Ratio: props.l1Ratio ?? 0.15,
            fitIntercept: props.fitIntercept ?? true,
            maxIter: props.maxIter ?? 1000,
            tol: props.tol === undefined ? 1e-3 : props.tol,
            shuffle: props.shuffle ?? true,
            randomState: props.randomState,
            learningRate: 'constant',
            eta0: props.eta0 ?? 1,
            powerT: 0.5,
            nIterNoChange: 5,
        });
    }

    public getParams(): Params {
        return {
            penalty: this.penalty,
            alpha: this.alpha,
            l1Ratio: this.l1Ratio,
            fitIntercept: this.fitIntercept,
            maxIter: this.maxIter,
            tol: this.tol,
            shuffle: this.shuffle,
            randomState: this.randomState,
            eta0: this.eta0,
        };
    }
}
registerEstimator('Perceptron', Perceptron);
