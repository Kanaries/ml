import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import {
    MLPParams,
    MLPProps,
    MLPWeights,
    predictMLPOutput,
    resolveMLPProps,
    trainMLP,
    validateMLPFitInput,
} from './mlpBase';

export interface MLPClassifierProps extends MLPProps {}

/**
 * Multi-layer perceptron classifier trained with minibatch backpropagation
 * (sklearn `MLPClassifier` parity for the 'adam' and 'sgd' solvers; the
 * lbfgs solver is not implemented).
 *
 * Binary problems use a SINGLE logistic output unit with binary
 * cross-entropy, exactly like sklearn's LabelBinarizer encoding; multiclass
 * problems use a softmax output layer with categorical cross-entropy.
 * The L2 penalty `alpha` applies to weights only, never biases.
 *
 * Early stopping (`earlyStopping: true`) holds out the LAST
 * `validationFraction` of the rows as a plain deterministic split (sklearn
 * uses a shuffled, stratified split for classifiers — documented deviation)
 * and restores the weights with the best validation accuracy.
 */
export class MLPClassifier extends ClassifierBase {
    private params: MLPParams;
    private fitted: boolean;
    private classesList: number[];
    private coefsState: number[][][];
    private interceptsState: number[][];
    private lossCurveState: number[];
    private nIterState: number;

    constructor(props: MLPClassifierProps = {}) {
        super();
        this.params = resolveMLPProps(props);
        this.fitted = false;
        this.classesList = [];
        this.coefsState = [];
        this.interceptsState = [];
        this.lossCurveState = [];
        this.nIterState = 0;
    }

    public getParams(): Params {
        return { ...this.params, hiddenLayerSizes: [...this.params.hiddenLayerSizes] };
    }

    /** Sorted class labels seen during fit. */
    public get classes(): number[] {
        return [...this.classesList];
    }

    /** Mean training loss recorded at the end of each epoch. */
    public get lossCurve(): number[] {
        return [...this.lossCurveState];
    }

    /** Number of epochs actually run by the last fit. */
    public get nIter(): number {
        return this.nIterState;
    }

    public fit(trainX: number[][], trainY: number[]): void {
        validateMLPFitInput(trainX, trainY);
        const classes = Array.from(new Set(trainY)).sort((a, b) => a - b);
        if (classes.length < 2) {
            throw new Error('MLPClassifier needs at least 2 classes');
        }
        const binary = classes.length === 2;
        let Y: number[][];
        if (binary) {
            // single output unit targeting classes[1], like sklearn's LabelBinarizer
            Y = trainY.map(v => [v === classes[1] ? 1 : 0]);
        } else {
            const classIndex = new Map(classes.map((c, k) => [c, k]));
            Y = trainY.map(v => {
                const row = new Array(classes.length).fill(0);
                row[classIndex.get(v)!] = 1;
                return row;
            });
        }
        const outActivation = binary ? 'logistic' : 'softmax';
        const result = trainMLP(trainX, Y, this.params, {
            outActivation,
            lossKind: binary ? 'binaryLog' : 'log',
            // validation accuracy on the encoded targets
            validationScore: (weights: MLPWeights, XVal: number[][], YVal: number[][]) => {
                const out = predictMLPOutput(XVal, weights, this.params.activation, outActivation);
                let correct = 0;
                for (let i = 0; i < out.length; i++) {
                    if (binary) {
                        if ((out[i][0] >= 0.5 ? 1 : 0) === YVal[i][0]) correct++;
                    } else {
                        let best = 0;
                        for (let k = 1; k < out[i].length; k++) {
                            if (out[i][k] > out[i][best]) best = k;
                        }
                        if (YVal[i][best] === 1) correct++;
                    }
                }
                return correct / out.length;
            },
        });
        this.classesList = classes;
        this.coefsState = result.weights.coefs;
        this.interceptsState = result.weights.intercepts;
        this.lossCurveState = result.lossCurve;
        this.nIterState = result.nIter;
        this.fitted = true;
    }

    /**
     * Class-membership probabilities, columns ordered by the sorted class
     * labels (`classes`); each row sums to 1.
     */
    public predictProba(testX: number[][]): number[][] {
        const out = this.rawOutput(testX);
        if (this.classesList.length === 2) {
            return out.map(row => [1 - row[0], row[0]]);
        }
        return out;
    }

    public predict(testX: number[][]): number[] {
        const out = this.rawOutput(testX);
        if (this.classesList.length === 2) {
            return out.map(row => (row[0] >= 0.5 ? this.classesList[1] : this.classesList[0]));
        }
        return out.map(row => {
            let best = 0;
            for (let k = 1; k < row.length; k++) {
                if (row[k] > row[best]) best = k;
            }
            return this.classesList[best];
        });
    }

    private rawOutput(testX: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        const weights: MLPWeights = { coefs: this.coefsState, intercepts: this.interceptsState };
        const outActivation = this.classesList.length === 2 ? 'logistic' : 'softmax';
        return predictMLPOutput(testX, weights, this.params.activation, outActivation);
    }
}
registerEstimator('MLPClassifier', MLPClassifier);
