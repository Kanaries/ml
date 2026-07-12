/**
 * Test-only base classifiers for the calibration suite.
 *
 * The library's GaussianNB does not expose predictProba, and
 * CalibratedClassifierCV requires the base classifier to provide continuous
 * scores, so the tests use a thin subclass deriving softmax probabilities
 * from the fitted Gaussian log-joint. A minimal decisionFunction-only
 * classifier covers the other score path.
 */
import { GaussianNB } from '../../bayes';
import { ClassifierBase } from '../../base';
import { Params, registerEstimator } from '../../base/estimator';

interface GaussianNBState {
    classes: number[];
    theta: number[][];
    variances: number[][];
    classPrior: number[];
    fitted: boolean;
}

export class GaussianNBProba extends GaussianNB {
    public predictProba(X: number[][]): number[][] {
        const state = this as unknown as GaussianNBState;
        if (!state.fitted) {
            throw new Error('GaussianNBProba must be fitted before calling predictProba');
        }
        return X.map((row) => {
            const logJoint = state.classes.map((_, c) => {
                let value = Math.log(state.classPrior[c]);
                for (let j = 0; j < row.length; j++) {
                    const variance = state.variances[c][j];
                    const diff = row[j] - state.theta[c][j];
                    value += -0.5 * Math.log(2 * Math.PI * variance) - (0.5 * diff * diff) / variance;
                }
                return value;
            });
            const max = Math.max(...logJoint);
            const exps = logJoint.map((v) => Math.exp(v - max));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map((e) => e / sum);
        });
    }
}
registerEstimator('test.GaussianNBProba', GaussianNBProba);

/** Binary classifier exposing only decisionFunction (margin = x0 - midpoint). */
export class MarginClassifier extends ClassifierBase {
    private classes: number[] = [];
    private midpoint = 0;
    private fitted = false;

    constructor(_props: Record<string, never> = {}) {
        super();
    }

    public getParams(): Params {
        return {};
    }

    public fit(X: number[][], y: number[]): void {
        this.classes = Array.from(new Set(y)).sort((a, b) => a - b);
        if (this.classes.length !== 2) {
            throw new Error('MarginClassifier is binary-only');
        }
        const meanOf = (label: number): number => {
            const rows = X.filter((_, i) => y[i] === label);
            return rows.reduce((s, r) => s + r[0], 0) / rows.length;
        };
        this.midpoint = (meanOf(this.classes[0]) + meanOf(this.classes[1])) / 2;
        this.fitted = true;
    }

    public decisionFunction(X: number[][]): number[] {
        if (!this.fitted) throw new Error('MarginClassifier must be fitted first');
        return X.map((row) => row[0] - this.midpoint);
    }

    public predict(X: number[][]): number[] {
        return this.decisionFunction(X).map((v) => (v > 0 ? this.classes[1] : this.classes[0]));
    }
}
registerEstimator('test.MarginClassifier', MarginClassifier);
