import { OutlierBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { MinCovDet, MinCovDetProps } from './minCovDet';

export interface EllipticEnvelopeProps extends MinCovDetProps { contamination?: number; }

export class EllipticEnvelope extends OutlierBase {
    private contamination: number;
    private covarianceProps: MinCovDetProps;
    private robustCovariance?: MinCovDet;
    private offsetState = 0;

    constructor(props: EllipticEnvelopeProps = {}) {
        super();
        const { contamination = 0.1, ...covarianceProps } = props;
        if (!(contamination > 0) || contamination > 0.5) throw new Error('contamination must be in (0, 0.5]');
        this.contamination = contamination;
        this.covarianceProps = covarianceProps;
    }

    public getParams(): Params { return { contamination: this.contamination, ...this.covarianceProps }; }

    public fit(X: number[][]): void {
        this.robustCovariance = new MinCovDet(this.covarianceProps);
        this.robustCovariance.fit(X);
        const scores = this.scoreSamples(X).sort((a, b) => a - b);
        const pos = (scores.length - 1) * this.contamination;
        const lo = Math.floor(pos), hi = Math.ceil(pos);
        this.offsetState = scores[lo] + (scores[hi] - scores[lo]) * (pos - lo);
    }

    public scoreSamples(X: number[][]): number[] {
        if (!this.robustCovariance) throw new Error('EllipticEnvelope is not fitted');
        return this.robustCovariance.mahalanobis(X).map(value => -value);
    }
    public decisionFunction(X: number[][]): number[] { return this.scoreSamples(X).map(score => score - this.offsetState); }
    public predict(X: number[][]): number[] { return this.decisionFunction(X).map(score => score < 0 ? -1 : 1); }
    public get offset(): number { return this.offsetState; }
    public get location(): number[] { return this.robustCovariance?.location ?? []; }
    public get covariance(): number[][] { return this.robustCovariance?.covariance ?? []; }
}
registerEstimator('EllipticEnvelope', EllipticEnvelope);
