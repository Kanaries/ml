export interface StandardScalerProps {
    withMean?: boolean;
    withStd?: boolean;
}

export interface MinMaxScalerProps {
    featureRange?: [number, number];
}

function validateMatrix(X: number[][]): void {
    if (X.length === 0) {
        throw new Error('X must be non-empty');
    }
    const nFeatures = X[0].length;
    if (nFeatures === 0) {
        throw new Error('X must contain at least one feature');
    }
    for (let i = 1; i < X.length; i++) {
        if (X[i].length !== nFeatures) {
            throw new Error('X must be a rectangular matrix');
        }
    }
}

export class StandardScaler {
    private withMean: boolean;
    private withStd: boolean;
    private means: number[];
    private scales: number[];
    private fitted: boolean;

    constructor(props: StandardScalerProps = {}) {
        const { withMean = true, withStd = true } = props;
        this.withMean = withMean;
        this.withStd = withStd;
        this.means = [];
        this.scales = [];
        this.fitted = false;
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nSamples = X.length;
        const nFeatures = X[0].length;

        this.means = new Array(nFeatures).fill(0);
        this.scales = new Array(nFeatures).fill(1);

        for (let j = 0; j < nFeatures; j++) {
            let sum = 0;
            for (let i = 0; i < nSamples; i++) {
                sum += X[i][j];
            }
            const mean = sum / nSamples;
            this.means[j] = mean;

            let varianceSum = 0;
            for (let i = 0; i < nSamples; i++) {
                const diff = X[i][j] - mean;
                varianceSum += diff * diff;
            }
            const std = Math.sqrt(varianceSum / nSamples);
            this.scales[j] = std === 0 ? 1 : std;
        }
        this.fitted = true;
    }

    private assertFittedAndShape(X: number[][]): void {
        if (!this.fitted) {
            throw new Error('StandardScaler must be fitted before calling transform');
        }
        validateMatrix(X);
        if (X[0].length !== this.means.length) {
            throw new Error('X has different number of features than fitted data');
        }
    }

    public transform(X: number[][]): number[][] {
        this.assertFittedAndShape(X);
        return X.map(row =>
            row.map((value, j) => {
                let ans = value;
                if (this.withMean) {
                    ans -= this.means[j];
                }
                if (this.withStd) {
                    ans /= this.scales[j];
                }
                return ans;
            })
        );
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }

    public inverseTransform(X: number[][]): number[][] {
        this.assertFittedAndShape(X);
        return X.map(row =>
            row.map((value, j) => {
                let ans = value;
                if (this.withStd) {
                    ans *= this.scales[j];
                }
                if (this.withMean) {
                    ans += this.means[j];
                }
                return ans;
            })
        );
    }
}

export class MinMaxScaler {
    private featureRange: [number, number];
    private dataMin: number[];
    private scales: number[];
    private offsets: number[];
    private fitted: boolean;

    constructor(props: MinMaxScalerProps = {}) {
        const { featureRange = [0, 1] } = props;
        if (featureRange[0] >= featureRange[1]) {
            throw new Error('featureRange min must be less than max');
        }
        this.featureRange = featureRange;
        this.dataMin = [];
        this.scales = [];
        this.offsets = [];
        this.fitted = false;
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nFeatures = X[0].length;
        const targetMin = this.featureRange[0];
        const targetMax = this.featureRange[1];
        const targetRange = targetMax - targetMin;

        this.dataMin = new Array(nFeatures).fill(0);
        this.scales = new Array(nFeatures).fill(1);
        this.offsets = new Array(nFeatures).fill(0);

        for (let j = 0; j < nFeatures; j++) {
            let minV = X[0][j];
            let maxV = X[0][j];
            for (let i = 1; i < X.length; i++) {
                minV = Math.min(minV, X[i][j]);
                maxV = Math.max(maxV, X[i][j]);
            }
            this.dataMin[j] = minV;

            const dataRange = maxV - minV;
            if (dataRange === 0) {
                this.scales[j] = 0;
                this.offsets[j] = targetMin;
            } else {
                this.scales[j] = targetRange / dataRange;
                this.offsets[j] = targetMin - minV * this.scales[j];
            }
        }
        this.fitted = true;
    }

    private assertFittedAndShape(X: number[][]): void {
        if (!this.fitted) {
            throw new Error('MinMaxScaler must be fitted before calling transform');
        }
        validateMatrix(X);
        if (X[0].length !== this.dataMin.length) {
            throw new Error('X has different number of features than fitted data');
        }
    }

    public transform(X: number[][]): number[][] {
        this.assertFittedAndShape(X);
        return X.map(row =>
            row.map((value, j) => value * this.scales[j] + this.offsets[j])
        );
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }

    public inverseTransform(X: number[][]): number[][] {
        this.assertFittedAndShape(X);
        return X.map(row =>
            row.map((value, j) => {
                const scale = this.scales[j];
                if (scale === 0) {
                    return this.dataMin[j];
                }
                return (value - this.offsets[j]) / scale;
            })
        );
    }
}
