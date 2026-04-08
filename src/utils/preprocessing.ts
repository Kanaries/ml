export interface StandardScalerProps {
    withMean?: boolean;
    withStd?: boolean;
}

export interface MinMaxScalerProps {
    featureRange?: [number, number];
}

export interface NormalizerProps {
    norm?: 'l1' | 'l2' | 'max';
}

export interface BinarizerProps {
    threshold?: number;
}

export interface SimpleImputerProps {
    strategy?: 'mean' | 'median' | 'mostFrequent' | 'constant';
    fillValue?: number;
    missingValues?: number | null;
}

export interface VarianceThresholdProps {
    threshold?: number;
}

export type FeatureScoreFunc = (X: number[][], y: number[]) => number[];

export interface SelectKBestProps {
    k?: number;
    scoreFunc?: FeatureScoreFunc;
}

export type CategoricalValue = string | number | boolean | null;

export interface OneHotEncoderProps {
    drop?: 'none' | 'first' | 'ifBinary';
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

function validateCategoricalMatrix(X: CategoricalValue[][]): void {
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

function assertSameFeatureCount(X: number[][], featureCount: number, message: string): void {
    validateMatrix(X);
    if (X[0].length !== featureCount) {
        throw new Error(message);
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

export class MaxAbsScaler {
    private maxAbs: number[];
    private fitted: boolean;

    constructor() {
        this.maxAbs = [];
        this.fitted = false;
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nFeatures = X[0].length;
        this.maxAbs = new Array(nFeatures).fill(0);
        for (let j = 0; j < nFeatures; j++) {
            let maxValue = 0;
            for (let i = 0; i < X.length; i++) {
                maxValue = Math.max(maxValue, Math.abs(X[i][j]));
            }
            this.maxAbs[j] = maxValue === 0 ? 1 : maxValue;
        }
        this.fitted = true;
    }

    public transform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('MaxAbsScaler must be fitted before calling transform');
        }
        assertSameFeatureCount(X, this.maxAbs.length, 'X has different number of features than fitted data');
        return X.map(row => row.map((value, j) => value / this.maxAbs[j]));
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }

    public inverseTransform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('MaxAbsScaler must be fitted before calling inverseTransform');
        }
        assertSameFeatureCount(X, this.maxAbs.length, 'X has different number of features than fitted data');
        return X.map(row => row.map((value, j) => value * this.maxAbs[j]));
    }
}

export class Normalizer {
    private norm: 'l1' | 'l2' | 'max';

    constructor(props: NormalizerProps = {}) {
        this.norm = props.norm || 'l2';
    }

    public fit(_X: number[][]): void {}

    public transform(X: number[][]): number[][] {
        validateMatrix(X);
        return X.map(row => {
            let scale = 1;
            if (this.norm === 'l1') {
                scale = row.reduce((sum, value) => sum + Math.abs(value), 0);
            } else if (this.norm === 'l2') {
                scale = Math.sqrt(row.reduce((sum, value) => sum + value * value, 0));
            } else {
                scale = row.reduce((maxValue, value) => Math.max(maxValue, Math.abs(value)), 0);
            }
            if (scale === 0) {
                return row.map(() => 0);
            }
            return row.map(value => value / scale);
        });
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }
}

export class LabelEncoder {
    private classes: number[] = [];
    private fitted = false;

    public fit(y: number[]): void {
        if (y.length === 0) {
            throw new Error('y must be non-empty');
        }
        this.classes = Array.from(new Set(y)).sort((a, b) => a - b);
        this.fitted = true;
    }

    public transform(y: number[]): number[] {
        if (!this.fitted) {
            throw new Error('LabelEncoder must be fitted before calling transform');
        }
        const indices = new Map<number, number>();
        this.classes.forEach((value, index) => indices.set(value, index));
        return y.map(value => {
            const encoded = indices.get(value);
            if (encoded === undefined) {
                throw new Error(`Unknown label ${value}`);
            }
            return encoded;
        });
    }

    public fitTransform(y: number[]): number[] {
        this.fit(y);
        return this.transform(y);
    }

    public inverseTransform(y: number[]): number[] {
        if (!this.fitted) {
            throw new Error('LabelEncoder must be fitted before calling inverseTransform');
        }
        return y.map(value => {
            if (!Number.isInteger(value) || value < 0 || value >= this.classes.length) {
                throw new Error(`Encoded label ${value} is out of range`);
            }
            return this.classes[value];
        });
    }
}

export class Binarizer {
    private threshold: number;

    constructor(props: BinarizerProps = {}) {
        this.threshold = props.threshold ?? 0;
    }

    public fit(_X: number[][]): void {}

    public transform(X: number[][]): number[][] {
        validateMatrix(X);
        return X.map(row => row.map(value => (value > this.threshold ? 1 : 0)));
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }
}

export class OrdinalEncoder {
    private categories: CategoricalValue[][];
    private fitted: boolean;

    constructor() {
        this.categories = [];
        this.fitted = false;
    }

    public fit(X: CategoricalValue[][]): void {
        validateCategoricalMatrix(X);
        const nFeatures = X[0].length;
        this.categories = new Array(nFeatures).fill(null).map(() => []);
        for (let j = 0; j < nFeatures; j++) {
            const values = Array.from(new Set(X.map(row => row[j])));
            values.sort((a, b) => String(a).localeCompare(String(b)));
            this.categories[j] = values;
        }
        this.fitted = true;
    }

    public transform(X: CategoricalValue[][]): number[][] {
        if (!this.fitted) {
            throw new Error('OrdinalEncoder must be fitted before calling transform');
        }
        validateCategoricalMatrix(X);
        if (X[0].length !== this.categories.length) {
            throw new Error('X has different number of features than fitted data');
        }
        return X.map(row =>
            row.map((value, j) => {
                const index = this.categories[j].indexOf(value);
                if (index === -1) {
                    throw new Error(`Unknown category ${String(value)} in column ${j}`);
                }
                return index;
            })
        );
    }

    public fitTransform(X: CategoricalValue[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }

    public inverseTransform(X: number[][]): CategoricalValue[][] {
        if (!this.fitted) {
            throw new Error('OrdinalEncoder must be fitted before calling inverseTransform');
        }
        assertSameFeatureCount(X, this.categories.length, 'X has different number of features than fitted data');
        return X.map(row =>
            row.map((value, j) => {
                if (!Number.isInteger(value) || value < 0 || value >= this.categories[j].length) {
                    throw new Error(`Encoded category ${value} is out of range for column ${j}`);
                }
                return this.categories[j][value];
            })
        );
    }
}

export class OneHotEncoder {
    private drop: 'none' | 'first' | 'ifBinary';
    private categories: CategoricalValue[][];
    private retainedCategories: CategoricalValue[][];
    private fitted: boolean;

    constructor(props: OneHotEncoderProps = {}) {
        this.drop = props.drop ?? 'none';
        this.categories = [];
        this.retainedCategories = [];
        this.fitted = false;
    }

    private categoriesToEncode(values: CategoricalValue[]): CategoricalValue[] {
        if (this.drop === 'none') {
            return values.slice();
        }
        if (this.drop === 'first') {
            return values.slice(1);
        }
        return values.length === 2 ? values.slice(1) : values.slice();
    }

    public fit(X: CategoricalValue[][]): void {
        validateCategoricalMatrix(X);
        const nFeatures = X[0].length;
        this.categories = new Array(nFeatures).fill(null).map(() => []);
        this.retainedCategories = new Array(nFeatures).fill(null).map(() => []);
        for (let j = 0; j < nFeatures; j++) {
            const values = Array.from(new Set(X.map(row => row[j])));
            values.sort((a, b) => String(a).localeCompare(String(b)));
            this.categories[j] = values;
            this.retainedCategories[j] = this.categoriesToEncode(values);
        }
        this.fitted = true;
    }

    public transform(X: CategoricalValue[][]): number[][] {
        if (!this.fitted) {
            throw new Error('OneHotEncoder must be fitted before calling transform');
        }
        validateCategoricalMatrix(X);
        if (X[0].length !== this.categories.length) {
            throw new Error('X has different number of features than fitted data');
        }
        return X.map(row => {
            const encoded: number[] = [];
            for (let j = 0; j < row.length; j++) {
                const categoryIndex = this.categories[j].indexOf(row[j]);
                if (categoryIndex === -1) {
                    throw new Error(`Unknown category ${String(row[j])} in column ${j}`);
                }
                for (const retained of this.retainedCategories[j]) {
                    encoded.push(row[j] === retained ? 1 : 0);
                }
            }
            return encoded;
        });
    }

    public fitTransform(X: CategoricalValue[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }

    public inverseTransform(X: number[][]): CategoricalValue[][] {
        if (!this.fitted) {
            throw new Error('OneHotEncoder must be fitted before calling inverseTransform');
        }
        validateMatrix(X);
        const expectedWidth = this.retainedCategories.reduce((sum, values) => sum + values.length, 0);
        if (X[0].length !== expectedWidth) {
            throw new Error('X has different number of encoded features than fitted data');
        }

        return X.map(row => {
            const decoded: CategoricalValue[] = [];
            let offset = 0;
            for (let j = 0; j < this.categories.length; j++) {
                const retained = this.retainedCategories[j];
                const width = retained.length;
                const slice = row.slice(offset, offset + width);
                offset += width;

                if (width === 0) {
                    decoded.push(this.categories[j][0]);
                    continue;
                }

                const activeIndex = slice.findIndex(value => value === 1);
                if (activeIndex !== -1) {
                    decoded.push(retained[activeIndex]);
                    continue;
                }

                const dropped = this.categories[j].find(category => !retained.includes(category));
                decoded.push(dropped === undefined ? this.categories[j][0] : dropped);
            }
            return decoded;
        });
    }
}

function isMissingValue(value: number, missingValues: number | null): boolean {
    if (missingValues === null) {
        return value === null;
    }
    return Number.isNaN(missingValues) ? Number.isNaN(value) : value === missingValues;
}

function median(values: number[]): number {
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
}

function mostFrequent(values: number[]): number {
    const counts = new Map<number, number>();
    let bestValue = values[0];
    let bestCount = 0;
    for (const value of values) {
        const count = (counts.get(value) || 0) + 1;
        counts.set(value, count);
        if (count > bestCount || (count === bestCount && value < bestValue)) {
            bestValue = value;
            bestCount = count;
        }
    }
    return bestValue;
}

export class SimpleImputer {
    private strategy: 'mean' | 'median' | 'mostFrequent' | 'constant';
    private fillValue: number;
    private missingValues: number | null;
    private statistics: number[];
    private fitted: boolean;

    constructor(props: SimpleImputerProps = {}) {
        const { strategy = 'mean', fillValue = 0, missingValues = Number.NaN } = props;
        this.strategy = strategy;
        this.fillValue = fillValue;
        this.missingValues = missingValues;
        this.statistics = [];
        this.fitted = false;
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nFeatures = X[0].length;
        this.statistics = new Array(nFeatures).fill(0);

        for (let j = 0; j < nFeatures; j++) {
            const observed: number[] = [];
            for (let i = 0; i < X.length; i++) {
                const value = X[i][j];
                if (!isMissingValue(value, this.missingValues)) {
                    observed.push(value);
                }
            }

            if (this.strategy === 'constant') {
                this.statistics[j] = this.fillValue;
                continue;
            }
            if (observed.length === 0) {
                throw new Error(`Feature ${j} has no observed values`);
            }
            if (this.strategy === 'mean') {
                this.statistics[j] = observed.reduce((sum, value) => sum + value, 0) / observed.length;
            } else if (this.strategy === 'median') {
                this.statistics[j] = median(observed);
            } else {
                this.statistics[j] = mostFrequent(observed);
            }
        }
        this.fitted = true;
    }

    public transform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('SimpleImputer must be fitted before calling transform');
        }
        assertSameFeatureCount(X, this.statistics.length, 'X has different number of features than fitted data');
        return X.map(row =>
            row.map((value, j) => (isMissingValue(value, this.missingValues) ? this.statistics[j] : value))
        );
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }
}

export class VarianceThreshold {
    private threshold: number;
    private selectedIndices: number[];
    private fitted: boolean;

    constructor(props: VarianceThresholdProps = {}) {
        this.threshold = props.threshold ?? 0;
        this.selectedIndices = [];
        this.fitted = false;
    }

    public fit(X: number[][]): void {
        validateMatrix(X);
        const nFeatures = X[0].length;
        this.selectedIndices = [];
        for (let j = 0; j < nFeatures; j++) {
            let mean = 0;
            for (let i = 0; i < X.length; i++) {
                mean += X[i][j];
            }
            mean /= X.length;

            let variance = 0;
            for (let i = 0; i < X.length; i++) {
                const diff = X[i][j] - mean;
                variance += diff * diff;
            }
            variance /= X.length;
            if (variance > this.threshold) {
                this.selectedIndices.push(j);
            }
        }
        if (this.selectedIndices.length === 0) {
            throw new Error('No feature meets the variance threshold');
        }
        this.fitted = true;
    }

    public transform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('VarianceThreshold must be fitted before calling transform');
        }
        validateMatrix(X);
        return X.map(row => this.selectedIndices.map(index => row[index]));
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }
}

export function fRegression(X: number[][], y: number[]): number[] {
    validateMatrix(X);
    if (X.length !== y.length) {
        throw new Error('X and y must have the same length');
    }
    const nSamples = X.length;
    const nFeatures = X[0].length;
    const yMean = y.reduce((sum, value) => sum + value, 0) / nSamples;
    const yVariance = y.reduce((sum, value) => sum + (value - yMean) ** 2, 0);

    return Array.from({ length: nFeatures }, (_, j) => {
        let xMean = 0;
        for (let i = 0; i < nSamples; i++) {
            xMean += X[i][j];
        }
        xMean /= nSamples;

        let covariance = 0;
        let xVariance = 0;
        for (let i = 0; i < nSamples; i++) {
            const xDiff = X[i][j] - xMean;
            const yDiff = y[i] - yMean;
            covariance += xDiff * yDiff;
            xVariance += xDiff * xDiff;
        }

        if (xVariance === 0 || yVariance === 0) {
            return 0;
        }
        const correlation = covariance / Math.sqrt(xVariance * yVariance);
        const rSquared = Math.max(0, Math.min(1, correlation * correlation));
        if (rSquared >= 1) {
            return Number.POSITIVE_INFINITY;
        }
        return (rSquared / (1 - rSquared)) * (nSamples - 2);
    });
}

export class SelectKBest {
    private k: number;
    private scoreFunc: FeatureScoreFunc;
    private selectedIndices: number[];
    private fitted: boolean;

    constructor(props: SelectKBestProps = {}) {
        this.k = props.k ?? 10;
        this.scoreFunc = props.scoreFunc || fRegression;
        this.selectedIndices = [];
        this.fitted = false;
    }

    public fit(X: number[][], y: number[]): void {
        validateMatrix(X);
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }
        const nFeatures = X[0].length;
        if (!Number.isInteger(this.k) || this.k <= 0 || this.k > nFeatures) {
            throw new Error('k must be an integer between 1 and the number of features');
        }
        const scores = this.scoreFunc(X, y);
        if (scores.length !== nFeatures) {
            throw new Error('scoreFunc must return one score per feature');
        }
        this.selectedIndices = scores
            .map((score, index) => ({ score, index }))
            .sort((a, b) => {
                if (b.score === a.score) {
                    return a.index - b.index;
                }
                return b.score - a.score;
            })
            .slice(0, this.k)
            .map(item => item.index)
            .sort((a, b) => a - b);
        this.fitted = true;
    }

    public transform(X: number[][]): number[][] {
        if (!this.fitted) {
            throw new Error('SelectKBest must be fitted before calling transform');
        }
        validateMatrix(X);
        return X.map(row => this.selectedIndices.map(index => row[index]));
    }

    public fitTransform(X: number[][], y: number[]): number[][] {
        this.fit(X, y);
        return this.transform(X);
    }
}
