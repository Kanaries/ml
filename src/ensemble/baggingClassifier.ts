import { ClassifierBase } from '../base';
import { DecisionTreeClassifier, DecisionTreeProps } from '../tree';

function createRandomGenerator(seed?: number): () => number {
    if (seed === undefined) {
        return Math.random;
    }
    let state = Math.floor(seed) % 2147483647;
    if (state <= 0) {
        state += 2147483646;
    }
    return () => {
        state = (state * 16807) % 2147483647;
        return (state - 1) / 2147483646;
    };
}

function bootstrapSample(X: number[][], y: number[], random: () => number, sampleCount?: number) {
    const size = sampleCount ?? X.length;
    const sampleX: number[][] = [];
    const sampleY: number[] = [];
    for (let i = 0; i < size; i++) {
        const index = Math.floor(random() * X.length);
        sampleX.push(X[index]);
        sampleY.push(y[index]);
    }
    return { sampleX, sampleY };
}

export interface BaggingClassifierProps extends DecisionTreeProps {
    nEstimators?: number;
    maxSamples?: number;
    bootstrap?: boolean;
    randomState?: number;
    estimatorFactory?: (seed?: number) => { fit(X: number[][], y: number[]): void; predict(X: number[][]): number[] };
}

export class BaggingClassifier extends ClassifierBase {
    private nEstimators: number;
    private maxSamples?: number;
    private bootstrap: boolean;
    private randomState?: number;
    private estimatorFactory?: (seed?: number) => { fit(X: number[][], y: number[]): void; predict(X: number[][]): number[] };
    private treeProps: DecisionTreeProps;
    private estimators: Array<{ fit(X: number[][], y: number[]): void; predict(X: number[][]): number[] }>;
    private fitted: boolean;

    constructor(props: BaggingClassifierProps = {}) {
        super();
        const { nEstimators = 10, maxSamples, bootstrap = true, randomState, estimatorFactory, ...treeProps } = props;
        this.nEstimators = nEstimators;
        this.maxSamples = maxSamples;
        this.bootstrap = bootstrap;
        this.randomState = randomState;
        this.estimatorFactory = estimatorFactory;
        this.treeProps = treeProps;
        this.estimators = [];
        this.fitted = false;
    }

    private defaultEstimator(seed?: number) {
        return new DecisionTreeClassifier({ criterion: 'gini', ...this.treeProps, randomState: seed });
    }

    public fit(trainX: number[][], trainY: number[]): void {
        if (trainX.length === 0 || trainY.length === 0) {
            throw new Error('X and y must be non-empty');
        }
        if (trainX.length !== trainY.length) {
            throw new Error('X and y must have the same length');
        }
        const random = createRandomGenerator(this.randomState);
        this.estimators = [];
        for (let i = 0; i < this.nEstimators; i++) {
            const estimatorSeed = Math.floor(random() * 1_000_000_000);
            const estimator = this.estimatorFactory ? this.estimatorFactory(estimatorSeed) : this.defaultEstimator(estimatorSeed);
            let sampleX: number[][];
            let sampleY: number[];
            if (this.bootstrap) {
                ({ sampleX, sampleY } = bootstrapSample(trainX, trainY, random, this.maxSamples));
            } else {
                const size = this.maxSamples ?? trainX.length;
                const indices = Array.from({ length: trainX.length }, (_, index) => index);
                for (let j = indices.length - 1; j > 0; j--) {
                    const k = Math.floor(random() * (j + 1));
                    [indices[j], indices[k]] = [indices[k], indices[j]];
                }
                sampleX = indices.slice(0, size).map(index => trainX[index]);
                sampleY = indices.slice(0, size).map(index => trainY[index]);
            }
            estimator.fit(sampleX, sampleY);
            this.estimators.push(estimator);
        }
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        return testX.map((_, rowIndex) => {
            const votes = new Map<number, number>();
            for (const estimator of this.estimators) {
                const label = estimator.predict([testX[rowIndex]])[0];
                votes.set(label, (votes.get(label) || 0) + 1);
            }
            let bestLabel = 0;
            let bestCount = -1;
            const labels = Array.from(votes.keys()).sort((a, b) => a - b);
            for (const label of labels) {
                const count = votes.get(label)!;
                if (count > bestCount) {
                    bestCount = count;
                    bestLabel = label;
                }
            }
            return bestLabel;
        });
    }
}
