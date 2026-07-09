import { DecisionTreeRegressor } from '../tree';
import { createRandomGenerator } from '../utils';

export interface RandomForestRegressorProps {
    nEstimators?: number;
    bootstrap?: boolean;
    maxDepth?: number;
    minSamplesSplit?: number;
    maxFeatures?: number | 'sqrt' | 'log2';
    randomState?: number;
}

export class RandomForestRegressor {
    private nEstimators: number;
    private bootstrap: boolean;
    private maxDepth?: number;
    private minSamplesSplit?: number;
    private maxFeatures: number | 'sqrt' | 'log2';
    private randomState?: number;
    private estimators: DecisionTreeRegressor[];
    private fitted: boolean;

    constructor(props: RandomForestRegressorProps = {}) {
        const {
            nEstimators = 100,
            bootstrap = true,
            maxDepth,
            minSamplesSplit,
            maxFeatures = 1.0,
            randomState,
        } = props;
        this.nEstimators = nEstimators;
        this.bootstrap = bootstrap;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.maxFeatures = maxFeatures;
        this.randomState = randomState;
        this.estimators = [];
        this.fitted = false;
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
            const tree = new DecisionTreeRegressor({
                max_depth: this.maxDepth,
                min_samples_split: this.minSamplesSplit,
                max_features: this.maxFeatures,
                randomState: Math.floor(random() * 1_000_000_000),
            });
            const sampleX: number[][] = [];
            const sampleY: number[] = [];
            if (this.bootstrap) {
                for (let j = 0; j < trainX.length; j++) {
                    const index = Math.floor(random() * trainX.length);
                    sampleX.push(trainX[index]);
                    sampleY.push(trainY[index]);
                }
            } else {
                sampleX.push(...trainX);
                sampleY.push(...trainY);
            }
            tree.fit(sampleX, sampleY);
            this.estimators.push(tree);
        }
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        return testX.map((_, rowIndex) => {
            let sum = 0;
            for (const estimator of this.estimators) {
                sum += estimator.predict([testX[rowIndex]])[0];
            }
            return sum / this.estimators.length;
        });
    }
}
