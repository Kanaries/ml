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

export interface RandomForestClassifierProps extends DecisionTreeProps {
    nEstimators?: number;
    bootstrap?: boolean;
    maxFeatures?: number | 'sqrt' | 'log2';
    randomState?: number;
}

export class RandomForestClassifier extends ClassifierBase {
    private nEstimators: number;
    private bootstrap: boolean;
    private maxFeatures: number | 'sqrt' | 'log2';
    private randomState?: number;
    private treeProps: DecisionTreeProps;
    private estimators: DecisionTreeClassifier[];
    private fitted: boolean;

    constructor(props: RandomForestClassifierProps = {}) {
        super();
        const { nEstimators = 100, bootstrap = true, maxFeatures = 'sqrt', randomState, ...treeProps } = props;
        this.nEstimators = nEstimators;
        this.bootstrap = bootstrap;
        this.maxFeatures = maxFeatures;
        this.randomState = randomState;
        this.treeProps = treeProps;
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
            const tree = new DecisionTreeClassifier({
                criterion: 'gini',
                ...this.treeProps,
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
