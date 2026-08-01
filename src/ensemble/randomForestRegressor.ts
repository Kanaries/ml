import { RegressorBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { averageImportances, DecisionTreeRegressor } from '../tree';
import { SubsetSizeOption } from '../utils/paramResolvers';
import { fitForest, predictForestRegression } from './forest';

export interface RandomForestRegressorProps {
    nEstimators?: number;
    bootstrap?: boolean;
    maxDepth?: number;
    minSamplesSplit?: number;
    maxFeatures?: SubsetSizeOption;
    randomState?: number;
}

export class RandomForestRegressor extends RegressorBase {
    private nEstimators: number;
    private bootstrap: boolean;
    private maxDepth?: number;
    private minSamplesSplit?: number;
    private maxFeatures: SubsetSizeOption;
    private randomState?: number;
    private estimators: DecisionTreeRegressor[];
    private fitted: boolean;

    constructor(props: RandomForestRegressorProps = {}) {
        super();
        const {
            nEstimators = 100,
            bootstrap = true,
            maxDepth,
            minSamplesSplit,
            // sklearn's RandomForestRegressor default is all features
            maxFeatures = 'all',
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

    public getParams(): Params {
        return {
            nEstimators: this.nEstimators,
            bootstrap: this.bootstrap,
            maxDepth: this.maxDepth,
            minSamplesSplit: this.minSamplesSplit,
            maxFeatures: this.maxFeatures,
            randomState: this.randomState,
        };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        this.estimators = fitForest(trainX, trainY, {
            nEstimators: this.nEstimators,
            bootstrap: this.bootstrap,
            randomState: this.randomState,
            createEstimator: randomState => new DecisionTreeRegressor({
                max_depth: this.maxDepth,
                min_samples_split: this.minSamplesSplit,
                max_features: this.maxFeatures,
                randomState,
            }),
        });
        this.fitted = true;
    }

    public predict(testX: number[][]): number[] {
        if (!this.fitted) {
            throw new Error('model is not fitted');
        }
        return predictForestRegression(this.estimators, testX);
    }
    public get featureImportances(): number[] {
        if (!this.fitted) throw new Error('model is not fitted');
        return averageImportances(this.estimators.map(tree => tree.featureImportances));
    }
}
registerEstimator('RandomForestRegressor', RandomForestRegressor);
