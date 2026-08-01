import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { averageImportances, DecisionTreeClassifier, DecisionTreeProps } from '../tree';
import { SubsetSizeOption } from '../utils/paramResolvers';
import { definedProps } from './utils';
import { fitForest, predictForestClassification } from './forest';

export interface RandomForestClassifierProps extends DecisionTreeProps {
    nEstimators?: number;
    bootstrap?: boolean;
    maxFeatures?: SubsetSizeOption;
    randomState?: number;
}

export class RandomForestClassifier extends ClassifierBase {
    private nEstimators: number;
    private bootstrap: boolean;
    private maxFeatures: SubsetSizeOption;
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
        this.treeProps = definedProps(treeProps);
        this.estimators = [];
        this.fitted = false;
    }

    public getParams(): Params {
        return {
            nEstimators: this.nEstimators,
            bootstrap: this.bootstrap,
            maxFeatures: this.maxFeatures,
            randomState: this.randomState,
            max_depth: this.treeProps.max_depth,
            min_samples_split: this.treeProps.min_samples_split,
            criterion: this.treeProps.criterion,
            max_features: this.treeProps.max_features,
        };
    }

    public fit(trainX: number[][], trainY: number[]): void {
        this.estimators = fitForest(trainX, trainY, {
            nEstimators: this.nEstimators,
            bootstrap: this.bootstrap,
            randomState: this.randomState,
            createEstimator: randomState => new DecisionTreeClassifier({
                criterion: 'gini',
                ...this.treeProps,
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
        return predictForestClassification(this.estimators, testX);
    }
    public get featureImportances(): number[] {
        if (!this.fitted) throw new Error('model is not fitted');
        // sklearn averages each tree's normalized vector. Normalizing the
        // aggregate again also handles single-node (all-zero) trees.
        return averageImportances(this.estimators.map(tree => tree.featureImportances));
    }
}
registerEstimator('RandomForestClassifier', RandomForestClassifier);
