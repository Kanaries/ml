import { ClassifierBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { averageImportances, ExtraTreeClassifier } from '../tree';
import type { ExtraTreeProps } from '../tree/extraTreeClassifier';
import { SubsetSizeOption } from '../utils/paramResolvers';
import { fitForest } from './forest';

export interface ExtraTreesClassifierProps extends ExtraTreeProps {
    nEstimators?: number;
    bootstrap?: boolean;
    randomState?: number;
}

export class ExtraTreesClassifier extends ClassifierBase {
    private nEstimators: number;
    private bootstrap: boolean;
    private maxFeatures: SubsetSizeOption;
    private randomState?: number;
    private treeProps: ExtraTreeProps;
    private estimators: ExtraTreeClassifier[] = [];
    private classesState: number[] = [];

    constructor(props: ExtraTreesClassifierProps = {}) {
        super();
        const { nEstimators = 100, bootstrap = false, max_features = 'sqrt', randomState, ...treeProps } = props;
        this.nEstimators = nEstimators;
        this.bootstrap = bootstrap;
        this.maxFeatures = max_features;
        this.randomState = randomState;
        this.treeProps = treeProps;
    }

    public getParams(): Params {
        return { nEstimators: this.nEstimators, bootstrap: this.bootstrap, max_features: this.maxFeatures, randomState: this.randomState, ...this.treeProps };
    }

    public fit(X: number[][], y: number[]): void {
        this.classesState = Array.from(new Set(y)).sort((a, b) => a - b);
        this.estimators = fitForest(X, y, {
            nEstimators: this.nEstimators,
            bootstrap: this.bootstrap,
            randomState: this.randomState,
            createEstimator: seed => new ExtraTreeClassifier({ criterion: 'gini', ...this.treeProps, max_features: this.maxFeatures, randomState: seed }),
        });
    }

    public predict(X: number[][]): number[] {
        return this.predictProba(X).map(row => this.classesState[row.indexOf(Math.max(...row))]);
    }

    public predictProba(X: number[][]): number[][] {
        if (this.estimators.length === 0) throw new Error('model is not fitted');
        const sums = Array.from({ length: X.length }, () => new Array(this.classesState.length).fill(0));
        for (const estimator of this.estimators) {
            const localClasses = estimator.classes;
            const probabilities = estimator.predictProba(X);
            for (let i = 0; i < X.length; i++) for (let local = 0; local < localClasses.length; local++) {
                const global = this.classesState.indexOf(localClasses[local]);
                sums[i][global] += probabilities[i][local] / this.estimators.length;
            }
        }
        return sums;
    }

    public get classes(): number[] { return this.classesState.slice(); }

    public get featureImportances(): number[] {
        if (this.estimators.length === 0) throw new Error('model is not fitted');
        return averageImportances(this.estimators.map(tree => tree.featureImportances));
    }
}
registerEstimator('ExtraTreesClassifier', ExtraTreesClassifier);
