import { RegressorBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { averageImportances, ExtraTreeRegressor } from '../tree';
import type { ExtraTreeRegressorProps } from '../tree/extraTreeRegressor';
import { SubsetSizeOption } from '../utils/paramResolvers';
import { fitForest, predictForestRegression } from './forest';

export interface ExtraTreesRegressorProps extends ExtraTreeRegressorProps {
    nEstimators?: number;
    bootstrap?: boolean;
    randomState?: number;
}

export class ExtraTreesRegressor extends RegressorBase {
    private nEstimators: number;
    private bootstrap: boolean;
    private maxFeatures: SubsetSizeOption;
    private randomState?: number;
    private treeProps: ExtraTreeRegressorProps;
    private estimators: ExtraTreeRegressor[] = [];

    constructor(props: ExtraTreesRegressorProps = {}) {
        super();
        const { nEstimators = 100, bootstrap = false, max_features = 'all', randomState, ...treeProps } = props;
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
        this.estimators = fitForest(X, y, {
            nEstimators: this.nEstimators,
            bootstrap: this.bootstrap,
            randomState: this.randomState,
            createEstimator: seed => new ExtraTreeRegressor({ ...this.treeProps, max_features: this.maxFeatures, randomState: seed }),
        });
    }

    public predict(X: number[][]): number[] {
        if (this.estimators.length === 0) throw new Error('model is not fitted');
        return predictForestRegression(this.estimators, X);
    }

    public get featureImportances(): number[] {
        if (this.estimators.length === 0) throw new Error('model is not fitted');
        return averageImportances(this.estimators.map(tree => tree.featureImportances));
    }
}
registerEstimator('ExtraTreesRegressor', ExtraTreesRegressor);
