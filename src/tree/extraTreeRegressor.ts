import { RegressorBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { assert, createRandomGenerator } from '../utils';
import { resolveSubsetSize, SubsetSizeOption } from '../utils/paramResolvers';
import { mean } from '../utils/stat';
import { IDTree } from './decisionTreeClassifier';
import { defineHiddenField, filterWithIndices, valuesAllSame } from './utils';

interface IRegTree extends IDTree {}

export interface ExtraTreeRegressorProps {
    max_depth?: number;
    min_samples_split?: number;
    splitter?: 'random';
    max_features?: SubsetSizeOption;
    randomState?: number;
}

export class ExtraTreeRegressor extends RegressorBase {
    private feature_number: number;
    private regTree: IRegTree;
    private min_samples_split: number;
    private max_depth: number;
    private splitter: 'random';
    private max_features: SubsetSizeOption;
    private randomState?: number;
    private random: () => number;

    public constructor(props: ExtraTreeRegressorProps = {}) {
        super();
        // sklearn's ExtraTreeRegressor defaults max_features to all features
        const { max_depth = Infinity, min_samples_split = 2, splitter = 'random', max_features = 'all', randomState } = props;
        this.max_depth = max_depth;
        this.min_samples_split = min_samples_split;
        this.splitter = splitter;
        this.max_features = max_features;
        this.randomState = randomState;
        // hidden (non-enumerable) so serialization only sees plain data
        defineHiddenField(this, 'random', createRandomGenerator(this.randomState));
    }

    public getParams(): Params {
        return {
            max_depth: this.max_depth,
            min_samples_split: this.min_samples_split,
            splitter: this.splitter,
            max_features: this.max_features,
            randomState: this.randomState,
        };
    }

    private getFeatureSubset(): number[] {
        const size = resolveSubsetSize(this.max_features, this.feature_number);
        const indices = Array.from({ length: this.feature_number }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(this.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        return indices.slice(0, size);
    }

    private calErr(values: number[]): number {
        const _mean = mean(values);
        let sum = 0;
        for (let i = 0; i < values.length; i++) {
            sum += (values[i] - _mean) ** 2;
        }
        return sum;
    }

    private attributeSelection(sampleX: number[][], sampleY: number[]) {
        let minErr = Infinity;
        let minErrFeaIndex = -1;
        let minErrValue = 0;
        const featureSubset = this.getFeatureSubset();
        for (let feaIndex of featureSubset) {
            const values: number[] = sampleX.map(x => x[feaIndex]);
            const min = Math.min(...values);
            const max = Math.max(...values);
            if (min === max) continue;
            const feaValue = this.random() * (max - min) + min;
            const leftChild = filterWithIndices(values, x => x < feaValue);
            const rightChild = filterWithIndices(values, x => x >= feaValue);
            const leftY = leftChild.indices.map(i => sampleY[i]);
            const rightY = rightChild.indices.map(i => sampleY[i]);
            const err = this.calErr(leftY) + this.calErr(rightY);
            if (err < minErr) {
                minErr = err;
                minErrFeaIndex = feaIndex;
                minErrValue = feaValue;
            }
        }
        return {
            minErr,
            minErrFeaIndex,
            minErrValue
        };
    }

    private initTreeNode(sampleY: number[]): IRegTree {
        return {
            splitIndex: -1,
            nodeValue: -1,
            leftChild: null,
            rightChild: null,
            y: mean(sampleY)
        };
    }

    private buildTree(tree: IRegTree, sampleX: number[][], sampleY: number[], depth: number) {
        if (sampleX.length < this.min_samples_split) return;
        if (depth >= this.max_depth) return;
        if (valuesAllSame(sampleY)) return;
        const nodeErr = this.calErr(sampleY);
        const selection = this.attributeSelection(sampleX, sampleY);
        if (selection.minErrFeaIndex === -1) return;
        if (selection.minErr >= nodeErr) return;
        const values = sampleX.map(x => x[selection.minErrFeaIndex]);
        const leftSamples = filterWithIndices(values, v => v < selection.minErrValue);
        const rightSamples = filterWithIndices(values, v => v >= selection.minErrValue);
        tree.splitIndex = selection.minErrFeaIndex;
        tree.nodeValue = selection.minErrValue;

        tree.leftChild = this.initTreeNode(leftSamples.indices.map(i => sampleY[i]));
        tree.rightChild = this.initTreeNode(rightSamples.indices.map(i => sampleY[i]));
        this.buildTree(
            tree.leftChild,
            leftSamples.indices.map(i => sampleX[i]),
            leftSamples.indices.map(i => sampleY[i]),
            depth + 1
        );
        this.buildTree(
            tree.rightChild,
            rightSamples.indices.map(i => sampleX[i]),
            rightSamples.indices.map(i => sampleY[i]),
            depth + 1
        );
    }

    public fit(sampleX: number[][], sampleY: number[]): void {
        assert(sampleX.length > 0, 'fit data should not be empty');
        defineHiddenField(this, 'random', createRandomGenerator(this.randomState));
        this.feature_number = sampleX[0].length;
        this.regTree = this.initTreeNode(sampleY);
        this.buildTree(this.regTree, sampleX, sampleY, 0);
    }

    private findSample(X: number[], tree: IRegTree): number {
        if (tree.splitIndex === -1) return tree.y;
        if (X[tree.splitIndex] < tree.nodeValue) {
            return this.findSample(X, tree.leftChild);
        } else {
            return this.findSample(X, tree.rightChild);
        }
    }

    public predict(sampleX: number[][]): number[] {
        return sampleX.map(x => this.findSample(x, this.regTree));
    }
}
registerEstimator('ExtraTreeRegressor', ExtraTreeRegressor);
