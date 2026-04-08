import { assert } from "../utils";
import { mean } from "../utils/stat";
import { IDTree } from "./decisionTreeClassifier";
import { filterWithIndices } from "./utils";
interface IRegTree extends IDTree {
}
interface RegressionTreeProps {
    max_depth?: number;
    min_samples_split?: number;
    max_features?: number | 'sqrt' | 'log2';
    randomState?: number;
}
export class DecisionTreeRegressor {
    private feature_number: number;
    private regTree: IRegTree;
    private min_sample_split: number;
    private max_depth: number;
    private max_features?: number | 'sqrt' | 'log2';
    private randomState?: number;
    private random: () => number;
    public constructor (props: RegressionTreeProps = {}) {
        const {
            max_depth = Infinity,
            min_samples_split = 2,
            max_features,
            randomState
        } = props;
        this.max_depth = max_depth;
        this.min_sample_split = min_samples_split;
        this.max_features = max_features;
        this.randomState = randomState;
        this.random = this.createRandomGenerator(this.randomState);
    }
    private createRandomGenerator(seed?: number): () => number {
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
    private selectedFeatureIndices(): number[] {
        let size = this.feature_number;
        if (this.max_features !== undefined) {
            if (this.max_features === 'sqrt') {
                size = Math.max(1, Math.ceil(Math.sqrt(this.feature_number)));
            } else if (this.max_features === 'log2') {
                size = Math.max(1, Math.ceil(Math.log2(this.feature_number)));
            } else if (this.max_features > 0 && this.max_features <= 1) {
                size = Math.max(1, Math.ceil(this.feature_number * this.max_features));
            } else {
                size = Math.max(1, Math.min(this.feature_number, Math.floor(this.max_features)));
            }
        }
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
            sum += (values[i] - _mean) ** 2
        }
        return sum / values.length;
    }
    private attributeSelection (sampleX: number[][], sampleY: number[]) {
        let minErr = Infinity;
        let minErrFeaIndex = -1; // bad case none
        let minErrValue = 0;
        const featureIndices = this.selectedFeatureIndices();
        for (let feaIndex of featureIndices ) {
            const values: number[] = sampleX.map(x => x[feaIndex]);
            const valueSet: Set<number> = new Set(values);
            let localMinErr = Infinity;
            let localMinErrValue = 0;
            for (let feaValue of valueSet) {
                const leftChild = filterWithIndices(values, x => x < feaValue);
                const rightChild = filterWithIndices(values, x => x >= feaValue);
                const err = this.calErr(leftChild.indices.map(i => sampleY[i])) + this.calErr(rightChild.indices.map(i => sampleY[i]));
                if (err < localMinErr) {
                    localMinErr = err;
                    localMinErrValue = feaValue;
                }
            }
            if (localMinErr < minErr) {
                minErr = localMinErr;
                minErrFeaIndex = feaIndex;
                minErrValue = localMinErrValue;
            }
        }
        return {
            minErr,
            minErrFeaIndex,
            minErrValue
        }
    }
    private initTreeNode (sampleY: number[]): IRegTree {
        return {
            splitIndex: -1,
            nodeValue: -1,
            leftChild: null,
            rightChild: null,
            y: mean(sampleY)
        };
    }
    private buildTree (tree: IRegTree, sampleX: number[][], sampleY: number[], depth: number) {
        if (sampleX.length < this.min_sample_split) return;
        const nodeErr = this.calErr(sampleY);
        // if (nodeErr < 0.00001) return;
        const selection = this.attributeSelection(sampleX, sampleY);
        if (selection.minErrFeaIndex === -1) return;
        // console.log(tree, sampleX, sampleY, selection)
        // if (Math.abs(nodeErr - selection.minErr) < 0.00001) return;
        const values = sampleX.map(x => x[selection.minErrFeaIndex]);
        let leftSamples = filterWithIndices(values, v => v < selection.minErrValue);
        let rightSamples = filterWithIndices(values, v => v >= selection.minErrValue);
        tree.splitIndex = selection.minErrFeaIndex;
        tree.nodeValue = selection.minErrValue;

        tree.leftChild = this.initTreeNode(leftSamples.indices.map(i => sampleY[i]));
        tree.rightChild = this.initTreeNode(rightSamples.indices.map(i => sampleY[i]))
        this.buildTree(
            tree.leftChild,
            leftSamples.indices.map((i) => sampleX[i]),
            leftSamples.indices.map((i) => sampleY[i]),
            depth + 1
        );
        this.buildTree(
            tree.rightChild,
            rightSamples.indices.map((i) => sampleX[i]),
            rightSamples.indices.map((i) => sampleY[i]),
            depth + 1
        );
    }
    public fit (sampleX: number[][], sampleY: number[]): void {
        assert(sampleX.length > 0, 'fit data should not be empty');
        this.random = this.createRandomGenerator(this.randomState);
        this.feature_number = sampleX[0].length;
        this.regTree = this.initTreeNode(sampleY);
        this.buildTree(this.regTree, sampleX, sampleY, 0);
    }
    private findSample(X: number[], tree: IRegTree, depth: number) {
        if (tree.splitIndex === -1 || depth > this.max_depth) return tree.y;
        if (X[tree.splitIndex] < tree.nodeValue) {
            return this.findSample(X, tree.leftChild, depth + 1);
        } else {
            return this.findSample(X, tree.rightChild, depth + 1);
        }
    }
    public predict (sampleX: number[][]): number[] {
        return sampleX.map(x => this.findSample(x, this.regTree, 0));
    }
}
