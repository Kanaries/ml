import { assert } from "../utils";
import { mean } from "../utils/stat";
import { IDTree } from "./decisionTreeClassifier";
import { filterWithIndices } from "./utils";
interface IRegTree extends IDTree {
}
interface RegressionTreeProps {
    max_depth?: number;
    min_samples_split?: number;
}
export class DecisionTreeRegressor {
    private feature_number: number;
    private regTree: IRegTree;
    private min_sample_split: number;
    private max_depth: number;
    public constructor (props: RegressionTreeProps = {}) {
        const {
            max_depth = Infinity,
            min_samples_split = 2
        } = props;
        this.max_depth = max_depth;
        this.min_sample_split = min_samples_split;
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
        for (let feaIndex = 0; feaIndex < this.feature_number; feaIndex++ ) {
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
        console.log(tree, sampleX, sampleY, selection)
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