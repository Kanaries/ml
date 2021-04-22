import { entropy, gini, mode } from '../utils/stat';
import { filterWithIndices } from './utils';

export interface ISlice {
    X: number[][];
    Y: number[];
}
export interface IDTree {
    splitIndex: number;
    // 离散的情况下是当前节点的所有取值集合，连续则是一个区间
    nodeValue: number;
    y: number;
    leftChild: IDTree | null;
    rightChild: IDTree | null;
}
export interface DecisionTreeProps {
    max_depth?: number;
    min_samples_split?: number;
    criterion?: 'entropy' | 'gini';
}

class DecisionTreeBase {
    private dtree: IDTree | null;
    private max_depth: number;
    private feature_number: number;
    private min_samples_split: number;
    private criterion: 'entropy' | 'gini';
    private impurity: (freqs: number[]) => number;
    public constructor(props: DecisionTreeProps = {}) {
        const { max_depth = Infinity, criterion = 'entropy', min_samples_split = 2 } = props;
        this.dtree = null;
        this.criterion;
        this.max_depth = max_depth;
        this.criterion = criterion;
        this.feature_number = 0;
        this.min_samples_split = min_samples_split;
        if (criterion === 'entropy') {
            this.impurity = entropy;
        } else {
            this.impurity = gini;
        }
    }
    protected initTreeNode(sampleY: number[]): IDTree {
        return {
            splitIndex: -1,
            nodeValue: -1,
            leftChild: null,
            rightChild: null,
            y: -1,
        };
    }
    protected allowSplit(sampleX: number[][], sampleY: number[], depth: number): boolean {
        if (sampleX.length < this.min_samples_split) return false;
        if (depth >= this.max_depth) return false;
        return true;
    }
    private buildTree(tree: IDTree, sampleX: number[][], sampleY: number[], depth: number) {
        let _allowSplit = this.allowSplit(sampleX, sampleY, depth);
        if (!_allowSplit) return;
        const selection = this.attributeSelection(sampleX, sampleY);
        if (selection.minErrFeaIndex === -1) return;
        // if (Math.abs(nodeErr - selection.minErr) < 0.00001) return;
        const values = sampleX.map((x) => x[selection.minErrFeaIndex]);
        let leftSamples = filterWithIndices(values, (v) => v < selection.minErrValue);
        let rightSamples = filterWithIndices(values, (v) => v >= selection.minErrValue);
        tree.splitIndex = selection.minErrFeaIndex;
        tree.nodeValue = selection.minErrValue;

        tree.leftChild = this.initTreeNode(leftSamples.indices.map((i) => sampleY[i]));
        tree.rightChild = this.initTreeNode(rightSamples.indices.map((i) => sampleY[i]));
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
    private attributeSelection(sampleX: number[][], sampleY: number[]) {
        let minErr = Infinity;
        let minErrFeaIndex = -1; // bad case none
        let minErrValue = 0;
        // Q: what is best split
        for (let feaIndex = 0; feaIndex < this.feature_number; feaIndex++) {
            const values: number[] = sampleX.map((x) => x[feaIndex]);
            const valueSet: Set<number> = new Set(values);
            let localMinErr = Infinity;
            let localMinErrValue = 0;
            for (let feaValue of valueSet) {
                const leftChild = filterWithIndices(values, (x) => x < feaValue);
                const rightChild = filterWithIndices(values, (x) => x >= feaValue);
                const err =
                    (leftChild.indices.length / sampleX.length) *
                        this.impurity(leftChild.indices.map((i) => sampleY[i])) +
                    (rightChild.indices.length / sampleX.length) *
                        this.impurity(rightChild.indices.map((i) => sampleY[i]));
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
            minErrValue,
        };
    }
}