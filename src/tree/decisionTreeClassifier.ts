/**
 * 1. 为什么不做离散分割，因为使用one-hot encoding之后，按连续变量处理，可以获得不一定按照维度成员分割的效果，
 * 某一个维度可能个别成员作为单独的分割，其他则作为一个整体，这是直接写离散型分割比较难实现的。
 * 2. 多分割本质上可以被二分替代，所以没必要去做更复杂的多分割，目前也没有依据二者表现会有显著的差别。这样我们就可以直接使用二叉树来做。
 */
import { assert } from "../utils";
import { entropy, gini, mode } from "../utils/stat";
import { filterWithIndices, getUniqueFreqs, valuesAllSame } from "./utils";
export type IFeatureSplitType = 'continuous' | 'discrete';
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
    max_features?: number | 'sqrt' | 'log2';
    randomState?: number;
}

export class DecisionTreeClassifier {
    private dtree: IDTree | null;
    private max_depth: number;
    private feature_number: number;
    private min_samples_split: number;
    private criterion: 'entropy' | 'gini' = 'entropy';
    private impurity: (freqs: number[]) => number;
    private max_features?: number | 'sqrt' | 'log2';
    private randomState?: number;
    private random: () => number;
    public constructor(props: DecisionTreeProps = {}) {
        const { max_depth = Infinity, criterion = 'entropy', min_samples_split = 2, max_features, randomState } = props;
        this.dtree = null;
        this.max_depth = max_depth;
        this.criterion = criterion;
        this.feature_number = 0;
        this.min_samples_split = min_samples_split;
        this.max_features = max_features;
        this.randomState = randomState;
        this.random = this.createRandomGenerator(this.randomState);
        if (criterion === 'entropy') {
            this.impurity = entropy;
        } else {
            this.impurity = gini;
        }
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
    /**
     *
     * @param sampleX
     * @param sampleY
     * @param attributes indices of attributes.
     */
    public treeGenerate(tree: IDTree, sampleX: number[][], sampleY: number[], depth: number) {
        if (sampleX.length < this.min_samples_split) return;
        if (depth > this.max_depth) return;
        const vsame = valuesAllSame(sampleY);
        if (vsame) return;
    
        const split = this.attributeSelection(sampleX, sampleY);
        // if (split.gain <= 0) return;
        tree.splitIndex = split.attIndex;
        tree.nodeValue = split.splitValue;

        tree.leftChild = {
            splitIndex: -1,
            nodeValue: 0,
            leftChild: null,
            rightChild: null,
            y: mode(split.left.Y),
        };
        tree.rightChild = {
            splitIndex: -1,
            nodeValue: 0,
            leftChild: null,
            rightChild: null,
            y: mode(split.right.Y),
        };
        this.treeGenerate(tree.leftChild, split.left.X, split.left.Y, depth + 1);
        this.treeGenerate(tree.rightChild, split.right.X, split.right.Y, depth + 1);
    }
    private attributeSelection(sampleX: number[][], sampleY: number[]) {
        const imp = this.nodeImpurity(sampleY);
        const ans: { gain: number; left: ISlice; right: ISlice; attIndex: number; splitValue: number } = {
            gain: 0,
            left: { X: [], Y: [] },
            right: { X: [], Y: [] },
            attIndex: -1,
            splitValue: 0,
        };
        const featureIndices = this.selectedFeatureIndices();
        let maxGain = -Infinity;
        let maxGainAttIndex = 0;
        for (const i of featureIndices) {
            let attIndex = i;
            const values = sampleX.map((r) => r[attIndex]);
            const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
            
            // Try all possible split points between unique values
            for (let j = 0; j < uniqueValues.length - 1; j++) {
                const splitValue = (uniqueValues[j] + uniqueValues[j + 1]) / 2;
                const left = filterWithIndices(values, (v) => v < splitValue);
                const right = filterWithIndices(values, (v) => v >= splitValue);
                
                if (left.indices.length === 0 || right.indices.length === 0) continue;
                
                const leftImp = this.nodeImpurity(left.indices.map((index) => sampleY[index]));
                const rightImp = this.nodeImpurity(right.indices.map((index) => sampleY[index]));
                const totalImp = (left.subArr.length / sampleX.length) * leftImp + 
                                (right.subArr.length / sampleX.length) * rightImp;
                const gain = imp - totalImp;
                
                if (gain > maxGain) {
                    maxGain = gain;
                    maxGainAttIndex = attIndex;
                    ans.left = {
                        X: left.indices.map((index) => sampleX[index]),
                        Y: left.indices.map((index) => sampleY[index]),
                    };
                    ans.right = {
                        X: right.indices.map((index) => sampleX[index]),
                        Y: right.indices.map((index) => sampleY[index]),
                    };
                    ans.splitValue = splitValue;
                }
            }
        }
        ans.gain = maxGain;
        ans.attIndex = maxGainAttIndex;
        return ans;
    }

    private nodeImpurity (sampleY: number[]): number {
        const freqs = getUniqueFreqs(sampleY);
        return this.impurity(freqs);
    }

    public fit(sampleX: number[][], sampleY: number[]) {
        assert(sampleX.length > 0, 'fit data should not be empty');
        this.random = this.createRandomGenerator(this.randomState);
        this.feature_number = sampleX[0].length;
        this.dtree = {
            nodeValue: 0,
            splitIndex: -1,
            y: mode(sampleY),
            leftChild: null,
            rightChild: null,
        };
        this.treeGenerate(this.dtree, sampleX, sampleY, 0);
    }
    public predict(sampleX: number[][]): number[] {
        return sampleX.map((x) => this.findSample(x, this.dtree, 0));
    }
    private findSample(X: number[], tree: IDTree, depth: number): number {
        if (depth >= this.max_depth || tree.splitIndex === -1) {
            return tree.y;
        }
        if (X[tree.splitIndex] < tree.nodeValue) {
            return this.findSample(X, tree.leftChild, depth + 1);
        } else {
            return this.findSample(X, tree.rightChild, depth + 1);
        }
    }
}
