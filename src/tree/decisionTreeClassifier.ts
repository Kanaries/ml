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
interface IDTree {
    splitIndex: number;
    // 离散的情况下是当前节点的所有取值集合，连续则是一个区间
    nodeValue: number;
    y: number;
    leftChild: IDTree | null;
    rightChild: IDTree | null;
} 
export class DecisionTreeClassifier {
    private dtree: IDTree | null;
    private maxDepth: number;
    private featureNumber: number;
    private criterion: 'entropy' | 'gini' = 'entropy';
    private impurity: (freqs: number[]) => number;
    public constructor(maxDepth?: number, criterion?: 'entropy' | 'gini') {
        this.dtree = null;
        this.criterion;
        this.maxDepth = maxDepth || Infinity;
        this.criterion = criterion || 'entropy';
        this.featureNumber = 0;
        if (this.criterion === 'entropy') {
            this.impurity = entropy;
        } else {
            this.impurity = gini
        }
    }
    /**
     *
     * @param sampleX
     * @param sampleY
     * @param attributes indices of attributes.
     */
    public treeGenerate(tree: IDTree, sampleX: number[][], sampleY: number[], depth: number) {
        const vsame = valuesAllSame(sampleY);
        if (vsame) return;
        if (depth > this.maxDepth) return;
        const split = this.attributeSelection(sampleX, sampleY);
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
        const { featureNumber } = this;
        let maxGain = 0;
        let maxGainAttIndex = 0;
        for (let i = 0; i < featureNumber; i++) {
            let totalImp = 0;
            let attIndex = i;
            const values = sampleX.map((r) => r[attIndex]);
            const min = Math.min(...values);
            const max = Math.max(...values);
            console.log({ values, min, max, attIndex });
            // 可拓展，目前只生产一个随机的分割值
            const splitValue = Math.random() * (max - min) + min;
            const left = filterWithIndices(values, (v) => v < splitValue);
            const right = filterWithIndices(values, (v) => v >= splitValue);
            const leftImp = this.nodeImpurity(left.indices.map((i) => sampleY[i]));
            const rightImp = this.nodeImpurity(right.indices.map((i) => sampleY[i]));
            totalImp += (left.subArr.length / sampleX.length) * leftImp;
            totalImp += (right.subArr.length / sampleX.length) * rightImp;
            const gain = imp - totalImp;
            if (gain > maxGain) {
                maxGain = gain;
                maxGainAttIndex = attIndex;
                ans.left = {
                    X: left.indices.map((i) => sampleX[i]),
                    Y: left.indices.map((i) => sampleY[i]),
                };
                ans.right = {
                    X: right.indices.map((i) => sampleX[i]),
                    Y: right.indices.map((i) => sampleY[i]),
                };
                ans.splitValue = splitValue;
            }
        }
        ans.gain = maxGain;
        ans.attIndex = maxGainAttIndex;
        return ans;
    }

    // private nodeEntropy(sampleY: number[]): number {
    //     const freqMap: Map<any, number> = new Map();
    //     for (let i = 0; i < sampleY.length; i++) {
    //         if (!freqMap.has(sampleY[i])) {
    //             freqMap.set(sampleY[i], 0);
    //         }
    //         freqMap.set(sampleY[i], freqMap.get(sampleY[i]) + 1);
    //     }
    //     let ent = 0;
    //     for (let freq of freqMap.values()) {
    //         const p = freq / sampleY.length;
    //         ent += p * Math.log2(p);
    //     }
    //     return -ent;
    // }

    private nodeImpurity (sampleY: number[]): number {
        const freqs = getUniqueFreqs(sampleY);
        return this.impurity(freqs);
    }

    public fit(sampleX: number[][], sampleY: number[]) {
        assert(sampleX.length > 0, 'fit data should not be empty');
        this.featureNumber = sampleX[0].length;
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
        if (depth >= this.maxDepth || tree.splitIndex === -1) {
            return tree.y;
        }
        if (X[tree.splitIndex] < tree.nodeValue) {
            return this.findSample(X, tree.leftChild, depth + 1);
        } else {
            return this.findSample(X, tree.rightChild, depth + 1);
        }
    }
}