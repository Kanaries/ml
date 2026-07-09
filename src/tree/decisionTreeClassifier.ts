/**
 * 1. 为什么不做离散分割，因为使用one-hot encoding之后，按连续变量处理，可以获得不一定按照维度成员分割的效果，
 * 某一个维度可能个别成员作为单独的分割，其他则作为一个整体，这是直接写离散型分割比较难实现的。
 * 2. 多分割本质上可以被二分替代，所以没必要去做更复杂的多分割，目前也没有依据二者表现会有显著的差别。这样我们就可以直接使用二叉树来做。
 */
import { assert, createRandomGenerator } from "../utils";
import { entropy, gini, mode } from "../utils/stat";
import { getUniqueFreqs, valuesAllSame } from "./utils";
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
        this.random = createRandomGenerator(this.randomState);
        if (criterion === 'entropy') {
            this.impurity = entropy;
        } else {
            this.impurity = gini;
        }
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
        if (depth >= this.max_depth) return;
        if (sampleX.length < this.min_samples_split) return;
        const vsame = valuesAllSame(sampleY);
        if (vsame) return;

        const split = this.attributeSelection(sampleX, sampleY);
        // no feature separates the samples (e.g. duplicate rows with different labels)
        if (split.attIndex === -1) return;
        const leftX: number[][] = [];
        const leftY: number[] = [];
        const rightX: number[][] = [];
        const rightY: number[] = [];
        for (let i = 0; i < sampleX.length; i++) {
            if (sampleX[i][split.attIndex] <= split.splitValue) {
                leftX.push(sampleX[i]);
                leftY.push(sampleY[i]);
            } else {
                rightX.push(sampleX[i]);
                rightY.push(sampleY[i]);
            }
        }
        tree.splitIndex = split.attIndex;
        tree.nodeValue = split.splitValue;

        tree.leftChild = {
            splitIndex: -1,
            nodeValue: 0,
            leftChild: null,
            rightChild: null,
            y: mode(leftY),
        };
        tree.rightChild = {
            splitIndex: -1,
            nodeValue: 0,
            leftChild: null,
            rightChild: null,
            y: mode(rightY),
        };
        this.treeGenerate(tree.leftChild, leftX, leftY, depth + 1);
        this.treeGenerate(tree.rightChild, rightX, rightY, depth + 1);
    }
    /**
     * Maximizes impurity gain over midpoint thresholds, scanning each feature
     * once in sorted order with incremental class counts.
     */
    private attributeSelection(sampleX: number[][], sampleY: number[]) {
        const n = sampleY.length;
        const imp = this.nodeImpurity(sampleY);
        let maxGain = -Infinity;
        let bestAttIndex = -1;
        let bestSplitValue = 0;
        const featureIndices = this.selectedFeatureIndices();
        for (const attIndex of featureIndices) {
            const order = Array.from({ length: n }, (_, i) => i).sort(
                (a, b) => sampleX[a][attIndex] - sampleX[b][attIndex]
            );
            const leftCounts: Map<number, number> = new Map();
            const rightCounts: Map<number, number> = new Map();
            for (let i = 0; i < n; i++) {
                rightCounts.set(sampleY[i], (rightCounts.get(sampleY[i]) || 0) + 1);
            }
            for (let pos = 0; pos < n - 1; pos++) {
                const y = sampleY[order[pos]];
                leftCounts.set(y, (leftCounts.get(y) || 0) + 1);
                const rc = rightCounts.get(y) - 1;
                if (rc === 0) {
                    rightCounts.delete(y);
                } else {
                    rightCounts.set(y, rc);
                }
                const vCur = sampleX[order[pos]][attIndex];
                const vNext = sampleX[order[pos + 1]][attIndex];
                if (vCur === vNext) continue;
                const nLeft = pos + 1;
                const nRight = n - nLeft;
                const totalImp =
                    (nLeft / n) * this.impurity([...leftCounts.values()]) +
                    (nRight / n) * this.impurity([...rightCounts.values()]);
                const gain = imp - totalImp;
                if (gain > maxGain) {
                    maxGain = gain;
                    bestAttIndex = attIndex;
                    // a/2 + b/2 avoids overflow; the guard handles rounding up
                    // to b, which would empty the right child
                    const mid = vCur / 2 + vNext / 2;
                    bestSplitValue = mid === vNext ? vCur : mid;
                }
            }
        }
        return {
            gain: maxGain,
            attIndex: bestAttIndex,
            splitValue: bestSplitValue,
        };
    }

    private nodeImpurity (sampleY: number[]): number {
        const freqs = getUniqueFreqs(sampleY);
        return this.impurity(freqs);
    }

    public fit(sampleX: number[][], sampleY: number[]) {
        assert(sampleX.length > 0, 'fit data should not be empty');
        this.random = createRandomGenerator(this.randomState);
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
        return sampleX.map((x) => this.findSample(x, this.dtree));
    }
    private findSample(X: number[], tree: IDTree): number {
        if (tree.splitIndex === -1) {
            return tree.y;
        }
        if (X[tree.splitIndex] <= tree.nodeValue) {
            return this.findSample(X, tree.leftChild);
        } else {
            return this.findSample(X, tree.rightChild);
        }
    }
}
