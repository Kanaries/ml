import { RegressorBase } from "../base";
import { registerEstimator, Params } from "../base/estimator";
import { assert, createRandomGenerator } from "../utils";
import { resolveSubsetSize, SubsetSizeOption } from "../utils/paramResolvers";
import { mean } from "../utils/stat";
import { IDTree } from "./decisionTreeClassifier";
import { defineHiddenField, valuesAllSame } from "./utils";
interface IRegTree extends IDTree {
}
interface RegressionTreeProps {
    max_depth?: number;
    min_samples_split?: number;
    max_features?: SubsetSizeOption;
    randomState?: number;
}
export class DecisionTreeRegressor extends RegressorBase {
    private feature_number: number;
    private regTree: IRegTree;
    private min_sample_split: number;
    private max_depth: number;
    private max_features?: SubsetSizeOption;
    private randomState?: number;
    private random: () => number;
    public constructor (props: RegressionTreeProps = {}) {
        super();
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
        // hidden (non-enumerable) so serialization only sees plain data
        defineHiddenField(this, 'random', createRandomGenerator(this.randomState));
    }
    public getParams(): Params {
        return {
            max_depth: this.max_depth,
            min_samples_split: this.min_sample_split,
            max_features: this.max_features,
            randomState: this.randomState,
        };
    }
    private selectedFeatureIndices(): number[] {
        const size = resolveSubsetSize(this.max_features, this.feature_number);
        const indices = Array.from({ length: this.feature_number }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(this.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        return indices.slice(0, size);
    }
    /**
     * Minimizes total SSE = n_left * var_left + n_right * var_right over
     * midpoint thresholds, scanning each feature once in sorted order.
     */
    private attributeSelection (sampleX: number[][], sampleY: number[]) {
        const n = sampleY.length;
        let minErr = Infinity;
        let minErrFeaIndex = -1; // bad case none
        let minErrValue = 0;
        const featureIndices = this.selectedFeatureIndices();
        for (let feaIndex of featureIndices) {
            const order = Array.from({ length: n }, (_, i) => i).sort(
                (a, b) => sampleX[a][feaIndex] - sampleX[b][feaIndex]
            );
            let sumLeft = 0;
            let sumSqLeft = 0;
            let sumRight = 0;
            let sumSqRight = 0;
            for (let i = 0; i < n; i++) {
                sumRight += sampleY[i];
                sumSqRight += sampleY[i] * sampleY[i];
            }
            for (let pos = 0; pos < n - 1; pos++) {
                const y = sampleY[order[pos]];
                sumLeft += y;
                sumSqLeft += y * y;
                sumRight -= y;
                sumSqRight -= y * y;
                const vCur = sampleX[order[pos]][feaIndex];
                const vNext = sampleX[order[pos + 1]][feaIndex];
                if (vCur === vNext) continue;
                const nLeft = pos + 1;
                const nRight = n - nLeft;
                const err =
                    (sumSqLeft - (sumLeft * sumLeft) / nLeft) +
                    (sumSqRight - (sumRight * sumRight) / nRight);
                if (err < minErr) {
                    minErr = err;
                    minErrFeaIndex = feaIndex;
                    // a/2 + b/2 avoids overflow; the guard handles rounding up
                    // to b, which would empty the right child
                    const mid = vCur / 2 + vNext / 2;
                    minErrValue = mid === vNext ? vCur : mid;
                }
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
        if (depth >= this.max_depth) return;
        if (sampleX.length < this.min_sample_split) return;
        if (valuesAllSame(sampleY)) return;
        const selection = this.attributeSelection(sampleX, sampleY);
        if (selection.minErrFeaIndex === -1) return;
        const leftX: number[][] = [];
        const leftY: number[] = [];
        const rightX: number[][] = [];
        const rightY: number[] = [];
        for (let i = 0; i < sampleX.length; i++) {
            if (sampleX[i][selection.minErrFeaIndex] <= selection.minErrValue) {
                leftX.push(sampleX[i]);
                leftY.push(sampleY[i]);
            } else {
                rightX.push(sampleX[i]);
                rightY.push(sampleY[i]);
            }
        }
        tree.splitIndex = selection.minErrFeaIndex;
        tree.nodeValue = selection.minErrValue;

        tree.leftChild = this.initTreeNode(leftY);
        tree.rightChild = this.initTreeNode(rightY);
        this.buildTree(tree.leftChild, leftX, leftY, depth + 1);
        this.buildTree(tree.rightChild, rightX, rightY, depth + 1);
    }
    public fit (sampleX: number[][], sampleY: number[]): void {
        assert(sampleX.length > 0, 'fit data should not be empty');
        defineHiddenField(this, 'random', createRandomGenerator(this.randomState));
        this.feature_number = sampleX[0].length;
        this.regTree = this.initTreeNode(sampleY);
        this.buildTree(this.regTree, sampleX, sampleY, 0);
    }
    private findLeaf(X: number[], tree: IRegTree): IRegTree {
        if (tree.splitIndex === -1) return tree;
        if (X[tree.splitIndex] <= tree.nodeValue) {
            return this.findLeaf(X, tree.leftChild);
        } else {
            return this.findLeaf(X, tree.rightChild);
        }
    }
    public predict (sampleX: number[][]): number[] {
        return sampleX.map(x => this.findLeaf(x, this.regTree).y);
    }
    private collectLeaves(): IRegTree[] {
        const leaves: IRegTree[] = [];
        const walk = (t: IRegTree) => {
            if (t.splitIndex === -1) {
                leaves.push(t);
                return;
            }
            walk(t.leftChild);
            walk(t.rightChild);
        };
        if (this.regTree) walk(this.regTree);
        return leaves;
    }
    /**
     * Leaf index (in depth-first order) each sample falls into.
     */
    public apply (sampleX: number[][]): number[] {
        const leafIds: Map<IRegTree, number> = new Map();
        this.collectLeaves().forEach((leaf, id) => leafIds.set(leaf, id));
        return sampleX.map(x => leafIds.get(this.findLeaf(x, this.regTree)));
    }
    /**
     * Overwrite leaf predictions (ids as returned by apply). Gradient
     * boosting classifiers use this for the per-leaf Newton step.
     */
    public setLeafValues (values: Map<number, number>): void {
        const leaves = this.collectLeaves();
        for (const [leafId, value] of values) {
            if (leafId < 0 || leafId >= leaves.length) {
                throw new Error(`leaf id ${leafId} out of range`);
            }
            leaves[leafId].y = value;
        }
    }
}
registerEstimator('DecisionTreeRegressor', DecisionTreeRegressor);
