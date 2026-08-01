import { ClassifierBase } from '../base';
import { registerEstimator, Params } from '../base/estimator';
import { assert, createRandomGenerator } from '../utils';
import { resolveSubsetSize, SubsetSizeOption } from '../utils/paramResolvers';
import { entropy, gini, mode } from '../utils/stat';
import { defineHiddenField, filterWithIndices, valuesAllSame, getUniqueFreqs } from './utils';
import type { IDTree } from './decisionTreeClassifier';
import { normalizedTreeImportances, treeImportances } from './featureImportances';

export interface ExtraTreeProps {
    max_depth?: number;
    min_samples_split?: number;
    criterion?: 'entropy' | 'gini';
    max_features?: SubsetSizeOption;
    randomState?: number;
}

export class ExtraTreeClassifier extends ClassifierBase {
    private dtree: IDTree | null = null;
    private max_depth: number;
    private feature_number = 0;
    private min_samples_split: number;
    private criterion: 'entropy' | 'gini';
    private max_features_prop?: SubsetSizeOption;
    private max_features_: number;
    private randomState?: number;
    private random: () => number;
    private classesState: number[] = [];

    public constructor(props: ExtraTreeProps = {}) {
        super();
        const {
            max_depth = Infinity,
            min_samples_split = 2,
            criterion = 'entropy',
            max_features,
            randomState = 0,
        } = props;
        this.max_depth = max_depth;
        this.min_samples_split = min_samples_split;
        this.criterion = criterion;
        this.max_features_prop = max_features;
        this.max_features_ = 0;
        this.randomState = randomState;
        // hidden (non-enumerable) so serialization only sees plain data
        defineHiddenField(this, 'random', createRandomGenerator(this.randomState));
    }

    public getParams(): Params {
        return {
            max_depth: this.max_depth,
            min_samples_split: this.min_samples_split,
            criterion: this.criterion,
            max_features: this.max_features_prop,
            randomState: this.randomState,
        };
    }

    /** Impurity of a class-frequency vector, resolved from `criterion`. */
    private impurity(freqs: number[]): number {
        return this.criterion === 'gini' ? gini(freqs) : entropy(freqs);
    }

    private nodeImpurity(sampleY: number[]): number {
        const freqs = getUniqueFreqs(sampleY);
        return this.impurity(freqs);
    }

    private selectFeatures(): number[] {
        const indices = Array.from({ length: this.feature_number }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(this.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        return indices.slice(0, this.max_features_);
    }

    private attributeSelection(sampleX: number[][], sampleY: number[]) {
        const imp = this.nodeImpurity(sampleY);
        const ans: { gain: number; left: { X: number[][]; Y: number[] }; right: { X: number[][]; Y: number[] }; attIndex: number; splitValue: number } = {
            gain: 0,
            left: { X: [], Y: [] },
            right: { X: [], Y: [] },
            attIndex: -1,
            splitValue: 0,
        };
        let maxGain = -Infinity;
        let maxGainAttIndex = -1;
        const featureIndices = this.selectFeatures();
        for (const i of featureIndices) {
            const values = sampleX.map(r => r[i]);
            let lo = Infinity;
            let hi = -Infinity;
            for (const value of values) { lo = Math.min(lo, value); hi = Math.max(hi, value); }
            if (lo === hi) continue;
            const splitValue = this.random() * (hi - lo) + lo;
            const left = filterWithIndices(values, v => v < splitValue);
            const right = filterWithIndices(values, v => v >= splitValue);
            if (left.subArr.length === 0 || right.subArr.length === 0) {
                continue;
            }
            const leftImp = this.nodeImpurity(left.indices.map(index => sampleY[index]));
            const rightImp = this.nodeImpurity(right.indices.map(index => sampleY[index]));
            const totalImp = (left.subArr.length / sampleX.length) * leftImp + (right.subArr.length / sampleX.length) * rightImp;
            const gain = imp - totalImp;
            if (gain > maxGain) {
                maxGain = gain;
                maxGainAttIndex = i;
                ans.left = { X: left.indices.map(index => sampleX[index]), Y: left.indices.map(index => sampleY[index]) };
                ans.right = { X: right.indices.map(index => sampleX[index]), Y: right.indices.map(index => sampleY[index]) };
                ans.splitValue = splitValue;
            }
        }
        ans.gain = maxGain;
        ans.attIndex = maxGainAttIndex;
        return ans;
    }

    private treeGenerate(tree: IDTree, sampleX: number[][], sampleY: number[], depth: number) {
        if (sampleX.length < this.min_samples_split) return;
        if (depth >= this.max_depth) return;
        if (valuesAllSame(sampleY)) return;
        const split = this.attributeSelection(sampleX, sampleY);
        if (split.attIndex === -1) return;
        tree.splitIndex = split.attIndex;
        tree.nodeValue = split.splitValue;
        tree.weightedImpurityDecrease = Math.max(0, split.gain) * sampleX.length;
        tree.leftChild = this.initTreeNode(split.left.Y);
        tree.rightChild = this.initTreeNode(split.right.Y);
        this.treeGenerate(tree.leftChild, split.left.X, split.left.Y, depth + 1);
        this.treeGenerate(tree.rightChild, split.right.X, split.right.Y, depth + 1);
    }

    public fit(sampleX: number[][], sampleY: number[]) {
        assert(sampleX.length > 0, 'fit data should not be empty');
        assert(sampleX.length === sampleY.length, 'X and y must have the same length');
        defineHiddenField(this, 'random', createRandomGenerator(this.randomState));
        this.feature_number = sampleX[0].length;
        this.classesState = Array.from(new Set(sampleY)).sort((a, b) => a - b);
        // sklearn's ExtraTreeClassifier defaults max_features to 'sqrt'
        this.max_features_ = resolveSubsetSize(this.max_features_prop ?? 'sqrt', this.feature_number);
        this.dtree = this.initTreeNode(sampleY);
        this.treeGenerate(this.dtree, sampleX, sampleY, 0);
    }

    private initTreeNode(sampleY: number[]): IDTree {
        const counts = new Map<number, number>();
        for (const label of sampleY) counts.set(label, (counts.get(label) ?? 0) + 1);
        return {
            nodeValue: 0,
            splitIndex: -1,
            y: mode(sampleY),
            leftChild: null,
            rightChild: null,
            classProbabilities: this.classesState.map(label => (counts.get(label) ?? 0) / sampleY.length),
        };
    }

    private findLeaf(X: number[], tree: IDTree): IDTree {
        if (tree.splitIndex === -1 || !tree.leftChild || !tree.rightChild) {
            return tree;
        }
        if (X[tree.splitIndex] < tree.nodeValue) {
            return this.findLeaf(X, tree.leftChild);
        } else {
            return this.findLeaf(X, tree.rightChild);
        }
    }

    private resolvedClasses(): number[] {
        if (this.classesState?.length) return this.classesState;
        if (!this.dtree) return [];
        const labels = new Set<number>();
        const visit = (node: IDTree): void => {
            if (node.splitIndex === -1 || !node.leftChild || !node.rightChild) labels.add(node.y);
            else { visit(node.leftChild); visit(node.rightChild); }
        };
        visit(this.dtree);
        return Array.from(labels).sort((a, b) => a - b);
    }

    public predict(sampleX: number[][]): number[] {
        if (!this.dtree) throw new Error('ExtraTreeClassifier is not fitted');
        const classes = this.resolvedClasses();
        return sampleX.map(x => {
            const leaf = this.findLeaf(x, this.dtree!);
            if (!leaf.classProbabilities || leaf.classProbabilities.length !== classes.length) return leaf.y;
            return classes[leaf.classProbabilities.indexOf(Math.max(...leaf.classProbabilities))];
        });
    }
    public predictProba(sampleX: number[][]): number[][] {
        if (!this.dtree) throw new Error('ExtraTreeClassifier is not fitted');
        const classes = this.resolvedClasses();
        return sampleX.map(x => {
            const leaf = this.findLeaf(x, this.dtree!);
            if (leaf.classProbabilities?.length === classes.length) return leaf.classProbabilities.slice();
            // Pre-Wave-A serialized trees stored only the majority leaf label.
            // Preserve their prediction path and expose the only honest
            // recoverable distribution: a one-hot vector for that label.
            return classes.map(label => label === leaf.y ? 1 : 0);
        });
    }
    public get classes(): number[] { return this.resolvedClasses().slice(); }
    public get featureImportances(): number[] {
        if (!this.dtree) throw new Error('ExtraTreeClassifier must be fitted before featureImportances');
        return normalizedTreeImportances(this.dtree, this.feature_number);
    }
    public getRawFeatureImportances(): number[] {
        if (!this.dtree) throw new Error('ExtraTreeClassifier must be fitted before featureImportances');
        return treeImportances(this.dtree, this.feature_number, false);
    }
}
registerEstimator('ExtraTreeClassifier', ExtraTreeClassifier);
