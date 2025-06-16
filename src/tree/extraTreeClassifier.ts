import { assert } from '../utils';
import { entropy, gini, mode } from '../utils/stat';
import { filterWithIndices, valuesAllSame, getUniqueFreqs } from './utils';
import type { IDTree } from './decisionTreeClassifier';

export interface ExtraTreeProps {
    max_depth?: number;
    min_samples_split?: number;
    criterion?: 'entropy' | 'gini';
    max_features?: number;
}

export class ExtraTreeClassifier {
    private dtree: IDTree | null = null;
    private max_depth: number;
    private feature_number = 0;
    private min_samples_split: number;
    private criterion: 'entropy' | 'gini';
    private impurity: (freqs: number[]) => number;
    private max_features_prop?: number;
    private max_features_: number;

    public constructor(props: ExtraTreeProps = {}) {
        const { max_depth = Infinity, min_samples_split = 2, criterion = 'entropy', max_features } = props;
        this.max_depth = max_depth;
        this.min_samples_split = min_samples_split;
        this.criterion = criterion;
        this.max_features_prop = max_features;
        this.impurity = criterion === 'entropy' ? entropy : gini;
        this.max_features_ = 0;
    }

    private nodeImpurity(sampleY: number[]): number {
        const freqs = getUniqueFreqs(sampleY);
        return this.impurity(freqs);
    }

    private selectFeatures(): number[] {
        const indices = Array.from({ length: this.feature_number }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
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
            const uniqueVals = Array.from(new Set(values));
            if (uniqueVals.length <= 1) continue;
            let lo: number;
            let hi: number;
            if (uniqueVals.length === 2) {
                const [v1, v2] = uniqueVals.sort((x, y) => x - y);
                lo = v1;
                hi = v2;
            } else {
                let a = uniqueVals[Math.floor(Math.random() * uniqueVals.length)];
                let b = uniqueVals[Math.floor(Math.random() * uniqueVals.length)];
                while (a === b) {
                    b = uniqueVals[Math.floor(Math.random() * uniqueVals.length)];
                }
                lo = Math.min(a, b);
                hi = Math.max(a, b);
            }
            const splitValue = Math.random() * (hi - lo) + lo;
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
        if (depth > this.max_depth) return;
        if (valuesAllSame(sampleY)) return;
        const split = this.attributeSelection(sampleX, sampleY);
        if (split.attIndex === -1) return;
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

    public fit(sampleX: number[][], sampleY: number[]) {
        assert(sampleX.length > 0, 'fit data should not be empty');
        this.feature_number = sampleX[0].length;
        const sqrtFeatures = Math.floor(Math.sqrt(this.feature_number));
        this.max_features_ = this.max_features_prop ? Math.min(this.feature_number, Math.floor(this.max_features_prop)) : sqrtFeatures;
        this.dtree = {
            nodeValue: 0,
            splitIndex: -1,
            y: mode(sampleY),
            leftChild: null,
            rightChild: null,
        };
        this.treeGenerate(this.dtree, sampleX, sampleY, 0);
    }

    private findSample(X: number[], tree: IDTree, depth: number): number {
        if (depth >= this.max_depth || tree.splitIndex === -1 || !tree.leftChild || !tree.rightChild) {
            return tree.y;
        }
        if (X[tree.splitIndex] < tree.nodeValue) {
            return this.findSample(X, tree.leftChild, depth + 1);
        } else {
            return this.findSample(X, tree.rightChild, depth + 1);
        }
    }

    public predict(sampleX: number[][]): number[] {
        return sampleX.map(x => this.findSample(x, this.dtree, 0));
    }
}
