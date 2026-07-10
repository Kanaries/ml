import { OutlierBase } from '../base/outlier';
import { Sampling, assert, createRandomGenerator } from '../utils';
import { EULER } from '../constants';
interface IITree {
    field: number;
    exNode: boolean;
    value: number;
    size: number;
    left?: IITree | null;
    right?: IITree | null;
}
export class IsolationForest extends OutlierBase {
    private max_depth: number;
    private iforest: IITree[];
    private subsampling_size: number;
    private tree_num: number;
    private contamination: 'auto' | number;
    private random_state?: number;
    private rng: () => number;
    // effective subsample size (psi) from the last fit; the user-supplied
    // subsampling_size is never overwritten
    private fitted_psi: number;
    // score threshold fixed on the training data at fit time when
    // contamination is numeric (sklearn offset_ semantics)
    private offset: number;
    public constructor (subsampling_size: number = 256, tree_num: number = 100, contamination: 'auto' | number = 'auto', random_state?: number) {
        super();
        this.iforest = [];
        this.subsampling_size = subsampling_size;
        this.tree_num = tree_num;
        this.contamination = contamination;
        this.random_state = random_state;
        this.rng = Math.random;
        this.fitted_psi = 0;
        this.offset = 0.5;
    }
    private iTree (samples: number[][], depth: number): IITree {
        if (samples.length <= 1 || depth >= this.max_depth) {
            return {
                field: 0,
                exNode: true,
                value: 0,
                size: samples.length,
                left: null,
                right: null
            }
        }
        let fieldIndex: number = Math.floor(this.rng() * samples[0].length);
        let max = -Infinity;
        let min = Infinity;
        for (let i = 0; i < samples.length; i++) {
            max = Math.max(max, samples[i][fieldIndex]);
            min = Math.min(min, samples[i][fieldIndex]);
        }
        const splitValue = (max - min) * this.rng() + min;
        let itree: IITree = {
            field: fieldIndex,
            exNode: false,
            value: splitValue,
            size: samples.length,
            left: null,
            right: null
        }
        itree.left = this.iTree(samples.filter(s => s[fieldIndex] < splitValue), depth + 1);
        itree.right = this.iTree(samples.filter(s => s[fieldIndex] >= splitValue), depth + 1);
        return itree;
    }

    private pathLength(x: number[], itree: IITree, depth: number): number {
        if (itree.exNode || depth >= this.max_depth) {
            return IsolationForest.AFS(itree.size);
        }
        const fieldIndex = itree.field;
        if (x[fieldIndex] < itree.value) return this.pathLength(x, itree.left, depth + 1) + 1;
        return this.pathLength(x, itree.right, depth + 1) + 1;
    }


    private iForest (samples: number[][], psi: number): IITree[] {
        assert(samples.length > 0, 'isolation forest requires non-empty samples');
        for (let i = 0; i < this.tree_num; i++){
            const subsamples = Sampling.std(samples, psi, this.rng);
            const itree = this.iTree(subsamples, 0);
            this.iforest.push(itree);
        }
        return this.iforest;
    }

    /**
     * average unsuccessful searches in BST (Preiss, 1999)
     * @param Psi
    */
    public static AFS(Psi: number): number {
        if (Psi > 2) return 2 * (Math.log(Psi - 1) + EULER) - 2 * (Psi - 1) / Psi;
        if (Psi === 2) return 1;
        return 0;
    }

    /**
     * linearly interpolated quantile, q in [0, 1]
     */
    private static quantile(values: number[], q: number): number {
        const sorted = values.slice().sort((a, b) => a - b);
        const pos = (sorted.length - 1) * Math.min(Math.max(q, 0), 1);
        const lo = Math.floor(pos);
        const hi = Math.ceil(pos);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    }

    public anomalyScore(x: number[]) {
        assert(this.iforest.length > 0, 'isolation forest must be fitted before scoring');
        const { iforest, fitted_psi } = this;
        let avgPathLength = 0;
        for (let  i = 0; i < iforest.length; i++) {
            avgPathLength += this.pathLength(x, iforest[i], 0);
        }
        avgPathLength /= iforest.length;
        const avgNormalization = IsolationForest.AFS(fitted_psi);
        return Math.pow(2, -avgPathLength / avgNormalization);
    }

    public fit (samplesX: number[][]) {
        assert(samplesX.length > 0, 'isolation forest requires non-empty samples');
        this.iforest = [];
        this.rng = createRandomGenerator(this.random_state);
        const psi = Math.min(this.subsampling_size, samplesX.length);
        this.fitted_psi = psi;
        this.max_depth = Math.ceil(Math.log2(Math.max(psi, 2)));
        this.iForest(samplesX, psi);
        if (this.contamination !== 'auto') {
            // fix the decision threshold on the training scores so that a
            // contamination fraction of the *training* data is flagged;
            // predict then applies this fixed offset (sklearn semantics)
            const trainScores = samplesX.map(x => this.anomalyScore(x));
            this.offset = IsolationForest.quantile(trainScores, 1 - this.contamination);
        }
    }

    public predict (samples: number[][]) {
        assert(this.iforest.length > 0, 'isolation forest must be fitted before predict');
        const result: number[] = [];
        for (let i = 0; i < samples.length; i++) {
            result.push(this.anomalyScore(samples[i]))
        }
        if (this.contamination === 'auto') {
            return result.map(r => r > 0.5 ? 1 : 0)
        }
        return result.map(r => r > this.offset ? 1 : 0);
    }
}
