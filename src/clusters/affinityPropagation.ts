import { ClusterBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { validateMatrix } from '../utils/numerics';
import { createRandomGenerator } from '../utils/random';
import { normalRandom } from '../utils/numerics';

export interface AffinityPropagationProps { damping?: number; maxIter?: number; convergenceIter?: number; preference?: number | null; randomState?: number; }

export class AffinityPropagation extends ClusterBase {
    private damping: number; private maxIter: number; private convergenceIter: number; private preference: number | null; private randomState?: number;
    private centersState: number[][] = []; private centerIndicesState: number[] = []; private labelsState: number[] = []; private nIterState = 0; private trainingState: number[][] = [];
    constructor(props: AffinityPropagationProps = {}) { super(); const { damping = .5, maxIter = 200, convergenceIter = 15, preference = null, randomState } = props; if (!Number.isFinite(damping) || damping < .5 || damping >= 1 || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isInteger(convergenceIter) || convergenceIter < 1 || preference !== null && !Number.isFinite(preference)) throw new Error('invalid AffinityPropagation parameters'); this.damping = damping; this.maxIter = maxIter; this.convergenceIter = convergenceIter; this.preference = preference; this.randomState = randomState; }
    public getParams(): Params { return { damping: this.damping, maxIter: this.maxIter, convergenceIter: this.convergenceIter, preference: this.preference, randomState: this.randomState }; }
    private similarity(a: number[], b: number[]): number { return -a.reduce((sum, value, j) => sum + (value - b[j]) ** 2, 0); }
    public fit(X: number[][]): void {
        validateMatrix(X, 2); this.trainingState = X.map(row => row.slice()); const n = X.length;
        const S = X.map(row => X.map(other => this.similarity(row, other))), similarities = S.flat().sort((a, b) => a - b);
        const preference = this.preference ?? (similarities[Math.floor((similarities.length - 1) / 2)] + similarities[Math.ceil((similarities.length - 1) / 2)]) / 2;
        for (let i = 0; i < n; i++) S[i][i] = preference;
        const random = createRandomGenerator(this.randomState);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) S[i][j] += (Number.EPSILON * S[i][j] + Number.MIN_VALUE * 100) * normalRandom(random);
        let R = Array.from({ length: n }, () => new Array(n).fill(0)), A = Array.from({ length: n }, () => new Array(n).fill(0));
        const history = Array.from({ length: this.convergenceIter }, () => new Array(n).fill(false));
        for (this.nIterState = 0; this.nIterState < this.maxIter; this.nIterState++) {
            const nextR = Array.from({ length: n }, () => new Array(n).fill(0));
            for (let i = 0; i < n; i++) {
                let largest = -Infinity, second = -Infinity, argmax = -1;
                for (let k = 0; k < n; k++) { const value = A[i][k] + S[i][k]; if (value > largest) { second = largest; largest = value; argmax = k; } else if (value > second) second = value; }
                for (let k = 0; k < n; k++) nextR[i][k] = this.damping * R[i][k] + (1 - this.damping) * (S[i][k] - (k === argmax ? second : largest));
            }
            R = nextR; const nextA = Array.from({ length: n }, () => new Array(n).fill(0));
            for (let k = 0; k < n; k++) {
                let positive = 0; for (let i = 0; i < n; i++) if (i !== k) positive += Math.max(0, R[i][k]);
                for (let i = 0; i < n; i++) { const raw = i === k ? positive : Math.min(0, R[k][k] + positive - Math.max(0, R[i][k])); nextA[i][k] = this.damping * A[i][k] + (1 - this.damping) * raw; }
            }
            A = nextA; const exemplars = Array.from({ length: n }, (_, k) => A[k][k] + R[k][k] > 0); history[this.nIterState % this.convergenceIter] = exemplars;
            if (this.nIterState >= this.convergenceIter && Array.from({ length: n }, (_, k) => history.every(row => row[k]) || history.every(row => !row[k])).every(Boolean) && exemplars.some(Boolean)) { this.nIterState++; break; }
        }
        this.nIterState = Math.min(this.nIterState, this.maxIter); let indices = Array.from({ length: n }, (_, k) => k).filter(k => A[k][k] + R[k][k] > 0);
        if (indices.length === 0) { this.centerIndicesState = []; this.centersState = []; this.labelsState = new Array(n).fill(-1); return; }
        let labels = X.map(row => { let best = 0, similarity = -Infinity; indices.forEach((index, k) => { const value = this.similarity(row, X[index]); if (value > similarity) { similarity = value; best = k; } }); return best; });
        indices = indices.map((index, k) => { const members = labels.map((label, i) => ({ label, i })).filter(entry => entry.label === k).map(entry => entry.i); let best = index, score = -Infinity; for (const candidate of members) { const total = members.reduce((sum, member) => sum + S[member][candidate], 0); if (total > score) { score = total; best = candidate; } } return best; });
        indices = Array.from(new Set(indices)).sort((a, b) => a - b); labels = X.map(row => { let best = 0, similarity = -Infinity; indices.forEach((index, k) => { const value = this.similarity(row, X[index]); if (value > similarity) { similarity = value; best = k; } }); return best; });
        indices.forEach((sample, k) => labels[sample] = k); this.centerIndicesState = indices; this.centersState = indices.map(i => X[i].slice()); this.labelsState = labels;
    }
    public predict(X: number[][]): number[] { if (this.centersState.length === 0) return X.map(() => -1); return X.map(row => { let best = 0, score = -Infinity; this.centersState.forEach((center, i) => { const value = this.similarity(row, center); if (value > score) { score = value; best = i; } }); return best; }); }
    public fitPredict(X: number[][]): number[] { this.fit(X); return this.labelsState.slice(); }
    public get clusterCenters(): number[][] { return this.centersState.map(row => row.slice()); }
    public get clusterCenterIndices(): number[] { return this.centerIndicesState.slice(); }
    public get nIter(): number { return this.nIterState; }
}
registerEstimator('AffinityPropagation', AffinityPropagation);
