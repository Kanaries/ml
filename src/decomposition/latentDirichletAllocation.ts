import { TransformerBase } from '../base';
import { Params, registerEstimator } from '../base/estimator';
import { CSRMatrix, NumericMatrix, matrixRow, matrixShape } from '../data';
import { createRandomGenerator } from '../utils/random';
import { normalRandom } from '../utils/numerics';

export interface LatentDirichletAllocationProps { nComponents?: number; docTopicPrior?: number | null; topicWordPrior?: number | null; maxIter?: number; maxDocUpdateIter?: number; meanChangeTol?: number; randomState?: number; }

function digamma(x: number): number { let result = 0; while (x < 7) { result -= 1 / x; x++; } x -= .5; const inv = 1 / x, inv2 = inv * inv; return result + Math.log(x) + inv2 * (1 / 24 - inv2 * (7 / 960 - inv2 * 31 / 8064)); }
function expDirichlet(row: number[]): number[] { const denominator = digamma(row.reduce((a, b) => a + b, 0)); return row.map(value => Math.exp(digamma(value) - denominator)); }

export class LatentDirichletAllocation extends TransformerBase<any, any> {
    private nComponents: number; private docTopicPrior: number | null; private topicWordPrior: number | null; private maxIter: number; private maxDocUpdateIter: number; private meanChangeTol: number; private randomState?: number;
    private componentsState: number[][] = []; private nIterState = 0; private nFeaturesState = 0;
    constructor(props: LatentDirichletAllocationProps = {}) { super(); const { nComponents = 10, docTopicPrior = null, topicWordPrior = null, maxIter = 10, maxDocUpdateIter = 100, meanChangeTol = 1e-3, randomState } = props; if (!Number.isInteger(nComponents) || nComponents < 1 || docTopicPrior !== null && (!Number.isFinite(docTopicPrior) || docTopicPrior <= 0) || topicWordPrior !== null && (!Number.isFinite(topicWordPrior) || topicWordPrior <= 0) || !Number.isInteger(maxIter) || maxIter < 1 || !Number.isInteger(maxDocUpdateIter) || maxDocUpdateIter < 1 || !Number.isFinite(meanChangeTol) || meanChangeTol <= 0) throw new Error('invalid LatentDirichletAllocation parameters'); this.nComponents = nComponents; this.docTopicPrior = docTopicPrior; this.topicWordPrior = topicWordPrior; this.maxIter = maxIter; this.maxDocUpdateIter = maxDocUpdateIter; this.meanChangeTol = meanChangeTol; this.randomState = randomState; }
    public getParams(): Params { return { nComponents: this.nComponents, docTopicPrior: this.docTopicPrior, topicWordPrior: this.topicWordPrior, maxIter: this.maxIter, maxDocUpdateIter: this.maxDocUpdateIter, meanChangeTol: this.meanChangeTol, randomState: this.randomState }; }
    private validate(X: NumericMatrix): [number, number] { const [n, p] = matrixShape(X); if (n === 0 || p === 0) throw new Error('LDA requires a non-empty count matrix'); for (let i = 0; i < n; i++) if (matrixRow(X, i).some(value => !Number.isFinite(value) || value < 0)) throw new Error('LDA counts must be finite and non-negative'); return [n, p]; }
    private eStep(X: NumericMatrix, collect: boolean): { gamma: number[][]; sufficient: number[][] } {
        const [n] = matrixShape(X), alpha = this.docTopicPrior ?? 1 / this.nComponents, expTopic = this.componentsState.map(expDirichlet), sufficient = Array.from({ length: this.nComponents }, () => new Array(this.nFeaturesState).fill(0)), gamma: number[][] = [];
        for (let document = 0; document < n; document++) {
            const row = matrixRow(X, document), words = row.map((value, index) => ({ value, index })).filter(entry => entry.value > 0), total = words.reduce((sum, entry) => sum + entry.value, 0);
            let current = new Array(this.nComponents).fill(alpha + total / this.nComponents);
            for (let iteration = 0; iteration < this.maxDocUpdateIter; iteration++) {
                const expDoc = expDirichlet(current), next = new Array(this.nComponents).fill(alpha);
                for (const word of words) { let normalizer = 0; for (let topic = 0; topic < this.nComponents; topic++) normalizer += expDoc[topic] * expTopic[topic][word.index]; normalizer = Math.max(normalizer, 1e-100); for (let topic = 0; topic < this.nComponents; topic++) next[topic] += word.value * expDoc[topic] * expTopic[topic][word.index] / normalizer; }
                const change = next.reduce((sum, value, topic) => sum + Math.abs(value - current[topic]), 0) / this.nComponents; current = next; if (change < this.meanChangeTol) break;
            }
            gamma.push(current);
            if (collect) { const expDoc = expDirichlet(current); for (const word of words) { let normalizer = 0; for (let topic = 0; topic < this.nComponents; topic++) normalizer += expDoc[topic] * expTopic[topic][word.index]; for (let topic = 0; topic < this.nComponents; topic++) sufficient[topic][word.index] += word.value * expDoc[topic] * expTopic[topic][word.index] / Math.max(normalizer, 1e-100); } }
        }
        return { gamma, sufficient };
    }
    public fit(X: NumericMatrix): void {
        [, this.nFeaturesState] = this.validate(X); const random = createRandomGenerator(this.randomState), eta = this.topicWordPrior ?? 1 / this.nComponents;
        this.componentsState = Array.from({ length: this.nComponents }, () => Array.from({ length: this.nFeaturesState }, () => Math.max(1e-3, 1 + .1 * normalRandom(random))));
        for (this.nIterState = 1; this.nIterState <= this.maxIter; this.nIterState++) { const { sufficient } = this.eStep(X, true); this.componentsState = sufficient.map(row => row.map(value => eta + value)); }
        this.nIterState = this.maxIter;
    }
    public transform(X: NumericMatrix): number[][] { const [, p] = this.validate(X); if (this.componentsState.length === 0) throw new Error('LDA is not fitted'); if (p !== this.nFeaturesState) throw new Error('feature count differs from fitted LDA'); return this.eStep(X, false).gamma.map(row => { const sum = row.reduce((a, b) => a + b, 0); return row.map(value => value / sum); }); }
    public fitTransform(X: NumericMatrix): number[][] { this.fit(X); return this.transform(X); }
    public get components(): number[][] { return this.componentsState.map(row => row.slice()); }
    public get nIter(): number { return this.nIterState; }
}
registerEstimator('LatentDirichletAllocation', LatentDirichletAllocation);
