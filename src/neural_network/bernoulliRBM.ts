export interface BernoulliRBMOptions {
    nComponents?: number;
    learningRate?: number;
    batchSize?: number;
    nIter?: number;
    randomState?: number;
}

function logistic(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

function createRandomGenerator(seed?: number): () => number {
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

function sampleBernoulli(probs: number[], random: () => number): number[] {
    return probs.map(p => (random() < p ? 1 : 0));
}

export class BernoulliRBM {
    private nComponents: number;
    private learningRate: number;
    private batchSize: number;
    private nIter: number;
    private random: () => number;

    private components: number[][] = [];
    private interceptHidden: number[] = [];
    private interceptVisible: number[] = [];

    constructor(options: BernoulliRBMOptions = {}) {
        this.nComponents = options.nComponents ?? 256;
        this.learningRate = options.learningRate ?? 0.1;
        this.batchSize = options.batchSize ?? 10;
        this.nIter = options.nIter ?? 10;
        this.random = createRandomGenerator(options.randomState);
    }

    private initParams(nFeatures: number): void {
        this.components = [];
        for (let j = 0; j < this.nComponents; j++) {
            const row: number[] = [];
            for (let i = 0; i < nFeatures; i++) {
                row.push((this.random() - 0.5) * 0.1);
            }
            this.components.push(row);
        }
        this.interceptHidden = new Array(this.nComponents).fill(0);
        this.interceptVisible = new Array(nFeatures).fill(0);
    }

    private hiddenProb(v: number[]): number[] {
        const probs: number[] = new Array(this.nComponents).fill(0);
        for (let j = 0; j < this.nComponents; j++) {
            let sum = this.interceptHidden[j];
            for (let i = 0; i < v.length; i++) {
                sum += this.components[j][i] * v[i];
            }
            probs[j] = logistic(sum);
        }
        return probs;
    }

    private visibleProb(h: number[]): number[] {
        const nFeatures = this.components[0].length;
        const probs: number[] = new Array(nFeatures).fill(0);
        for (let i = 0; i < nFeatures; i++) {
            let sum = this.interceptVisible[i];
            for (let j = 0; j < this.nComponents; j++) {
                sum += this.components[j][i] * h[j];
            }
            probs[i] = logistic(sum);
        }
        return probs;
    }

    private cdStep(batch: number[][]): void {
        const nFeatures = this.components[0].length;
        const bSize = batch.length;
        const weightGrad: number[][] = new Array(this.nComponents)
            .fill(0)
            .map(() => new Array(nFeatures).fill(0));
        const visGrad: number[] = new Array(nFeatures).fill(0);
        const hidGrad: number[] = new Array(this.nComponents).fill(0);

        for (let s = 0; s < bSize; s++) {
            const v0 = batch[s];
            const h0Prob = this.hiddenProb(v0);
            const h0 = sampleBernoulli(h0Prob, this.random);
            const v1Prob = this.visibleProb(h0);
            const v1 = sampleBernoulli(v1Prob, this.random);
            const h1Prob = this.hiddenProb(v1);

            for (let j = 0; j < this.nComponents; j++) {
                hidGrad[j] += h0Prob[j] - h1Prob[j];
                for (let i = 0; i < nFeatures; i++) {
                    weightGrad[j][i] += v0[i] * h0Prob[j] - v1[i] * h1Prob[j];
                }
            }
            for (let i = 0; i < nFeatures; i++) {
                visGrad[i] += v0[i] - v1[i];
            }
        }

        for (let j = 0; j < this.nComponents; j++) {
            for (let i = 0; i < nFeatures; i++) {
                this.components[j][i] += (this.learningRate * weightGrad[j][i]) / bSize;
            }
            this.interceptHidden[j] += (this.learningRate * hidGrad[j]) / bSize;
        }
        for (let i = 0; i < nFeatures; i++) {
            this.interceptVisible[i] += (this.learningRate * visGrad[i]) / bSize;
        }
    }

    public fit(X: number[][]): void {
        const nFeatures = X[0].length;
        this.initParams(nFeatures);
        for (let iter = 0; iter < this.nIter; iter++) {
            for (let start = 0; start < X.length; start += this.batchSize) {
                const batch = X.slice(start, start + this.batchSize);
                this.cdStep(batch);
            }
        }
    }

    public partialFit(X: number[][]): void {
        if (this.components.length === 0) {
            this.initParams(X[0].length);
        }
        for (let iter = 0; iter < this.nIter; iter++) {
            for (let start = 0; start < X.length; start += this.batchSize) {
                const batch = X.slice(start, start + this.batchSize);
                this.cdStep(batch);
            }
        }
    }

    public transform(X: number[][]): number[][] {
        return X.map(v => this.hiddenProb(v));
    }

    public fitTransform(X: number[][]): number[][] {
        this.fit(X);
        return this.transform(X);
    }

    public gibbs(V: number[][]): number[][] {
        return V.map(v => {
            const h = sampleBernoulli(this.hiddenProb(v), this.random);
            const v1 = sampleBernoulli(this.visibleProb(h), this.random);
            return v1;
        });
    }
}
