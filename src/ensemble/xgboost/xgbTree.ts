/**
 * Regression tree over first/second-order gradients, as in the XGBoost
 * paper's exact greedy algorithm (Chen & Guestrin 2016, Alg. 1).
 *
 * Split gain:
 *   1/2 * (G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)) - gamma
 * Leaf weight:
 *   w = -G / (H + lambda)
 */
export interface XGBTreeParams {
    maxDepth: number;
    lambda: number;
    gamma: number;
    minChildWeight: number;
}

interface XGBNode {
    splitIndex: number;
    threshold: number;
    weight: number;
    leftChild: XGBNode | null;
    rightChild: XGBNode | null;
}

export class XGBTree {
    private params: XGBTreeParams;
    private root: XGBNode | null = null;

    constructor(params: XGBTreeParams) {
        this.params = params;
    }

    /**
     * X, g, h are aligned by row; featureIndices restricts the columns
     * considered (column subsampling is decided by the caller per tree).
     */
    public fit(X: number[][], g: number[], h: number[], featureIndices: number[]): void {
        const rows = Array.from({ length: X.length }, (_, i) => i);
        this.root = this.build(X, g, h, featureIndices, rows, 0);
    }

    private leafWeight(G: number, H: number): number {
        const denominator = H + this.params.lambda;
        // h can be exactly 0 (saturated sigmoid) and lambda may be 0:
        // a zero update beats an Infinity/NaN margin
        if (denominator <= 0) return 0;
        return -G / denominator;
    }

    private build(
        X: number[][],
        g: number[],
        h: number[],
        featureIndices: number[],
        rows: number[],
        depth: number
    ): XGBNode {
        let G = 0;
        let H = 0;
        for (const i of rows) {
            G += g[i];
            H += h[i];
        }
        const node: XGBNode = {
            splitIndex: -1,
            threshold: 0,
            weight: this.leafWeight(G, H),
            leftChild: null,
            rightChild: null,
        };
        if (depth >= this.params.maxDepth || rows.length < 2) {
            return node;
        }

        const parentScore = (G * G) / (H + this.params.lambda);
        let bestGain = 0;
        let bestFeature = -1;
        let bestThreshold = 0;
        for (const f of featureIndices) {
            const order = [...rows].sort((a, b) => X[a][f] - X[b][f]);
            let GL = 0;
            let HL = 0;
            for (let pos = 0; pos < order.length - 1; pos++) {
                GL += g[order[pos]];
                HL += h[order[pos]];
                const vCur = X[order[pos]][f];
                const vNext = X[order[pos + 1]][f];
                if (vCur === vNext) continue;
                const GR = G - GL;
                const HR = H - HL;
                if (HL < this.params.minChildWeight || HR < this.params.minChildWeight) continue;
                const gain =
                    0.5 *
                        ((GL * GL) / (HL + this.params.lambda) +
                            (GR * GR) / (HR + this.params.lambda) -
                            parentScore) -
                    this.params.gamma;
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = f;
                    // a/2 + b/2 avoids overflow; the guard handles rounding up
                    // to b, which would empty the right child
                    const mid = vCur / 2 + vNext / 2;
                    bestThreshold = mid === vNext ? vCur : mid;
                }
            }
        }
        if (bestFeature === -1) {
            return node;
        }

        const leftRows: number[] = [];
        const rightRows: number[] = [];
        for (const i of rows) {
            if (X[i][bestFeature] <= bestThreshold) {
                leftRows.push(i);
            } else {
                rightRows.push(i);
            }
        }
        node.splitIndex = bestFeature;
        node.threshold = bestThreshold;
        node.leftChild = this.build(X, g, h, featureIndices, leftRows, depth + 1);
        node.rightChild = this.build(X, g, h, featureIndices, rightRows, depth + 1);
        return node;
    }

    public predict(X: number[][]): number[] {
        if (this.root === null) {
            throw new Error('tree is not fitted');
        }
        return X.map(x => {
            let node = this.root;
            while (node.splitIndex !== -1) {
                node = x[node.splitIndex] <= node.threshold ? node.leftChild : node.rightChild;
            }
            return node.weight;
        });
    }
}
