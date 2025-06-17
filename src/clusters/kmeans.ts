import { ClusterBase } from '../base/cluster';

export class KMeans extends ClusterBase {
    private n_clusters: number;
    private centers: number[][] | null;
    private samplesX: number[][];
    private samplesY: number[];
    private sampleWeights: number[];
    private opt_ratio: number;
    private objective: number;
    private max_iter: number;
    private iter: number;
    constructor(n_clusters: number = 2, opt_ratio: number = 0.05, initCenters?: number[][], max_iter: number = 30, random_state?: number) {
        super();
        this.n_clusters = n_clusters;
        this.centers = null;
        this.samplesX = [];
        this.samplesY = [];
        this.max_iter = max_iter;
        this.iter = 0;
        this.opt_ratio = opt_ratio;
        if (initCenters) {
            this.centers = initCenters;
        }
    }
    private assignment(): void {
        const { centers, samplesX } = this;
        const samplesY = samplesX.map(() => 0);
        let objective: number = 0;
        for (let i = 0; i < samplesX.length; i++) {
            let nearestIndex = 0;
            let nearestDis = Infinity;
            for (let j = 0; j < centers.length; j++) {
                const dis_square = centers[j].reduce((sum, value, index) => sum + (value - samplesX[i][index]) ** 2, 0);
                if (dis_square < nearestDis) {
                    nearestDis = dis_square;
                    nearestIndex = j;
                }
            }
            samplesY[i] = nearestIndex;
            objective += nearestDis;
        }
        this.samplesY = samplesY;
        this.iter++;

        const shouldContinue = this.iter < this.max_iter && 
            (this.objective === Infinity || Math.abs(objective - this.objective) / this.objective > this.opt_ratio);
        
        this.objective = objective;
        
        if (shouldContinue) {
            this.updateCentroids();
            this.assignment();
        }
    }
    private updateCentroids(): void {
        const { n_clusters, samplesX, samplesY, sampleWeights } = this;
        const centers: number[][] = []; // = new Array(n_clusters).fill(0);
        const featureSize = samplesX[0].length;
        for (let i = 0; i < n_clusters; i++) {
            centers.push(new Array(featureSize).fill(0));
        }
        let neighborCounter: number[] = new Array(n_clusters).fill(0);
        for (let i = 0; i < samplesX.length; i++) {
            let centerIndex = samplesY[i];
            // centers[centerIndex]
            neighborCounter[centerIndex] += sampleWeights[i];
            for (let j = 0; j < centers[centerIndex].length; j++) {
                centers[centerIndex][j] += samplesX[i][j] * sampleWeights[i];
            }
        }
        for (let i = 0; i < centers.length; i++) {
            for (let j = 0; j < centers[i].length; j++) {
                centers[i][j] /= neighborCounter[i];
            }
        }
        this.centers = centers;
    }
    private initCentroids(): void {
        this.centers = [];
        const usedIndices: Set<number> = new Set();
        
        // For better determinism, use first data point as first center
        this.centers.push([...this.samplesX[0]]);
        usedIndices.add(0);
        
        // Choose remaining centers using k-means++ method
        for (let k = 1; k < this.n_clusters; k++) {
            const distances: number[] = [];
            let totalDistance = 0;
            
            // Calculate distance to nearest center for each point
            for (let i = 0; i < this.samplesX.length; i++) {
                if (usedIndices.has(i)) {
                    distances[i] = 0;
                    continue;
                }
                
                let minDist = Infinity;
                for (const center of this.centers) {
                    const dist = center.reduce((sum, val, idx) => 
                        sum + (val - this.samplesX[i][idx]) ** 2, 0);
                    minDist = Math.min(minDist, dist);
                }
                distances[i] = minDist;
                totalDistance += minDist;
            }
            
            // For determinism, choose the point with maximum distance to nearest center
            let maxDist = -1;
            let maxIndex = -1;
            for (let i = 0; i < this.samplesX.length; i++) {
                if (usedIndices.has(i)) continue;
                if (distances[i] > maxDist) {
                    maxDist = distances[i];
                    maxIndex = i;
                }
            }
            
            if (maxIndex !== -1) {
                this.centers.push([...this.samplesX[maxIndex]]);
                usedIndices.add(maxIndex);
            }
        }
    }
    public fitPredict(sampleX: number[][], sampleWeights?: number[]) {
        if (sampleWeights) {
            this.sampleWeights = sampleWeights;
        } else {
            this.sampleWeights = sampleX.map(() => 1);
        }
        this.samplesX = sampleX;
        this.objective = Infinity;
        this.iter = 0;
        if (this.centers === null) {
            this.initCentroids();
        }
        // this.initCentroids();
        this.assignment();
        return this.samplesY;
    }
    public getCentroids () {
        return this.centers;
    }
}