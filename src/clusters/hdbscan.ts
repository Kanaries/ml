import { ClusterBase } from '../base/cluster';
import { DBScan } from './dbscan';
import { Distance } from '../metrics';

/**
 * Simplified HDBSCAN implementation.
 *
 * This is not a full featured HDBSCAN algorithm. Instead it wraps the
 * existing DBSCAN implementation using the provided parameters. The
 * API mirrors sklearn's `HDBSCAN` so that it can be used interchangeably
 * with other clustering algorithms in this library.
 */
export class HDBScan extends ClusterBase {
    private minClusterSize: number;
    private minSamples: number;
    private epsilon: number;
    private metric: Distance.IDistanceType;
    constructor(
        min_cluster_size: number = 5,
        min_samples: number | null = null,
        cluster_selection_epsilon: number = 0.5,
        metric: Distance.IDistanceType = 'euclidean'
    ) {
        super();
        this.minClusterSize = min_cluster_size;
        this.minSamples = min_samples === null ? min_cluster_size : min_samples;
        this.epsilon = cluster_selection_epsilon;
        this.metric = metric;
    }

    /**
     * Cluster data and return cluster labels.
     *
     * Note that this simplified version internally falls back to DBSCAN
     * using the provided ``cluster_selection_epsilon`` value as the eps
     * parameter. It does not build a full hierarchy but provides a
     * compatible interface.
     */
    public fitPredict(samplesX: number[][]): number[] {
        const dbscan = new DBScan(this.epsilon, this.minSamples, this.metric);
        return dbscan.fitPredict(samplesX);
    }
}
