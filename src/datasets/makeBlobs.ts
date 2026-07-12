import {
    createRandomGenerator,
    createGaussianGenerator,
    shuffleInUnison,
    uniform,
    resolveSamplesPerGroup,
    Dataset,
} from './common';

export interface MakeBlobsProps {
    /**
     * total number of samples (split as evenly as possible between the
     * clusters), or an array with the number of samples per cluster
     */
    nSamples?: number | number[];
    /** number of features per sample (ignored when explicit centers are given) */
    nFeatures?: number;
    /**
     * number of centers to generate uniformly inside `centerBox`, or an
     * explicit array of center coordinates of shape [nCenters][nFeatures]
     */
    centers?: number | number[][];
    /** standard deviation of the clusters; a single value or one per cluster */
    clusterStd?: number | number[];
    /** bounding box [min, max] for randomly generated centers */
    centerBox?: [number, number];
    /** shuffle the samples (default true) */
    shuffle?: boolean;
    /** seed for reproducible output */
    randomState?: number;
}

/**
 * Generates isotropic Gaussian blobs for clustering
 * (port of sklearn.datasets.make_blobs).
 */
export function makeBlobs(props: MakeBlobsProps = {}): Dataset {
    const {
        nSamples = 100,
        nFeatures = 2,
        centers,
        clusterStd = 1,
        centerBox = [-10, 10],
        shuffle = true,
        randomState,
    } = props;

    const rng = createRandomGenerator(randomState);
    const gaussian = createGaussianGenerator(rng);

    let centerCoords: number[][];
    let featureCount = nFeatures;
    if (Array.isArray(nSamples)) {
        // per-cluster sample counts: the number of clusters comes from nSamples
        const nCenters = nSamples.length;
        if (centers === undefined) {
            centerCoords = generateCenters(nCenters, featureCount, centerBox, rng);
        } else if (typeof centers === 'number') {
            throw new Error('when nSamples is an array, centers must be an array of coordinates (or omitted), not a count');
        } else {
            if (centers.length !== nCenters) {
                throw new Error(`centers length (${centers.length}) must equal nSamples length (${nCenters})`);
            }
            centerCoords = centers.map((c) => c.slice());
            featureCount = centerCoords[0].length;
        }
    } else if (centers === undefined || typeof centers === 'number') {
        const nCenters: number = typeof centers === 'number' ? centers : 3;
        if (nCenters < 1) {
            throw new Error('centers must be at least 1');
        }
        centerCoords = generateCenters(nCenters, featureCount, centerBox, rng);
    } else {
        centerCoords = centers.map((c) => c.slice());
        featureCount = centerCoords[0].length;
    }

    const nCenters = centerCoords.length;
    const stds: number[] = Array.isArray(clusterStd) ? clusterStd.slice() : new Array(nCenters).fill(clusterStd);
    if (stds.length !== nCenters) {
        throw new Error(`clusterStd length (${stds.length}) must equal the number of centers (${nCenters})`);
    }

    const samplesPerCenter = resolveSamplesPerGroup(nSamples, nCenters);

    let X: number[][] = [];
    let y: number[] = [];
    for (let k = 0; k < nCenters; k++) {
        for (let s = 0; s < samplesPerCenter[k]; s++) {
            const sample: number[] = new Array(featureCount);
            for (let d = 0; d < featureCount; d++) {
                sample[d] = centerCoords[k][d] + stds[k] * gaussian();
            }
            X.push(sample);
            y.push(k);
        }
    }

    if (shuffle) {
        ({ X, y } = shuffleInUnison(X, y, rng));
    }
    return { X, y };
}

function generateCenters(nCenters: number, nFeatures: number, centerBox: [number, number], rng: () => number): number[][] {
    const centers: number[][] = [];
    for (let k = 0; k < nCenters; k++) {
        const center: number[] = new Array(nFeatures);
        for (let d = 0; d < nFeatures; d++) {
            center[d] = uniform(rng, centerBox[0], centerBox[1]);
        }
        centers.push(center);
    }
    return centers;
}
