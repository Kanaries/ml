import {
    createRandomGenerator,
    createGaussianGenerator,
    permutation,
    randInt,
    shuffleInUnison,
    uniformMatrix,
    Dataset,
} from './common';

export interface MakeClassificationProps {
    /** total number of samples */
    nSamples?: number;
    /** total number of features */
    nFeatures?: number;
    /** number of informative features */
    nInformative?: number;
    /** number of redundant features (random linear combinations of the informative ones) */
    nRedundant?: number;
    /** number of duplicated features (copies of informative/redundant columns) */
    nRepeated?: number;
    /** number of classes */
    nClasses?: number;
    /** number of gaussian clusters per class */
    nClustersPerClass?: number;
    /**
     * class proportions; length nClasses, or nClasses - 1 (the last weight
     * is inferred as 1 - sum)
     */
    weights?: number[];
    /** fraction of samples whose class is assigned randomly (label noise) */
    flipY?: number;
    /** factor multiplying the hypercube size; larger values spread the classes apart */
    classSep?: number;
    /** put cluster centroids on the vertices of a hypercube (vs a random polytope) */
    hypercube?: boolean;
    /**
     * shift added to all features; a single value or one per feature.
     * Pass null to draw a random shift in [-classSep, classSep] (sklearn's
     * `shift=None` behaviour).
     */
    shift?: number | number[] | null;
    /**
     * factor multiplying all features; a single value or one per feature.
     * Pass null to draw a random scale in [1, 100] (sklearn's `scale=None`
     * behaviour). Scaling happens after shifting.
     */
    scale?: number | number[] | null;
    /** shuffle the samples and the feature columns (default true) */
    shuffle?: boolean;
    /** seed for reproducible output */
    randomState?: number;
}

/**
 * Generates a random n-class classification problem
 * (port of sklearn.datasets.make_classification).
 *
 * Informative features are drawn from gaussian clusters placed on the
 * vertices of an nInformative-dimensional hypercube with sides of length
 * 2 * classSep, each cluster given a random covariance. Redundant features
 * are random linear combinations of the informative ones, repeated
 * features are exact duplicates, and the remaining features are noise.
 */
export function makeClassification(props: MakeClassificationProps = {}): Dataset {
    const {
        nSamples = 100,
        nFeatures = 20,
        nInformative = 2,
        nRedundant = 2,
        nRepeated = 0,
        nClasses = 2,
        nClustersPerClass = 2,
        weights,
        flipY = 0.01,
        classSep = 1,
        hypercube = true,
        shift = 0,
        scale = 1,
        shuffle = true,
        randomState,
    } = props;

    if (nInformative + nRedundant + nRepeated > nFeatures) {
        throw new Error(
            `nFeatures (${nFeatures}) must be at least nInformative + nRedundant + nRepeated`
            + ` (${nInformative + nRedundant + nRepeated})`,
        );
    }
    if (nInformative < Math.log2(nClasses * nClustersPerClass)) {
        throw new Error(
            `nClasses * nClustersPerClass (${nClasses * nClustersPerClass}) must be`
            + ` smaller or equal 2 ** nInformative (${2 ** nInformative})`,
        );
    }

    let classWeights: number[];
    if (weights !== undefined) {
        if (weights.length === nClasses - 1) {
            const partialSum = weights.reduce((acc, w) => acc + w, 0);
            classWeights = [...weights, 1 - partialSum];
        } else if (weights.length === nClasses) {
            classWeights = weights.slice();
        } else {
            throw new Error(`weights must have length nClasses (${nClasses}) or nClasses - 1`);
        }
    } else {
        classWeights = new Array(nClasses).fill(1 / nClasses);
    }

    const rng = createRandomGenerator(randomState);
    const gaussian = createGaussianGenerator(rng);

    const nUseless = nFeatures - nInformative - nRedundant - nRepeated;
    const nClusters = nClasses * nClustersPerClass;

    // distribute samples over the clusters, honouring the class weights
    const samplesPerCluster: number[] = [];
    for (let k = 0; k < nClusters; k++) {
        samplesPerCluster.push(Math.floor((nSamples * classWeights[k % nClasses]) / nClustersPerClass));
    }
    let assigned = samplesPerCluster.reduce((acc, c) => acc + c, 0);
    for (let i = 0; assigned < nSamples; i++, assigned++) {
        samplesPerCluster[i % nClusters] += 1;
    }

    // cluster centroids on (a subset of) the hypercube vertices
    const centroids = generateHypercube(nClusters, nInformative, rng)
        .map((row) => row.map((v) => v * 2 * classSep - classSep));
    if (!hypercube) {
        for (let k = 0; k < nClusters; k++) {
            const rowFactor = rng();
            for (let j = 0; j < nInformative; j++) {
                centroids[k][j] *= rowFactor;
            }
        }
        for (let j = 0; j < nInformative; j++) {
            const colFactor = rng();
            for (let k = 0; k < nClusters; k++) {
                centroids[k][j] *= colFactor;
            }
        }
    }

    const X: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
        X.push(new Array(nFeatures).fill(0));
    }
    const y: number[] = new Array(nSamples).fill(0);

    // informative features: per-cluster gaussians with a random covariance
    let rowStart = 0;
    for (let k = 0; k < nClusters; k++) {
        const count = samplesPerCluster[k];
        const covariance = uniformMatrix(nInformative, nInformative, rng, -1, 1);
        for (let s = 0; s < count; s++) {
            const i = rowStart + s;
            const raw: number[] = new Array(nInformative);
            for (let j = 0; j < nInformative; j++) {
                raw[j] = gaussian();
            }
            for (let j = 0; j < nInformative; j++) {
                let value = centroids[k][j];
                for (let d = 0; d < nInformative; d++) {
                    value += raw[d] * covariance[d][j];
                }
                X[i][j] = value;
            }
            y[i] = k % nClasses;
        }
        rowStart += count;
    }

    // redundant features: random linear combinations of the informative ones
    if (nRedundant > 0) {
        const B = uniformMatrix(nInformative, nRedundant, rng, -1, 1);
        for (let i = 0; i < nSamples; i++) {
            for (let j = 0; j < nRedundant; j++) {
                let value = 0;
                for (let d = 0; d < nInformative; d++) {
                    value += X[i][d] * B[d][j];
                }
                X[i][nInformative + j] = value;
            }
        }
    }

    // repeated features: exact duplicates of informative/redundant columns
    if (nRepeated > 0) {
        const n = nInformative + nRedundant;
        for (let j = 0; j < nRepeated; j++) {
            const sourceIndex = Math.floor((n - 1) * rng() + 0.5);
            for (let i = 0; i < nSamples; i++) {
                X[i][n + j] = X[i][sourceIndex];
            }
        }
    }

    // remaining features: pure noise
    if (nUseless > 0) {
        const offset = nFeatures - nUseless;
        for (let i = 0; i < nSamples; i++) {
            for (let j = 0; j < nUseless; j++) {
                X[i][offset + j] = gaussian();
            }
        }
    }

    // randomly flip a fraction of the labels
    if (flipY >= 0) {
        for (let i = 0; i < nSamples; i++) {
            if (rng() < flipY) {
                y[i] = randInt(rng, nClasses);
            }
        }
    }

    // shift then scale
    const shiftPerFeature: number[] = resolvePerFeature(
        shift, nFeatures, () => (2 * rng() - 1) * classSep, 'shift',
    );
    const scalePerFeature: number[] = resolvePerFeature(
        scale, nFeatures, () => 1 + 100 * rng(), 'scale',
    );
    for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < nFeatures; j++) {
            X[i][j] = (X[i][j] + shiftPerFeature[j]) * scalePerFeature[j];
        }
    }

    let outX = X;
    let outY = y;
    if (shuffle) {
        ({ X: outX, y: outY } = shuffleInUnison(outX, outY, rng));
        const featureOrder = permutation(nFeatures, rng);
        outX = outX.map((row) => featureOrder.map((j) => row[j]));
    }
    return { X: outX, y: outY };
}

function resolvePerFeature(
    value: number | number[] | null,
    nFeatures: number,
    randomDraw: () => number,
    name: string,
): number[] {
    if (value === null) {
        return Array.from({ length: nFeatures }, () => randomDraw());
    }
    if (Array.isArray(value)) {
        if (value.length !== nFeatures) {
            throw new Error(`${name} array must have length nFeatures (${nFeatures})`);
        }
        return value.slice();
    }
    return new Array(nFeatures).fill(value);
}

/**
 * Returns `samples` distinct vertices of the `dimensions`-dimensional unit
 * hypercube as 0/1 rows. For dimensions > 30 (where enumerating vertices as
 * integers is unsafe), the trailing 30 coordinates are kept distinct and the
 * leading ones are random bits, mirroring sklearn's `_generate_hypercube`.
 */
function generateHypercube(samples: number, dimensions: number, rng: () => number): number[][] {
    if (dimensions > 30) {
        const tail = generateHypercube(samples, 30, rng);
        return tail.map((row) => {
            const head: number[] = new Array(dimensions - 30);
            for (let j = 0; j < dimensions - 30; j++) {
                head[j] = rng() < 0.5 ? 0 : 1;
            }
            return [...head, ...row];
        });
    }
    const total = 2 ** dimensions;
    let chosen: number[];
    if (total <= 2 * samples) {
        // dense case: partial Fisher-Yates over all vertices
        chosen = permutation(total, rng).slice(0, samples);
    } else {
        // sparse case: rejection sampling of distinct vertex ids
        const seen = new Set<number>();
        chosen = [];
        while (chosen.length < samples) {
            const candidate = randInt(rng, total);
            if (!seen.has(candidate)) {
                seen.add(candidate);
                chosen.push(candidate);
            }
        }
    }
    return chosen.map((id) => {
        const bits: number[] = new Array(dimensions);
        for (let j = 0; j < dimensions; j++) {
            bits[j] = (id >> j) & 1;
        }
        return bits;
    });
}
