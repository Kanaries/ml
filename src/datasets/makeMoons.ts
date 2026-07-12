import {
    createRandomGenerator,
    createGaussianGenerator,
    shuffleInUnison,
    Dataset,
} from './common';

export interface MakeMoonsProps {
    /**
     * total number of samples (split between the two moons, the outer moon
     * receiving floor(n / 2)), or [nOuter, nInner]
     */
    nSamples?: number | [number, number];
    /** standard deviation of gaussian noise added to the points */
    noise?: number;
    /** shuffle the samples (default true) */
    shuffle?: boolean;
    /** seed for reproducible output */
    randomState?: number;
}

/**
 * Generates two interleaving half circles
 * (port of sklearn.datasets.make_moons). Label 0 is the outer moon
 * (upper half of the unit circle centered at the origin), label 1 the
 * inner moon (lower half circle centered at (1, 0.5)).
 */
export function makeMoons(props: MakeMoonsProps = {}): Dataset {
    const {
        nSamples = 100,
        noise,
        shuffle = true,
        randomState,
    } = props;

    let nOut: number;
    let nIn: number;
    if (Array.isArray(nSamples)) {
        if (nSamples.length !== 2) {
            throw new Error('nSamples must be a number or a two-element array [nOuter, nInner]');
        }
        [nOut, nIn] = nSamples;
    } else {
        nOut = Math.floor(nSamples / 2);
        nIn = nSamples - nOut;
    }

    const rng = createRandomGenerator(randomState);
    const gaussian = createGaussianGenerator(rng);

    let X: number[][] = [];
    let y: number[] = [];
    for (let i = 0; i < nOut; i++) {
        // linspace(0, pi, nOut), endpoint included
        const t = nOut > 1 ? (Math.PI * i) / (nOut - 1) : 0;
        X.push([Math.cos(t), Math.sin(t)]);
        y.push(0);
    }
    for (let i = 0; i < nIn; i++) {
        const t = nIn > 1 ? (Math.PI * i) / (nIn - 1) : 0;
        X.push([1 - Math.cos(t), 1 - Math.sin(t) - 0.5]);
        y.push(1);
    }

    if (noise !== undefined) {
        for (const sample of X) {
            sample[0] += noise * gaussian();
            sample[1] += noise * gaussian();
        }
    }

    if (shuffle) {
        ({ X, y } = shuffleInUnison(X, y, rng));
    }
    return { X, y };
}
