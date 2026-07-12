import {
    createRandomGenerator,
    createGaussianGenerator,
    shuffleInUnison,
    Dataset,
} from './common';

export interface MakeCirclesProps {
    /**
     * total number of samples (split between the two circles, the outer
     * circle receiving floor(n / 2)), or [nOuter, nInner]
     */
    nSamples?: number | [number, number];
    /** standard deviation of gaussian noise added to the points */
    noise?: number;
    /** scale factor between inner and outer circle, in [0, 1) */
    factor?: number;
    /** shuffle the samples (default true) */
    shuffle?: boolean;
    /** seed for reproducible output */
    randomState?: number;
}

/**
 * Generates a large circle containing a smaller circle
 * (port of sklearn.datasets.make_circles). Label 0 is the outer unit
 * circle, label 1 the inner circle of radius `factor`.
 */
export function makeCircles(props: MakeCirclesProps = {}): Dataset {
    const {
        nSamples = 100,
        noise,
        factor = 0.8,
        shuffle = true,
        randomState,
    } = props;

    if (factor >= 1 || factor < 0) {
        throw new Error("'factor' has to be between 0 and 1");
    }

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
        // linspace(0, 2*pi, nOut, endpoint=False)
        const t = (2 * Math.PI * i) / nOut;
        X.push([Math.cos(t), Math.sin(t)]);
        y.push(0);
    }
    for (let i = 0; i < nIn; i++) {
        const t = (2 * Math.PI * i) / nIn;
        X.push([factor * Math.cos(t), factor * Math.sin(t)]);
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
