/**
 * Park-Miller LCG. Returns Math.random when no seed is given,
 * otherwise a deterministic generator in [0, 1).
 */
export function createRandomGenerator(seed?: number): () => number {
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
