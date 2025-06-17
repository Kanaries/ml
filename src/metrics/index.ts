import * as Distance from './distances';


export function accuracyScore(actual: number[], expected: number[], normalize: boolean = true) {
    let score = 0;
    for (let i = 0; i < actual.length; i++) {
        if (actual[i] === expected[i]) score++;
    }
    if (normalize) {
        return score / actual.length;
    }
    return score;
}

export { Distance };