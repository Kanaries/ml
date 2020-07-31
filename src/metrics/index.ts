export * as Distance from './distances';

export function accuracyScore(acutal: number[], expected: number[], normalize: boolean = true) {
    let score = 0;
    for (let i = 0; i < acutal.length; i++) {
        if (acutal[i] !== expected[i]) score++;
    }
    if (normalize) {
        return score /= acutal.length;
    }
    return score;
}