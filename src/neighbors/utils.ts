export type IWeightType = 'uniform' | 'distance';

export function getWeights(dis: number[], weightType: IWeightType): number[] {
    const weights = dis.map(() => 1);
    if (weightType === 'uniform') return weights;
    else if (weightType === 'distance') {
        for (let i = 0; i < weights.length; i++) {
            weights[i] = 1 / dis[i];
        }
        return weights;
    } else {
        throw new Error(`Do not support weightType: ${weightType}. Use 'uniform' or 'distance' instead.`);
    }
}

export function votes(voteClasses: number[], weights: number[]): number {
    const counter: Map<any, number> = new Map();
    for (let i = 0; i < voteClasses.length; i++) {
        if (!counter.has(voteClasses[i])) {
            counter.set(voteClasses[i], 0);
        }
        counter.set(voteClasses[i], counter.get(voteClasses[i]) + weights[i]);
    }
    let mode = 0;
    let modeCount = 0;
    for (let [m, mc] of counter) {
        if (mc > modeCount) {
            mode = m;
            modeCount = mc;
        }
    }
    return mode;
}