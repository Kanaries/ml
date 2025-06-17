export type IDistance = (pos1: number[], pos2: number[], p?: number) => number;

export const manhattan: IDistance = function (pos1, pos2) {
    let dis = 0;
    for (let i = 0; i < pos1.length; i++) {
        dis += Math.abs(pos1[i] - pos2[i]);
    }
    return dis;
}

export const euclidean: IDistance = function (pos1, pos2) {
    let dis = 0;
    for (let i = 0; i < pos1.length; i++) {
        dis += (pos1[i] - pos2[i]) ** 2;
    }
    dis = Math.sqrt(dis);
    return dis;
}

export const minkowski: IDistance = function (pos1, pos2, p = 2) {
    let dis = 0;
    for (let i = 0; i < pos1.length; i++) {
        dis += (Math.abs(pos1[i] - pos2[i])) ** p;
    }
    dis = Math.pow(dis, 1 / p);
    return dis;
}
export type IDistanceType = 'manhattan' | 'euclidean' | 'minkowski' | '1-norm' | '2-norm' | 'p-norm';
export function useDistance(distanceType: IDistanceType): IDistance {
    switch (distanceType) {
        case '1-norm':
        case 'manhattan':
            return manhattan;
        case '2-norm':
        case 'euclidean':
            return euclidean;
        case 'p-norm':
        case 'minkowski':
            return minkowski;
        default:
            throw new Error(`Does not support distance type ${distanceType}`)
    }
}