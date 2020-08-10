import { KMeans } from '../kmeans';

test('init', () => {
    const kmeans = new KMeans();
    expect(kmeans).toBeDefined();
})

test('kmeans results', () => {
    const X = [
        [0, 0],
        [0.5, 0],
        [0.5, 1],
        [1, 1],
    ];
    const sampleWeights = [3, 1, 1, 3];
    const initCenters = [[0, 0], [1, 1]];
    const expected = [0, 0, 1, 1];

    const kmeans = new KMeans(2, 0.05, initCenters);
    const result = kmeans.fitPredict(X, sampleWeights);
    console.log(result)
    expect(result).toEqual(expected);
})
function reAssignLabel (labels: number[]): number[] {
    const encoder: Map<number, number> = new Map();
    let ans: number[] = [];
    let counter = 0;
    for (let i = 0; i < labels.length; i++) {
        if (!encoder.has(labels[i])) {
            encoder.set(labels[i], counter++);
        }
        ans.push(encoder.get(labels[i]))
    }
    return ans;
}
test('kmeans bad centers', () => {
    const X = [
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ];
    const initCenters = [
        [+0, 1, 0, 0],
        [0.2, 0, 0.2, 0.2],
        [+0, 0, 0, 0],
    ];
    const expected = [0, 1, 2, 1, 1, 2];

    const kmeans = new KMeans(3, 0.05, initCenters, 10);
    const result = reAssignLabel(kmeans.fitPredict(X));

    expect(result).toEqual(expected);
});