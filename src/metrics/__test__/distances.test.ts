import { manhattan, euclidean, minkowski, useDistance } from '../distances';
function randPosPair (size: number): [number[], number[]] {
    const pos1 = new Array(size).fill(0).map(() => Math.round(Math.random() * 20 - 10));
    const pos2 = new Array(size).fill(0).map(() => Math.round(Math.random() * 20 - 10));
    return [pos1, pos2];
}
test('useDistance', () => {
    expect(useDistance('euclidean')).toBe(euclidean);
    expect(useDistance('2-norm')).toBe(euclidean);
    expect(useDistance('1-norm')).toBe(manhattan);
    expect(useDistance('manhattan')).toBe(manhattan);
    expect(useDistance('minkowski')).toBe(minkowski);
    expect(useDistance('p-norm')).toBe(minkowski);
})

test('manhattan', () => {
    for (let i = 0; i < 10; i++) {
        const [pos1, pos2] = randPosPair(5);
        const dis = pos1.reduce((total, p1, index) => total + Math.abs(p1 - pos2[index]), 0);
        expect(manhattan(pos1, pos2)).toBe(dis);
    }
})

test('euclidean', () => {
    for (let i = 0; i < 10; i++) {
        const [pos1, pos2] = randPosPair(5);
        let dis = pos1.reduce((total, p1, index) => total + Math.pow(p1 - pos2[index], 2), 0);
        dis = Math.sqrt(dis);
        expect(euclidean(pos1, pos2)).toBeCloseTo(dis);
    }
});

test('minkowski', () => {
    for (let i = 0; i < 10; i++) {
        const [pos1, pos2] = randPosPair(5);
        let dis = pos1.reduce((total, p1, index) => total + Math.pow(Math.abs(p1 - pos2[index]), 3), 0);
        dis = Math.cbrt(dis);
        expect(minkowski(pos1, pos2, 3)).toBeCloseTo(dis);
    }
});