import { det } from '../determinant'

function randInt () {
    return Math.round(Math.random() * 10);
}

test('illegal cases', () => {
    expect(() => {
        det([[3,4,5]])
    }).toThrowError();

    expect(() => {
        det([[]])
    }).toThrowError();
});

test('det 1 * 1', () => {
    const A = [[5]];
    const result = det(A);
    expect(result).toBe(5);
})

test('det 2 * 2', () => {
    const a11 = randInt();
    const a12 = randInt();
    const a21 = randInt();
    const a22 = randInt();
    const A = [
        [a11, a12],
        [a21, a22]
    ];
    const ans = a11 * a22 - a12 * a21;
    const result = det(A);
    expect(result).toBe(ans);
})

test('det 3 * 3', () => {
    const A = [
        [2, 5, 4],
        [3, 1, 2],
        [5, 4, 6]
    ];
    const result = det(A);
    expect(result).toBe(-16);
})