import { asyncMode } from '../asyncMode';

test('asyncMode resolves result', async () => {
    const sum = (a: number, b: number) => a + b;
    const asyncSum = asyncMode(sum);
    const result = await asyncSum(2, 3);
    expect(result).toBe(5);
});

test('asyncMode rejects on error', async () => {
    const fail = () => { throw new Error('fail'); };
    const asyncFail = asyncMode(fail);
    await expect(asyncFail()).rejects.toThrow('fail');
});
