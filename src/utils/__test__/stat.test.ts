import { freqs2Probs } from '../stat';

test('freqs2Probs', () => {
    const freqs: number[] = [];
    let sum = 0;
    for (let i = 0; i < 100; i++) {
        const v = Math.round(Math.random() * 100);
        freqs.push(v);
    }
    const probs = freqs2Probs(freqs);
    expect(probs.length).toBe(freqs.length);
    for (let i = 0; i < 100; i++) {
        sum += probs[i];
    }
    expect(sum).toBeCloseTo(1);
})