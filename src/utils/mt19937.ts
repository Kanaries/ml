/** Matches NumPy's legacy RandomState draws used by scikit-learn for integer seeds. */
export function mt19937(seed?: number) {
    const state = new Uint32Array(624);
    state[0] = (seed ?? 0) >>> 0;
    for (let i = 1; i < state.length; i++) {
        const previous = state[i - 1] ^ (state[i - 1] >>> 30);
        state[i] = (Math.imul(1812433253, previous) + i) >>> 0;
    }
    let index = state.length;
    function uint32(): number {
        if (index === state.length) {
            for (let i = 0; i < state.length; i++) {
                const bits = (state[i] & 0x80000000) | (state[(i + 1) % 624] & 0x7fffffff);
                state[i] = state[(i + 397) % 624] ^ (bits >>> 1) ^ (bits & 1 ? 0x9908b0df : 0);
            }
            index = 0;
        }
        let bits = state[index++];
        bits ^= bits >>> 11;
        bits ^= (bits << 7) & 0x9d2c5680;
        bits ^= (bits << 15) & 0xefc60000;
        return (bits ^ (bits >>> 18)) >>> 0;
    }

    function interval(max: number): number {
        if (seed === undefined) return Math.floor(Math.random() * (max + 1));
        let mask = max;
        mask |= mask >>> 1;
        mask |= mask >>> 2;
        mask |= mask >>> 4;
        mask |= mask >>> 8;
        mask |= mask >>> 16;
        let value: number;
        do {
            value = (uint32() & mask) >>> 0;
        } while (value > max);
        return value;
    }

    function uniform(): number {
        if (seed === undefined) return Math.random();
        const high = uint32() >>> 5;
        const low = uint32() >>> 6;
        return (high * 0x4000000 + low) / 0x20000000000000;
    }

    let spare: number | undefined;
    function normal(): number {
        if (spare !== undefined) {
            const value = spare;
            spare = undefined;
            return value;
        }
        let x: number, y: number, radius: number;
        do {
            x = 2 * uniform() - 1;
            y = 2 * uniform() - 1;
            radius = x * x + y * y;
        } while (radius >= 1 || radius === 0);
        const scale = Math.sqrt(-2 * Math.log(radius) / radius);
        spare = x * scale;
        return y * scale;
    }
    return { interval, uniform, normal };
}
