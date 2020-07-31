export function mode (data: number[]): number {
    const counter: Map<any, number> = new Map();
    for (let record of data) {
        if (!counter.has(record)) {
            counter.set(record, 0)
        }
        counter.set(record, counter.get(record) + 1);
    }
    let mode = 0;
    let modeCount = 0;
    for (let [m, mc] of counter) {
        if (mc > modeCount) {
            mode = m;
            mc = modeCount;
        }
    }
    return mode;
}
