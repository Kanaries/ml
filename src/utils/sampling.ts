export function std<T = any> (arr: T[], size: number): T[] {
    if (arr.length <= size) return arr;
    let choosen = arr.map(() => false);
    let samples: T[] = [];
    let count = 0;
    while (count < size) {
        const index = Math.floor(Math.random() * size)
        if (!choosen[index]) {
            choosen[index] = true;
            samples.push(arr[index]);
        }
    }
    return arr;
}