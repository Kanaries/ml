export function std<T = any> (arr: T[], size: number): T[] {
    if (arr.length <= size) return arr;
    let choosen = arr.map(() => false);
    let samples: T[] = [];
    let count = 0;
    while (count < size) {
        let index = Math.floor(Math.random() * size)
        while (choosen[index]) {
            index++;
        }
        if (!choosen[index]) {
            choosen[index] = true;
            samples.push(arr[index]);
            count++;
        }
    }
    return arr;
}