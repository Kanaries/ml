import { transpose, product } from "./basic";
import { elementary as inverse } from './inverse';

export function LinearRegression(X: number[][], y: number[][]) {
    const XT = transpose(X);
    let beta: number[][] | boolean = product(XT, X);
    beta = inverse(beta);
    if (beta) {
        beta = product(beta, XT);
        beta = product(beta, y);
    }
    return beta;
}
