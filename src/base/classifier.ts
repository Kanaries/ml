import { accuracyScore } from "../metrics";

export abstract class ClassifierBase {
    public abstract fit(trainX: number[][], trainY: number[]): void;
    public abstract predict(testX: number[][]): number[];
    public score (X: number[][], Y: number[]): number {
        const resultY = this.predict(X);
        return accuracyScore(resultY, Y);
    }
}
