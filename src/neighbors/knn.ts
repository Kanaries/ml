import { ClassifierBase } from "../base";
import { Distance } from "../metrics";
import { mode } from "../utils/stat";
import { getWeights, IWeightType, votes } from "./utils";

export class KNearstNeighbors extends ClassifierBase {
           private samplesX: number[][];
           private samplesY: number[];
           private distance: Distance.IDistance;
           private kNeighbors: number;
           private weightType: IWeightType;
           private pNorm: number;
           public constructor(
               kNeighbors: number = 5,
               weightType: IWeightType = 'uniform',
               distanceType: Distance.IDistanceType = 'euclidiean',
               pNorm: number = 2
           ) {
               super();
               this.samplesX = [];
               this.samplesY = [];
               this.distance = Distance.useDistance(distanceType);
               this.kNeighbors = kNeighbors;
               this.weightType = weightType;
               this.pNorm = pNorm;
           }
           public fit(trainX: number[][], trainY: number[]): void {
               this.samplesX = trainX;
               this.samplesY = trainY;
           }
           public predict(testX: number[][]): number[] {
               let Y: number[] = [];
               const { distance, kNeighbors, samplesY, weightType, pNorm } = this;
               for (let i = 0; i < testX.length; i++) {
                   const x = testX[i];
                   const neighbors: Array<{ index: number; dis: number }> = this.samplesX.map((sample, index) => {
                       return {
                           index,
                           dis: distance(x, sample, pNorm),
                       };
                   });
                   neighbors.sort((a, b) => a.dis - b.dis);
                   const knns = neighbors.slice(0, kNeighbors);
                   const classes = knns.map((nei) => samplesY[nei.index]);
                   const weights = getWeights(
                       knns.map((n) => n.dis),
                       weightType
                   );
                   Y.push(votes(classes, weights));
               }
               return Y;
           }
       }