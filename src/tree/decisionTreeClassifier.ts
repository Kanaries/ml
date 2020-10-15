import { assert } from "../utils";
import { valuesAllSame } from "./utils";

interface IDTree {
    splitIndex: number;
    splitValue: number;
    y: number;
    children: IDTree[] | null;
} 
export class DecisionTreeClassifier {
    private dtree: IDTree | null;
    private maxDepth: number;
    public constructor () {
        this.dtree = null;
        this.maxDepth = Infinity;
    }
    /**
     * 
     * @param sampleX 
     * @param sampleY 
     * @param attributes indices of attributes.
     */
    // public treeGenerate (tree: IDTree, sampleX: number[][], sampleY: number[], attributes: number[], depth: number) {
    //     const vsame = valuesAllSame(sampleY);
    //     if (vsame) return;
    //     if (attributes.length === 0) return;
    //     if (depth > this.maxDepth) return;
    //     const splitIndex = this.attributeSelection(sampleX, sampleY, attributes);
    //     const groupIndices: Map<any, number[]> = new Map();
    //     for (let i = 0; i < sampleX.length; i++) {
    //         if (!groupIndices.has(sampleX[i][splitIndex])) {
    //             groupIndices.set(sampleX[i], []);
    //         }
    //         groupIndices.get(sampleX[i]).push(i);
    //     }
    //     tree.children = [];
    //     tree.splitIndex = splitIndex;
    //     for (let [key, indices] of groupIndices) {
    //         let child: IDTree = {
    //             attIndex: splitIndex,
    //             splitIndex: -1,
    //             value: key,
    //             children: null
    //         };
    //         tree.children.push(child);
    //         const childSampleX: number[][] = [];
    //         const childSampleY: number[] = [];
    //         for (let i of indices) {
    //             childSampleX.push(sampleX[i]);
    //             childSampleY.push(sampleY[i])
    //         }
    //         this.treeGenerate(child, sampleX, sampleY, attributes.filter(a => a !== splitIndex), depth + 1)
    //     }
    //     // if ()
    // }
    // private attributeSelection(sampleX: number[][], sampleY: number[], attributes: number[]): number {
    //     const ent = this.nodeEntropy(sampleY);
    //     let maxGain = 0;
    //     let maxGainAttIndex = 0;
    //     for (let i = 0; i < attributes.length; i++) {
    //         let attIndex = attributes[i];
    //         let groups: Map<any, number[]> = new Map();
    //         // get groups by specific att.
    //         // for each group, get ent
    //         // get gain by all entropy.
    //         for (let j = 0; j < sampleX.length; j++) {
    //             if (!groups.has(sampleX[j][attIndex])) {
    //                 // todo: samples contains index in origin samples, not all real item.
    //                 groups.set(sampleX[j][attIndex], [])
    //             }
    //             groups.get(sampleX[j][attIndex]).push(sampleY[j])
    //         }
    //         let totalEnt = 0;
    //         for (let group of groups.values()) {
    //             const subEnt = this.nodeEntropy(group);
    //             totalEnt += (group.length / sampleX.length) * subEnt;
    //         }
    //         const gain = ent - totalEnt;
    //         if (gain > maxGain) {
    //             maxGain = gain;
    //             maxGainAttIndex = i;
    //         }
    //     }
    //     return maxGainAttIndex;
    // }
    private attributeSelection(sampleX: number[][], sampleY: number[]) {
        
    }
    private nodeEntropy(sampleY: number[]): number {
        const freqMap: Map<any, number> = new Map();
        for (let i = 0; i < sampleY.length; i++) {
            if (!freqMap.has(sampleY[i])) {
                freqMap.set(sampleY[i], 0);
            }
            freqMap.set(sampleY[i], freqMap.get(sampleY[i]) + 1);
        }
        let ent = 0;
        for (let freq of freqMap.values()) {
            const p = freq / sampleY.length;
            ent += p * Math.log2(p);
        }
        return -ent;
    }
    // no cache
    // private findSampleInDTree (X: number[], tree: IDTree, depth: number): number[] {
    //     // if leaf node, then return target cat prob vec.
    //     // else find childNode of which value = X
    //     if (tree.children === null || tree.children.length === 0) {
    //         1
    //     }

    // }
    // public fit (sampleX: number[][], sampleY: number[]) {
    //     assert(sampleX.length > 0, 'fit data should not be empty');
    //     const attributes: number[] = sampleX[0].map((s, i) => i);
    //     // const splitIndex = this.attributeSelection(sampleX, sampleY, attributes);
    //     this.dtree = {
    //         attIndex: -1,
    //         value: null,
    //         splitIndex: -1,
    //         children: []
    //     };
    //     this.treeGenerate(this.dtree, sampleX, sampleY, attributes, 0);
    //     // this.dtree = {

    //     // }
    // }
    // public predict (sampleX: number[][]): number[] {

    // }
}