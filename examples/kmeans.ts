import { kmeansIterations } from './kmeansIterations';
import embed from 'vega-embed';

function getSamplesAroundCenter(center: number[], size: number = 2000): number[][] {
    const scale = 12;
    let ans: number[][] = [];
    for (let i = 0; i < size; i++) {
        const randX = Math.random() * 6 - 3;
        const randY = Math.random() * 6 - 3;
        const rand = Math.random();
        if (
            rand < (1 / Math.sqrt(2 * Math.PI)) * Math.pow(Math.E, -(randX ** 2) / 2) &&
            rand < (1 / Math.sqrt(2 * Math.PI)) * Math.pow(Math.E, -(randY ** 2) / 2)
        ) {
            ans.push([center[0] + randX * scale, center[1] + randY * scale * 6]);
        }
    }
    return ans;
}
export function kmeansTest () {
    const root = document.querySelector('#root') as HTMLDivElement;

    let data: number[][] = [];
    let groups1 = getSamplesAroundCenter([10, 8])
    let groups2 = getSamplesAroundCenter([80, 120])
    let groups3 = getSamplesAroundCenter([150, 20])
    data.push(...groups1, ...groups2, ...groups3)
    embed(root, {
        data: { values: kmeansIterations(data) },
        mark: 'point',
        encoding: {
            x: {
                field: 'x',
                type: 'quantitative',
            },
            y: {
                field: 'y',
                type: 'quantitative',
            },
            color: {
                field: 'group',
                type: 'nominal'
            },
            facet: {
                field: 'iterate',
                type: 'ordinal',
                columns: 3
            }
        },
    });
}
