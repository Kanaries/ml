import { KMeans } from '../src/clusters/kmeans';
import vega from 'vega';
import vegaLite from 'vega-lite';
import embed from 'vega-embed';
function reAssignLabel(labels: number[]): number[] {
    const encoder: Map<number, number> = new Map();
    let ans: number[] = [];
    let counter = 0;
    for (let i = 0; i < labels.length; i++) {
        if (!encoder.has(labels[i])) {
            encoder.set(labels[i], counter++);
        }
        ans.push(encoder.get(labels[i]));
    }
    return ans;
}

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
    let values = 'x,y,group,iterate\n';
    for (let it = 0; it < 6; it++) {
        const kmeans = new KMeans(3, 0.001, [
            [20, 20],
            [60, 60],
            [100, 100]
        ], it);
        const tags = reAssignLabel(kmeans.fitPredict(data));
        for (let i = 0; i < data.length; i++) {
            values += data[i].join(',');
            values += ',' + tags[i];
            values += ',' + it;
            values += '\n';
        }
    }

    embed(root, {
        data: {
            values,
            format: {
                type: 'csv'
            }
        },
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
