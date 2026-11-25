import { KMeans } from '../src/clusters/kmeans';

export function kmeansIterations(points: number[][]) {
    const rows: { x: number; y: number; group: number; iterate: number }[] = [];
    for (let iterate = 1; iterate <= 6; iterate++) {
        const model = new KMeans({
            n_clusters: 3,
            tol: 0.001,
            initCenters: [[20, 20], [60, 60], [100, 100]],
            max_iter: iterate,
        });
        const labels = model.fitPredict(points);
        points.forEach(([x, y], i) => rows.push({ x, y, group: labels[i], iterate }));
    }
    return rows;
}
