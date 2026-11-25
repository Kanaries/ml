import { EMeans } from '../src/clusters/eMeans';
import { KMeans } from '../src/clusters/kmeans';
import { MiniBatchKMeans } from '../src/clusters/miniBatchKMeans';
import { createRandomGenerator } from '../src/utils/random';

const preset = document.querySelector<HTMLSelectElement>('#data')!;
const budget = document.querySelector<HTMLSelectElement>('#epochs')!;
const revival = document.querySelector<HTMLInputElement>('#revival')!;
const canvas = document.querySelector('canvas')!;
const colors = ['#26599b', '#be631d', '#20755a'];

function render() {
    const rare = preset.selectedIndex === 1;
    const random = createRandomGenerator(42);
    const normal = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    const groupCenters = rare ? [[0, 0], [2, 0], [18, 6]] : [[0, 0], [4, 0], [2, 4]];
    function sample(n: number) {
        return Array.from({ length: n }, (_, i) => {
            const group = rare ? (i % 100 === 0 ? 2 : i % 2) : i % 3;
            return groupCenters[group].map(x => x + normal() * 0.7);
        });
    }
    const train = sample(900), test = sample(600), epochs = Number(budget.value);
    const models = [
        ['E-Means', new EMeans({ nClusters: 3, epochs, initCenters: groupCenters, randomState: 42, revivalThreshold: revival.checked ? 0.1 : 0 })],
        ['Lloyd', new KMeans({ n_clusters: 3, max_iter: epochs, initCenters: groupCenters, tol: 0 })],
        ['MiniBatchKMeans', new MiniBatchKMeans({ nClusters: 3, maxIter: epochs, batchSize: 36, nInit: 1, randomState: 42, maxNoImprovement: null })],
    ] as const;
    const scores = document.querySelector<HTMLTableSectionElement>('#scores')!;
    scores.replaceChildren();
    for (const [name, model] of models) {
        const labels = model.fitPredict(train), centers = model.getCentroids()!;
        const mse = test.reduce((sum, [x, y]) => sum + Math.min(...centers.map(([cx, cy]) => (x - cx) ** 2 + (y - cy) ** 2)), 0) / test.length;
        const row = scores.insertRow();
        row.insertCell().textContent = name;
        row.insertCell().textContent = mse.toFixed(3);
        if (name !== 'E-Means') continue;
        const ctx = canvas.getContext('2d')!;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const px = (x: number) => (x + 3) * 800 / (rare ? 24 : 10);
        const py = (y: number) => 340 - (y + 3) * 340 / (rare ? 12 : 10);
        train.forEach(([x, y], i) => {
            ctx.fillStyle = colors[labels[i]];
            ctx.fillRect(px(x) - 1.5, py(y) - 1.5, 3, 3);
        });
        ctx.lineWidth = 3;
        for (const [x, y] of centers) {
            ctx.beginPath(); ctx.moveTo(px(x) - 7, py(y)); ctx.lineTo(px(x) + 7, py(y));
            ctx.moveTo(px(x), py(y) - 7); ctx.lineTo(px(x), py(y) + 7); ctx.stroke();
        }
    }
}
for (const control of [preset, budget, revival]) control.addEventListener('change', render);
render();
