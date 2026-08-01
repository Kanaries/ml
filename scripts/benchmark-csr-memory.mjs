#!/usr/bin/env node
import { Bayes, Data } from '../build/index.mjs';

const nRows = 20_000;
const nCols = 30_000;
const nonZerosPerRow = 20;
global.gc?.();
const baselineHeap = process.memoryUsage().heapUsed;
let data = [];
let indices = [];
let indptr = [0];
const labels = new Array(nRows);

for (let row = 0; row < nRows; row++) {
    labels[row] = row % 2;
    for (let k = 0; k < nonZerosPerRow; k++) {
        data.push(1);
        indices.push((row * 31 + k * 997) % nCols);
    }
    const start = indptr[row];
    const pairs = indices.slice(start).map((column, i) => [column, data[start + i]]).sort((a, b) => a[0] - b[0]);
    for (let i = 0; i < pairs.length; i++) {
        indices[start + i] = pairs[i][0];
        data[start + i] = pairs[i][1];
    }
    indptr.push(data.length);
}

const matrix = new Data.CSRMatrix(data, indices, indptr, [nRows, nCols]);
const constructionHeapMiB = (process.memoryUsage().heapUsed - baselineHeap) / 1024 / 1024;
data = [];
indices = [];
indptr = [];
global.gc?.();
const model = new Bayes.MultinomialNB();
const started = performance.now();
model.fit(matrix, labels);
const predictions = model.predict(matrix);
const elapsedMs = performance.now() - started;
const steadyHeapMiB = (process.memoryUsage().heapUsed - baselineHeap) / 1024 / 1024;
const denseGiB = nRows * nCols * 8 / 1024 / 1024 / 1024;

if (predictions.length !== nRows) throw new Error('sparse NB prediction length mismatch');
const result = {
    shape: matrix.shape,
    nnz: matrix.nnz,
    constructionHeapMiB: +constructionHeapMiB.toFixed(1),
    steadyHeapMiB: +steadyHeapMiB.toFixed(1),
    denseGiB: +denseGiB.toFixed(1),
    fitAndPredictMs: +elapsedMs.toFixed(1),
};
if (constructionHeapMiB > 256 || steadyHeapMiB > 256) {
    throw new Error('CSR + MultinomialNB exceeded the 256 MiB browser-oriented memory budget');
}
if (elapsedMs > 30_000) {
    throw new Error(`CSR + MultinomialNB took ${elapsedMs.toFixed(1)} ms; expected <= 30000 ms`);
}
console.log(JSON.stringify(result));
