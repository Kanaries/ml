"use client";

import React, { useEffect, useRef, useState } from 'react';
import * as Plot from '@observablehq/plot';
import { Clusters } from '@kanaries/ml';

const { KMeans, DBScan, OPTICS, MeanShift, HDBScan } = Clusters;

// Generate double curve dataset - classic example for comparing clustering algorithms
function generateDoubleCurveData(n_samples = 200, noise = 0.1) {
    const data: number[][] = [];

    // First curve (top)
    for (let i = 0; i < n_samples / 2; i++) {
        const t = (i / (n_samples / 2)) * 2 * Math.PI;
        const x = Math.cos(t) + Math.random() * noise;
        const y = Math.sin(t) + 1 + Math.random() * noise;
        data.push([x, y]);
    }

    // Second curve (bottom)
    for (let i = 0; i < n_samples / 2; i++) {
        const t = (i / (n_samples / 2)) * 2 * Math.PI;
        const x = Math.cos(t) + Math.random() * noise;
        const y = Math.sin(t) - 1 + Math.random() * noise;
        data.push([x, y]);
    }

    return data;
}

// Generate two moons dataset as alternative
function generateTwoMoonsData(n_samples = 200, noise = 0.1) {
    const data: number[][] = [];

    // First moon (top)
    for (let i = 0; i < n_samples / 2; i++) {
        const t = (i / (n_samples / 2)) * Math.PI;
        const x = Math.cos(t) + Math.random() * noise;
        const y = Math.sin(t) + Math.random() * noise;
        data.push([x, y]);
    }

    // Second moon (bottom, inverted)
    for (let i = 0; i < n_samples / 2; i++) {
        const t = (i / (n_samples / 2)) * Math.PI;
        const x = 1 - Math.cos(t) + Math.random() * noise;
        const y = -Math.sin(t) - 0.5 + Math.random() * noise;
        data.push([x, y]);
    }

    return data;
}

interface ClusteringResult {
    name: string;
    data: Array<{ x: number; y: number; cluster: number }>;
    colors: string[];
}

export default function ClusteringComparison() {
    const [dataset, setDataset] = useState<'double-curve' | 'two-moons'>('two-moons');
    const [results, setResults] = useState<ClusteringResult[]>([]);
    const plotRefs = useRef<(HTMLDivElement | null)[]>([]);

    useEffect(() => {
        // Generate data based on selected dataset
        const rawData = dataset === 'double-curve' ? generateDoubleCurveData(200, 0.1) : generateTwoMoonsData(200, 0.1);

        // Apply different clustering algorithms
        const clusteringResults: ClusteringResult[] = [];

        // K-Means
        const kmeans = new KMeans(2);
        const kmeansLabels = kmeans.fitPredict(rawData);
        clusteringResults.push({
            name: 'K-Means',
            data: rawData.map((point, i) => ({
                x: point[0],
                y: point[1],
                cluster: kmeansLabels[i],
            })),
            colors: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'],
        });

        // DBSCAN
        const dbscan = new DBScan(0.3, 5);
        const dbscanLabels = dbscan.fitPredict(rawData);
        clusteringResults.push({
            name: 'DBSCAN',
            data: rawData.map((point, i) => ({
                x: point[0],
                y: point[1],
                cluster: dbscanLabels[i],
            })),
            colors: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#ddd'],
        });

        // OPTICS
        const optics = new OPTICS({ min_samples: 5, eps: 0.5 });
        const opticsLabels = optics.fitPredict(rawData);
        clusteringResults.push({
            name: 'OPTICS',
            data: rawData.map((point, i) => ({
                x: point[0],
                y: point[1],
                cluster: opticsLabels[i],
            })),
            colors: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#ddd'],
        });

        // Mean Shift
        const meanShift = new MeanShift(0.8);
        const meanShiftLabels = meanShift.fitPredict(rawData);
        clusteringResults.push({
            name: 'Mean Shift',
            data: rawData.map((point, i) => ({
                x: point[0],
                y: point[1],
                cluster: meanShiftLabels[i],
            })),
            colors: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'],
        });

        // HDBSCAN
        const hdbscan = new HDBScan(5, null, 0.3);
        const hdbscanLabels = hdbscan.fitPredict(rawData);
        clusteringResults.push({
            name: 'HDBSCAN',
            data: rawData.map((point, i) => ({
                x: point[0],
                y: point[1],
                cluster: hdbscanLabels[i],
            })),
            colors: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#ddd'],
        });

        setResults(clusteringResults);
    }, [dataset]);

    useEffect(() => {
        // Create plots for each clustering result
        results.forEach((result, index) => {
            const container = plotRefs.current[index];
            if (!container) return;

            // Clear previous plot
            container.innerHTML = '';

            // Create color scale
            const uniqueClusters = [...new Set(result.data.map((d) => d.cluster))].sort();
            const colorScale = (cluster: number) => {
                if (cluster === -1) return '#ddd'; // noise points
                return result.colors[cluster % result.colors.length];
            };

            const plot = Plot.plot({
                title: result.name,
                width: 400,
                height: 300,
                marginLeft: 50,
                marginBottom: 50,
                x: {
                    label: 'X',
                    grid: true,
                },
                y: {
                    label: 'Y',
                    grid: true,
                },
                color: {
                    legend: false,
                },
                marks: [
                    Plot.dot(result.data, {
                        x: 'x',
                        y: 'y',
                        fill: (d) => colorScale(d.cluster),
                        stroke: '#000',
                        strokeWidth: 0.5,
                        r: 3,
                        opacity: 0.8,
                    }),
                ],
            });

            container.appendChild(plot);
        });
    }, [results]);

    return (
        <div className="p-6 max-w-6xl mx-auto">
            <div className="mb-6">
                <h1 className="text-3xl font-bold mb-4">Clustering Algorithms Comparison</h1>
                <p className="text-gray-600 mb-4">
                    Compare different clustering algorithms on classic datasets. The datasets shown here are commonly
                    used to demonstrate the strengths and weaknesses of different clustering approaches.
                </p>

                <div className="flex gap-4 mb-6">
                    <button
                        onClick={() => setDataset('two-moons')}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                            dataset === 'two-moons'
                                ? 'bg-blue-500 text-white'
                                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                    >
                        Two Moons Dataset
                    </button>
                    <button
                        onClick={() => setDataset('double-curve')}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                            dataset === 'double-curve'
                                ? 'bg-blue-500 text-white'
                                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                    >
                        Double Curve Dataset
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {results.map((result, index) => (
                    <div key={result.name} className="bg-white rounded-lg shadow-md p-4">
                        <div
                            ref={(el) => {
                                plotRefs.current[index] = el;
                            }}
                            className="w-full"
                        />
                        <div className="mt-4 text-sm text-gray-600">
                            <p className="font-medium">{result.name}</p>
                            <p>Clusters found: {new Set(result.data.map((d) => d.cluster)).size}</p>
                            {result.data.some((d) => d.cluster === -1) && (
                                <p className="text-gray-500">Gray points represent noise/outliers</p>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            <div className="mt-8 bg-gray-50 rounded-lg p-6">
                <h2 className="text-xl font-semibold mb-4">Algorithm Comparison Notes</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                        <h3 className="font-medium text-blue-600 mb-2">K-Means</h3>
                        <p>
                            Assumes circular clusters and struggles with non-convex shapes. Works well when clusters are
                            spherical and similar in size.
                        </p>
                    </div>
                    <div>
                        <h3 className="font-medium text-green-600 mb-2">DBSCAN</h3>
                        <p>
                            Excellent for non-convex shapes and handles noise well. Requires tuning of epsilon and
                            min_samples parameters.
                        </p>
                    </div>
                    <div>
                        <h3 className="font-medium text-purple-600 mb-2">OPTICS</h3>
                        <p>
                            Extension of DBSCAN that works with varying densities. Creates a reachability plot for
                            cluster extraction.
                        </p>
                    </div>
                    <div>
                        <h3 className="font-medium text-orange-600 mb-2">Mean Shift</h3>
                        <p>
                            Finds clusters by shifting points towards modes of the data distribution. Automatically
                            determines cluster count.
                        </p>
                    </div>
                    <div>
                        <h3 className="font-medium text-red-600 mb-2">HDBSCAN</h3>
                        <p>
                            Hierarchical extension of DBSCAN that works well with varying densities and hierarchical
                            cluster structures.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
