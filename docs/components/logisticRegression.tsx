"use client";

import React, { useEffect, useRef, useState } from 'react';
import * as Plot from '@observablehq/plot';
import { Linear } from '@kanaries/ml';

const { LogisticRegression } = Linear;

function generateData(n_samples = 200, noise = 0.5) {
    const X: number[][] = [];
    const Y: number[] = [];
    for (let i = 0; i < n_samples; i++) {
        const x1 = Math.random() * 4 - 2;
        const x2 = Math.random() * 4 - 2;
        const label = x1 + x2 + (Math.random() - 0.5) * noise > 0 ? 1 : 0;
        X.push([x1, x2]);
        Y.push(label);
    }
    return { X, Y };
}

export default function LogisticRegressionDemo() {
    const plotRef = useRef<HTMLDivElement | null>(null);
    const [seed, setSeed] = useState(0);

    useEffect(() => {
        const { X, Y } = generateData();
        const lr = new LogisticRegression({ learningRate: 0.2, maxIter: 800 });
        lr.fit(X, Y);
        const weights = (lr as any).weights as number[];
        const bias = (lr as any).bias as number;
        const preds = lr.predict(X);
        const data = X.map((p, idx) => ({ x: p[0], y: p[1], label: preds[idx] }));

        const container = plotRef.current;
        if (!container) return;
        container.innerHTML = '';

        const linePoints = [-2.5, 2.5].map((x) => ({ x, y: -(weights[0] * x + bias) / weights[1] }));

        const plot = Plot.plot({
            width: 420,
            height: 400,
            marginLeft: 50,
            marginBottom: 50,
            x: { label: 'X1', grid: true, domain: [-2.5, 2.5] },
            y: { label: 'X2', grid: true, domain: [-2.5, 2.5] },
            color: { legend: false },
            marks: [
                Plot.dot(data, {
                    x: 'x',
                    y: 'y',
                    fill: (d) => (d.label ? '#4ecdc4' : '#ff6b6b'),
                    r: 4,
                    opacity: 0.8,
                }),
                Plot.line(linePoints, { x: 'x', y: 'y', stroke: 'black' }),
            ],
        });

        container.appendChild(plot);
    }, [seed]);

    return (
        <div className="p-4">
            <button
                onClick={() => setSeed((s) => s + 1)}
                className="mb-4 px-3 py-1 rounded bg-blue-500 text-white"
            >
                New Data
            </button>
            <div ref={plotRef} />
        </div>
    );
}
