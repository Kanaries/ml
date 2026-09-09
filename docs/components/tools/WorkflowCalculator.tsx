'use client';
import { useMemo, useRef, useState } from 'react';
import { CodeTabs } from './CodeTabs';
import { downloadText, downloadSvgAsPng } from './clientUtils';
import { csv, numeric, parseCsv, parseNumberList } from './workflowMath';
import s from './workflow.module.css';

import { calculateWorkflow, type WorkflowKind, type WorkflowResult } from './workflowResult';
export type { WorkflowKind } from './workflowResult';
const samples = {
    outlier: 'record_id,value\n001,10\n002,11\n003,12\n004,12\n005,13\n006,14\n007,15\n008,16\n009,17\n010,80',
    polynomial:
        'x,y\n0,2\n1,4\n2,10\n3,20\n4,34\n5,52\n6,74\n7,100\n8,130\n9,164\n10,202\n11,244\n12,290\n13,340\n14,394',
    customer:
        'customer_id,recency_days,orders,spend\n001,4,15,1200\n002,7,12,1000\n003,3,18,1500\n004,60,2,90\n005,90,1,40\n006,75,2,110\n007,20,6,400\n008,25,5,350\n009,18,7,480\n010,8,13,1150\n011,85,1,60\n012,22,6,420',
};
const transactions =
    'customer_id,date,amount\n001,2026-08-28,120\n001,2026-08-30,80\n002,2026-08-29,190\n003,2026-06-01,20\n004,2026-06-15,25\n005,2026-08-10,60\n005,2026-08-15,50\n006,2026-08-12,90';

const fmt = (n: number | null) => (n === null ? 'Undefined' : Number(n.toPrecision(6)).toString());
function Plot({
    points,
    line = [],
    title,
    xLabel,
    yLabel,
}: {
    points: [number, number, boolean][];
    line?: [number, number][];
    title: string;
    xLabel: string;
    yLabel: string;
}) {
    const ref = useRef<SVGSVGElement>(null);
    const all = [...points, ...line],
        xs = all.map((p) => p[0]),
        ys = all.map((p) => p[1]);
    const xmin = Math.min(...xs),
        xmax = Math.max(...xs),
        ymin = Math.min(...ys),
        ymax = Math.max(...ys);
    const px = (x: number) => 65 + (530 * (x - xmin)) / (xmax - xmin || 1),
        py = (y: number) => 265 - (220 * (y - ymin)) / (ymax - ymin || 1);
    return (
        <section>
            <h3>{title}</h3>
            <svg
                ref={ref}
                className={s.plot}
                viewBox="0 0 640 330"
                role="img"
                aria-label={`${title}. ${points.length} observations. ${xLabel} versus ${yLabel}.`}
            >
                <rect width="640" height="330" fill="#fff" />
                <path d="M65 35V265H600" fill="none" stroke="#64748b" />
                <text x="65" y="288" fill="#334155" fontSize="13">
                    {fmt(xmin)}
                </text>
                <text x="595" y="288" textAnchor="end" fill="#334155" fontSize="13">
                    {fmt(xmax)}
                </text>
                <text x="58" y="50" textAnchor="end" fill="#334155" fontSize="12">
                    {fmt(ymax)}
                </text>
                <text x="58" y="265" textAnchor="end" fill="#334155" fontSize="12">
                    {fmt(ymin)}
                </text>
                <text x="330" y="315" textAnchor="middle" fill="#334155" fontSize="14">
                    {xLabel}
                </text>
                <text x="65" y="22" fill="#334155" fontSize="14">
                    {yLabel}
                </text>
                {line.length > 0 && (
                    <polyline
                        points={line.map(([x, y]) => `${px(x)},${py(y)}`).join(' ')}
                        stroke="#059669"
                        strokeWidth="2"
                        fill="none"
                    />
                )}
                {points.map(([x, y, flag], i) => (
                    <circle
                        key={i}
                        cx={px(x)}
                        cy={py(y)}
                        r={flag ? 5 : 3.5}
                        fill={flag ? '#dc2626' : '#2563eb'}
                        fillOpacity=".8"
                    >
                        <title>{`Row ${i + 1}: ${xLabel}=${x}, ${yLabel}=${y}${flag ? ', flagged' : ''}`}</title>
                    </circle>
                ))}
            </svg>
            <button
                type="button"
                onClick={() => downloadSvgAsPng(ref.current, `${title.toLowerCase().replaceAll(' ', '-')}.png`)}
            >
                Download chart PNG
            </button>
        </section>
    );
}
function BoxPlot({ values }: { values: number[] }) {
    const ref = useRef<SVGSVGElement>(null),
        [lo, q1, med, q3, hi] = values;
    const px = (v: number) => 65 + (510 * (v - lo)) / (hi - lo || 1);
    return (
        <>
            <svg
                ref={ref}
                className={s.plot}
                viewBox="0 0 640 125"
                role="img"
                aria-label={`IQR box plot: lower whisker ${lo}, Q1 ${q1}, median ${med}, Q3 ${q3}, upper whisker ${hi}`}
            >
                <rect width="640" height="125" fill="white" />
                <path d={`M${px(lo)} 60H${px(hi)} M${px(lo)} 45V75 M${px(hi)} 45V75`} stroke="#334155" fill="none" />
                <rect
                    x={px(q1)}
                    y="35"
                    width={Math.max(1, px(q3) - px(q1))}
                    height="50"
                    fill="#dbeafe"
                    stroke="#2563eb"
                />
                <path d={`M${px(med)} 35V85`} stroke="#2563eb" strokeWidth="3" />
                <text
                    x="65"
                    y="112"
                    fill="#334155"
                    fontSize="14"
                >{`Whiskers ${fmt(lo)}–${fmt(hi)} · Q1 ${fmt(q1)} · Median ${fmt(med)} · Q3 ${fmt(q3)}`}</text>
            </svg>
            <button type="button" onClick={() => downloadSvgAsPng(ref.current, 'iqr-box-plot.png')}>
                Download box plot PNG
            </button>
        </>
    );
}
function TablePreview({ headers, rows }: { headers: string[]; rows: (string | number | boolean | null)[][] }) {
    return (
        <div className={s.table}>
            <table>
                <thead>
                    <tr>
                        {headers.map((h, i) => (
                            <th key={i}>{h}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.slice(0, 20).map((r, i) => (
                        <tr key={i}>
                            {r.map((v, j) => (
                                <td key={j}>{typeof v === 'number' ? fmt(v) : v === null ? '—' : String(v)}</td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
            {rows.length > 20 && <p>Showing 20 of {rows.length} rows. Export includes every row.</p>}
        </div>
    );
}
export function WorkflowCalculator({ kind }: { kind: WorkflowKind }) {
    const [text, setText] = useState(samples[kind]),
        [a, setA] = useState(kind === 'outlier' ? 1 : 0),
        [b, setB] = useState(1),
        [c, setC] = useState(2);
    const [method, setMethod] = useState('iqr'),
        [degree, setDegree] = useState(2),
        [k, setK] = useState(3),
        [mode, setMode] = useState('features'),
        [reference, setReference] = useState('2026-09-01');
    const [inputMode, setInputMode] = useState('csv'),
        [predictX, setPredictX] = useState('3.5');
    const [error, setError] = useState(''),
        [result, setResult] = useState<(WorkflowResult & { key: string }) | null>(null);
    const parsed = useMemo(() => {
        try {
            return { table: inputMode === 'list' ? parseNumberList(text) : parseCsv(text), error: '' };
        } catch (e) {
            return { table: null, error: (e as Error).message };
        }
    }, [text, inputMode]);
    const key = JSON.stringify([text, a, b, c, method, degree, k, mode, reference, inputMode]);
    const visible = result?.key === key ? result : null;
    function select(label: string, value: number, update: (v: number) => void) {
        return (
            <label>
                {label}
                <select value={value} onChange={(e) => update(Number(e.target.value))}>
                    {parsed.table?.headers.map((h, i) => (
                        <option value={i} key={h}>
                            {h}
                        </option>
                    ))}
                </select>
            </label>
        );
    }
    function run() {
        try {
            if (!parsed.table) throw new Error(parsed.error);
            const output = calculateWorkflow(parsed.table, kind, {
                a,
                b,
                c,
                method,
                inputMode,
                degree,
                k,
                mode,
                reference,
            });
            setResult({ ...output, key });
            setError('');
            window.dispatchEvent(
                new CustomEvent('ml-tool-result', { detail: { tool: kind, rows: parsed.table.rows.length } }),
            );
        } catch (e) {
            setError((e as Error).message);
            setResult(null);
        }
    }
    let predicted: string | null = null;
    if (visible?.model) {
        try {
            const x = numeric(predictX, 'Prediction X'),
                model = visible.model,
                z = (x - model.center) / model.scale;
            const y = model.coefficients.reduce((sum, coef, i) => sum + coef * z ** i, 0);
            predicted = Number.isFinite(y) ? fmt(y) : 'Prediction exceeds the numerical range.';
        } catch (e) {
            predicted = (e as Error).message;
        }
    }
    return (
        <div className={s.root}>
            <div className={s.inputs}>
                <div>
                    {kind === 'outlier' && (
                        <label>
                            Input type
                            <select
                                value={inputMode}
                                onChange={(e) => {
                                    setInputMode(e.target.value);
                                    setA(1);
                                    setText(
                                        e.target.value === 'list'
                                            ? '10, 11, 12, 12, 13, 14, 15, 16, 17, 80'
                                            : samples.outlier,
                                    );
                                }}
                            >
                                <option value="csv">CSV with header</option>
                                <option value="list">Number list</option>
                            </select>
                        </label>
                    )}
                    <label htmlFor={`${kind}-csv`}>
                        {inputMode === 'list'
                            ? 'Numbers separated by commas or whitespace'
                            : 'CSV data (first row contains column names)'}
                    </label>
                    <textarea
                        id={`${kind}-csv`}
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        spellCheck={false}
                    />
                    <div className={s.actions}>
                        <button
                            type="button"
                            onClick={() => {
                                setText(samples[kind]);
                                setInputMode('csv');
                                setA(kind === 'outlier' ? 1 : 0);
                                setB(1);
                                setC(2);
                                setMode('features');
                                setError('');
                            }}
                        >
                            Load sample
                        </button>
                        <a href={`/samples/${kind}.csv`} download>
                            Download sample CSV
                        </a>
                    </div>
                    <label>
                        Open local CSV or TSV
                        <input
                            type="file"
                            accept=".csv,.tsv,text/csv,text/tab-separated-values"
                            onChange={async (e) => {
                                const f = e.target.files?.[0];
                                if (!f) return;
                                if (f.size > 1_000_000) {
                                    setError('Use a file smaller than 1 MB.');
                                    return;
                                }
                                try {
                                    setInputMode('csv');
                                    setText(await f.text());
                                    setError('');
                                } catch {
                                    setError('Could not read this file. Try pasting the CSV.');
                                }
                                e.target.value = '';
                            }}
                        />
                    </label>
                    <p>
                        Data stays in this tab. Limits: 2,000 rows, 30 columns; customer clustering: 500 customers and
                        12 features.
                    </p>
                </div>
                <div className={s.controls}>
                    {kind === 'customer' && (
                        <>
                            <label>
                                Input format
                                <select value={mode} onChange={(e) => setMode(e.target.value)}>
                                    <option value="features">One row per customer, numeric features</option>
                                    <option value="rfm">Transactions → RFM features</option>
                                </select>
                            </label>
                            <button
                                type="button"
                                onClick={() => {
                                    setText(transactions);
                                    setMode('rfm');
                                    setA(0);
                                    setB(1);
                                    setC(2);
                                    setK(3);
                                }}
                            >
                                Load transaction example
                            </button>
                        </>
                    )}
                    {inputMode !== 'list' &&
                        select(
                            kind === 'outlier'
                                ? 'Value column'
                                : kind === 'customer'
                                  ? 'Customer ID column'
                                  : 'X column',
                            a,
                            setA,
                        )}
                    {kind === 'outlier' && (
                        <label>
                            Detection rule
                            <select value={method} onChange={(e) => setMethod(e.target.value)}>
                                <option value="iqr">IQR: 1.5 × interquartile range</option>
                                <option value="z">Z-score: |z| &gt; 3 (population SD)</option>
                                <option value="modified">Modified Z-score: |z| &gt; 3.5</option>
                            </select>
                        </label>
                    )}
                    {kind === 'polynomial' && (
                        <>
                            {select('Y column', b, setB)}
                            <label>
                                Polynomial degree
                                <input
                                    type="number"
                                    min="1"
                                    max="6"
                                    value={degree}
                                    onChange={(e) => setDegree(Number(e.target.value))}
                                />
                            </label>
                        </>
                    )}
                    {kind === 'customer' && (
                        <>
                            {mode === 'rfm' && (
                                <>
                                    {select('Date column', b, setB)}
                                    {select('Amount column', c, setC)}
                                    <label>
                                        Reference date (UTC)
                                        <input
                                            type="date"
                                            value={reference}
                                            onChange={(e) => setReference(e.target.value)}
                                        />
                                    </label>
                                </>
                            )}
                            <label>
                                Number of groups
                                <input
                                    type="number"
                                    min="2"
                                    max="8"
                                    value={k}
                                    onChange={(e) => setK(Number(e.target.value))}
                                />
                            </label>
                        </>
                    )}
                    <button className={s.primary} type="button" onClick={run}>
                        Calculate results
                    </button>
                    <p>
                        Choose columns, then calculate. Changing an input hides the previous result until you run again.
                    </p>
                </div>
            </div>
            {(error || parsed.error) && (
                <p role="alert" className={s.error}>
                    {error || parsed.error}
                </p>
            )}
            {parsed.table && (
                <details>
                    <summary>Preview input ({parsed.table.rows.length} rows)</summary>
                    <TablePreview {...parsed.table} />
                </details>
            )}
            {visible && (
                <div aria-live="polite">
                    <div className={s.stats}>
                        {visible.stats.map(([label, value]) => (
                            <div key={label}>
                                <span>{label}</span>
                                <strong>{value}</strong>
                            </div>
                        ))}
                    </div>
                    <p>{visible.note}</p>
                    {visible.box && (
                        <section>
                            <h3>IQR box plot</h3>
                            <p>
                                Box: Q1 to Q3; center: median; whiskers: most extreme observations inside the IQR
                                fences. This summary always uses IQR, independently of the selected flagging rule.
                            </p>
                            <BoxPlot values={visible.box} />
                        </section>
                    )}
                    {visible.model && (
                        <div>
                            <label>
                                Predict Y at X
                                <input
                                    value={predictX}
                                    onChange={(e) => setPredictX(e.target.value)}
                                    inputMode="decimal"
                                />
                            </label>
                            <p>
                                Predicted Y: <strong>{predicted}</strong>. Values outside the observed X range are
                                extrapolations.
                            </p>
                        </div>
                    )}
                    <Plot
                        points={visible.points}
                        line={visible.line}
                        title={
                            kind === 'outlier'
                                ? 'Outlier review'
                                : kind === 'polynomial'
                                  ? 'Fitted curve'
                                  : 'Customer groups'
                        }
                        xLabel={
                            kind === 'outlier'
                                ? 'Data row'
                                : kind === 'polynomial'
                                  ? (parsed.table?.headers[a] ?? 'X')
                                  : (visible.features?.[0] ?? 'Feature')
                        }
                        yLabel={
                            kind === 'outlier'
                                ? (parsed.table?.headers[a] ?? 'Value')
                                : kind === 'polynomial'
                                  ? (parsed.table?.headers[b] ?? 'Y')
                                  : 'Group ID'
                        }
                    />
                    {visible.extra && (
                        <Plot
                            points={visible.extra}
                            title="Residuals"
                            xLabel={parsed.table?.headers[a] ?? 'X'}
                            yLabel="Observed minus predicted"
                        />
                    )}
                    {visible.groups && (
                        <>
                            <h3>Group profiles in original units</h3>
                            <TablePreview
                                headers={['Group', 'Customers', ...(visible.features ?? [])]}
                                rows={visible.groups.map((g) => [g.label, g.count, ...g.averages])}
                            />
                        </>
                    )}
                    <h3>Results</h3>
                    <TablePreview headers={visible.headers} rows={visible.rows} />
                    <button
                        type="button"
                        onClick={() => downloadText(`${kind}-results.csv`, csv([visible.headers, ...visible.rows]))}
                    >
                        Export all results CSV
                    </button>
                    <button
                        type="button"
                        onClick={() =>
                            downloadText(
                                `${kind}-analysis.json`,
                                JSON.stringify(
                                    {
                                        tool: kind,
                                        settings: {
                                            method: kind === 'outlier' ? method : undefined,
                                            degree: kind === 'polynomial' ? degree : undefined,
                                            k: kind === 'customer' ? k : undefined,
                                            mode: kind === 'customer' ? mode : undefined,
                                            reference: kind === 'customer' && mode === 'rfm' ? reference : undefined,
                                        },
                                        stats: visible.stats,
                                        model: visible.model,
                                        note: visible.note,
                                        headers: visible.headers,
                                        rows: visible.rows,
                                        javascript: visible.js,
                                        python: visible.py,
                                    },
                                    null,
                                    2,
                                ),
                                'application/json',
                            )
                        }
                    >
                        Export analysis JSON with settings and code
                    </button>
                    <h3>Reproduce this calculation</h3>
                    <p>
                        The code contains the current numeric data. JavaScript runs in browser or Node.js; Python
                        requires NumPy and scikit-learn for regression and clustering.
                    </p>
                    <CodeTabs javascript={visible.js} python={visible.py} />
                </div>
            )}
        </div>
    );
}
