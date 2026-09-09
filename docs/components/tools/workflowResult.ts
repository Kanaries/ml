import { column, numeric, outliers, resultHeaders, polynomial, rfm, segment, type Table } from './workflowMath';
export type WorkflowKind = 'outlier' | 'polynomial' | 'customer';
export type WorkflowResult = {
    headers: string[];
    rows: (string | number | boolean | null)[][];
    stats: [string, string][];
    points: [number, number, boolean][];
    line?: [number, number][];
    extra?: [number, number, boolean][];
    js: string;
    py: string;
    note: string;
    groups?: { label: number; count: number; averages: (number | null)[] }[];
    features?: string[];
    box?: number[];
    model?: { center: number; scale: number; coefficients: number[] };
};
const fmt = (n: number | null) => (n === null ? 'Undefined' : Number(n.toPrecision(6)).toString());
export type WorkflowSettings = {
    a: number;
    b: number;
    c: number;
    method: string;
    inputMode: string;
    degree: number;
    k: number;
    mode: string;
    reference: string;
};
export function calculateWorkflow(
    t: Table,
    kind: WorkflowKind,
    { a, b, c, method, inputMode, degree, k, mode, reference }: WorkflowSettings,
): WorkflowResult {
    let output: WorkflowResult;
    if (kind === 'outlier') {
        const values = column(t, inputMode === 'list' ? 1 : a),
            fit = outliers(values, method),
            iqr = outliers(values, 'iqr');
        const js = `const values = ${JSON.stringify(values)};\nconst sorted = [...values].sort((a,b)=>a-b);\nconst q = (a,p) => { const i=(a.length-1)*p,j=Math.floor(i); return a[j]+(a[Math.ceil(i)]-a[j])*(i-j); };\nconst median=q(sorted,.5), avg=values.reduce((a,b)=>a+b,0)/values.length;\nconst sd=Math.sqrt(values.reduce((s,v)=>s+(v-avg)**2,0)/values.length);\nconst mad=q(values.map(v=>Math.abs(v-median)).sort((a,b)=>a-b),.5);\nconst lower=${method === 'iqr' ? 'q(sorted,.25)-1.5*(q(sorted,.75)-q(sorted,.25))' : method === 'z' ? 'avg-3*sd' : 'median-3.5*mad/.6744897501960817'};\nconst upper=${method === 'iqr' ? 'q(sorted,.75)+1.5*(q(sorted,.75)-q(sorted,.25))' : method === 'z' ? 'avg+3*sd' : 'median+3.5*mad/.6744897501960817'};\nconsole.log(values.map(value=>({value,outlier:value<lower||value>upper})));`;
        const py = `import numpy as np\nx = np.array(${JSON.stringify(values)}, dtype=float)\nq1,q3 = np.quantile(x,[.25,.75],method="linear")\nmedian=np.median(x)\nmad=np.median(np.abs(x-median))\nlower,upper = ${method === 'iqr' ? '(q1-1.5*(q3-q1),q3+1.5*(q3-q1))' : method === 'z' ? '(x.mean()-3*x.std(ddof=0),x.mean()+3*x.std(ddof=0))' : '(median-3.5*mad/.6744897501960817,median+3.5*mad/.6744897501960817)'}\nprint((x<lower)|(x>upper))`;
        output = {
            headers: resultHeaders(t.headers, ['analysis_outlier']),
            rows: t.rows.map((r, i) => [...r, fit.flags[i]]),
            stats: [
                ['Flagged', String(fit.flags.filter(Boolean).length)],
                ['Lower fence', fmt(fit.lower)],
                ['Upper fence', fmt(fit.upper)],
                ['Median', fmt(fit.median)],
            ],
            points: values.map((v, i) => [i + 1, v, fit.flags[i]]),
            box: [
                Math.min(...values.filter((_, i) => !iqr.flags[i])),
                fit.q1,
                fit.median,
                fit.q3,
                Math.max(...values.filter((_, i) => !iqr.flags[i])),
            ],
            js,
            py,
            note: 'Red points fall strictly outside the selected fences. Original rows are retained. A flag is a reason to investigate, not an instruction to delete.',
        };
    } else if (kind === 'polynomial') {
        if (a === b) throw new Error('Select different X and Y columns.');
        const x = column(t, a),
            y = column(t, b),
            fit = polynomial(x, y, degree);
        const min = Math.min(...x),
            max = Math.max(...x),
            line = Array.from({ length: 101 }, (_, i) => {
                const v = min + ((max - min) * i) / 100;
                return [v, fit.predict(v)] as [number, number];
            });
        output = {
            headers: resultHeaders(t.headers, ['analysis_prediction', 'analysis_residual']),
            rows: t.rows.map((r, i) => [...r, fit.predictions[i], fit.residuals[i]]),
            stats: [
                ['Full-data R²', fmt(fit.r2)],
                ['Full-data RMSE', fmt(fit.rmse)],
                ['Holdout RMSE', fmt(fit.holdoutRmse)],
                ['Degree', String(degree)],
            ],
            points: x.map((v, i) => [v, y[i], false]),
            line,
            extra: x.map((v, i) => [v, fit.residuals[i], false]),
            model: { center: fit.center, scale: fit.scale, coefficients: fit.coefficients },
            note: `Full-data equation: y = ${fit.coefficients.map((v, i) => `${fmt(v)}${i ? ` × z^${i}` : ''}`).join(' + ')}; z = (x − ${fmt(fit.center)}) / ${fmt(fit.scale)}. Validation uses rows 1, 6, 11… as holdout and fits scaling on training rows only. This split is not suitable for time-series forecasting.`,
            js: `import { Linear } from '@kanaries/ml';\nconst x=${JSON.stringify(x)}, y=${JSON.stringify(y)}, degree=${degree};\nfunction fit(x,y) {\n const center=x.reduce((a,b)=>a+b,0)/x.length;\n const scale=Math.sqrt(x.reduce((s,v)=>s+(v-center)**2,0)/x.length);\n const features=v=>Array.from({length:degree},(_,j)=>((v-center)/scale)**(j+1));\n const model=new Linear.LinearRegression({}); model.fit(x.map(features),y);\n return v=>model.predict([features(v)])[0];\n}\nconst predict=fit(x,y); console.log(x.map(predict));\nconst train=x.map((_,i)=>i).filter(i=>i%5!==0), test=x.map((_,i)=>i).filter(i=>i%5===0);\nconst validate=fit(train.map(i=>x[i]),train.map(i=>y[i]));\nconsole.log('Holdout RMSE',Math.sqrt(test.reduce((s,i)=>s+(y[i]-validate(x[i]))**2,0)/test.length));`,
            py: `import numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler, PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\nx=np.array(${JSON.stringify(x)}).reshape(-1,1)\ny=np.array(${JSON.stringify(y)})\ndef model(): return make_pipeline(StandardScaler(),PolynomialFeatures(${degree},include_bias=False),LinearRegression())\nprint(model().fit(x,y).predict(x))\ntest=np.arange(len(x))%5==0\npred=model().fit(x[~test],y[~test]).predict(x[test])\nprint('Holdout RMSE',np.sqrt(np.mean((y[test]-pred)**2)))`,
        };
    } else {
        let ids: string[], x: number[][], features: string[];
        if (mode === 'rfm') {
            if (new Set([a, b, c]).size !== 3) throw new Error('Choose distinct ID, date and amount columns.');
            const prepared = rfm(t, a, b, c, reference);
            ids = prepared.ids;
            x = prepared.x;
            features = ['Recency (days)', 'Frequency (transaction rows)', 'Monetary (total)'];
        } else {
            ids = t.rows.map((r) => r[a]);
            features = t.headers.filter((_, i) => i !== a);
            x = t.rows.map((r, i) =>
                r.filter((_, j) => j !== a).map((v, j) => numeric(v, `Row ${i + 2}, feature ${j + 1}`)),
            );
        }
        const fit = segment(ids, x, k);
        output = {
            headers: resultHeaders(resultHeaders(['customer_id'], features), ['cluster']),
            rows: x.map((r, i) => [ids[i], ...r, fit.labels[i]]),
            stats: [
                ['Customers', String(x.length)],
                ['Groups', String(k)],
                ['Features', String(features.length)],
                ['Standardized inertia', fmt(fit.inertia)],
            ],
            points: x.map((r, i) => [r[0], fit.labels[i], false]),
            groups: fit.groups,
            features,
            note: `Features are standardized with population standard deviation. Group IDs have no ranking. ${mode === 'rfm' ? `RFM reference: ${reference} UTC. Frequency counts CSV transaction rows; deduplicate orders and use one currency before importing. ` : 'Every column except the customer ID is used as a numeric feature. '}The chart shows only the first feature against assigned group; consult all group means below. Python and JavaScript may produce different cluster labels or local minima despite the same seed.`,
            js: `import { Clusters } from '@kanaries/ml';\n// Prepared feature rows; see CSV export for customer IDs and RFM aggregation.\nconst x=${JSON.stringify(x)};\nconst centers=x[0].map((_,j)=>x.reduce((s,r)=>s+r[j],0)/x.length);\nconst scales=centers.map((c,j)=>Math.sqrt(x.reduce((s,r)=>s+(r[j]-c)**2,0)/x.length)||1);\nconst z=x.map(r=>r.map((v,j)=>(v-centers[j])/scales[j]));\nconst model=new Clusters.KMeans({n_clusters:${k},random_state:42,n_init:5,max_iter:60});\nconsole.log(model.fitPredict(z));`,
            py: `import numpy as np\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.cluster import KMeans\n# Prepared features, in the same order as the exported customer rows.\nx=np.array(${JSON.stringify(x)},dtype=float)\nz=StandardScaler().fit_transform(x)\nprint(KMeans(n_clusters=${k},random_state=42,n_init=5,max_iter=60).fit_predict(z))`,
        };
    }
    return output;
}
