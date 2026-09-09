import { Linear, Clusters } from '@kanaries/ml';

export type Table = { headers: string[]; rows: string[][] };
export function parseCsv(text: string): Table {
    if (text.length > 1_000_000) throw new Error('Use a CSV smaller than 1 MB.');
    const source = text.replace(/^\uFEFF/, '').trim();
    if (!source) throw new Error('Paste a CSV with a header and at least one data row.');
    const delimiter = source.split(/\r?\n/, 1)[0].includes('\t') ? '\t' : ',';
    const records: string[][] = [];
    let row: string[] = [],
        cell = '',
        quoted = false,
        closed = false;
    for (let i = 0; i < source.length; i++) {
        const c = source[i];
        if (quoted) {
            if (c === '"' && source[i + 1] === '"') {
                cell += '"';
                i++;
            } else if (c === '"') {
                quoted = false;
                closed = true;
            } else cell += c;
        } else if (c === '"' && !cell && !closed) quoted = true;
        else if (c === delimiter || c === '\n' || c === '\r') {
            row.push(cell.trim());
            cell = '';
            closed = false;
            if (c !== delimiter) {
                if (row.some(Boolean) || row.length > 1) records.push(row);
                row = [];
                if (c === '\r' && source[i + 1] === '\n') i++;
            }
        } else {
            if (c === '"' || (closed && c.trim()))
                throw new Error('Malformed CSV quotes. Use double quotes around fields containing commas.');
            cell += c;
        }
    }
    if (quoted) throw new Error('Unclosed quoted CSV field.');
    row.push(cell.trim());
    if (row.some(Boolean) || row.length > 1) records.push(row);
    const headers = records.shift() ?? [];
    if (headers.some((h) => !h) || new Set(headers).size !== headers.length)
        throw new Error('Headers must be nonempty and unique.');
    if (headers.length > 30 || records.length > 2000) throw new Error('Limit: 30 columns and 2,000 data rows.');
    if (!records.length) throw new Error('Include at least one data row after the header.');
    records.forEach((r, i) => {
        if (r.length !== headers.length)
            throw new Error(`Row ${i + 2}: expected ${headers.length} fields, found ${r.length}.`);
    });
    return { headers, rows: records };
}
export function numeric(value: string, label: string): number {
    if (
        !value.trim() ||
        !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim()) ||
        !Number.isFinite(Number(value)) ||
        Math.abs(Number(value)) > 1e100
    )
        throw new Error(`${label}: enter a finite number with magnitude at most 1e100; missing values are not zero.`);
    return Number(value);
}
export function column(table: Table, index: number): number[] {
    return table.rows.map((r, i) =>
        numeric(r[index] ?? '', `Row ${i + 2}, ${table.headers[index] ?? 'selected column'}`),
    );
}
export function csv(rows: (string | number | boolean | null)[][]): string {
    return rows
        .map((row) =>
            row
                .map((value) => {
                    let s = value == null ? '' : String(value);
                    if (typeof value === 'string' && /^[=+@\-\t\r]/.test(s) && !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(s)) s = `'${s}`;
                    return `"${s.replace(/"/g, '""')}"`;
                })
                .join(','),
        )
        .join('\n');
}
export function resultHeaders(original: string[], additions: string[]): string[] {
    const used = new Set(original);
    return [
        ...original,
        ...additions.map((name) => {
            let candidate = name,
                suffix = 2;
            while (used.has(candidate)) candidate = `${name}_${suffix++}`;
            used.add(candidate);
            return candidate;
        }),
    ];
}
export function parseNumberList(text: string): Table {
    if (/(?:^|[,;])\s*(?:[,;]|$)/.test(text.trim())) throw new Error('Number list contains a missing value.');
    const values = text.trim().split(/[\s,;]+/);
    if (values.length > 2000 || text.length > 1_000_000) throw new Error('Limit: 2,000 values and 1 MB.');
    values.forEach((v, i) => numeric(v, `Value ${i + 1}`));
    return { headers: ['row', 'value'], rows: values.map((v, i) => [String(i + 1), v]) };
}
export const mean = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length;
export function quantile(x: number[], p: number): number {
    const a = [...x].sort((u, v) => u - v),
        at = (a.length - 1) * p,
        lo = Math.floor(at);
    return a[lo] + (a[Math.ceil(at)] - a[lo]) * (at - lo);
}
export function outliers(values: number[], method: string) {
    if (values.length < 4) throw new Error('Use at least four observations.');
    const q1 = quantile(values, 0.25),
        q3 = quantile(values, 0.75),
        median = quantile(values, 0.5);
    const avg = mean(values),
        sd = Math.sqrt(mean(values.map((v) => (v - avg) ** 2)));
    const mad = quantile(
        values.map((v) => Math.abs(v - median)),
        0.5,
    );
    if (method === 'modified' && mad === 0)
        throw new Error('MAD is zero: modified Z-scores are undefined. Choose IQR instead.');
    if (method === 'z' && sd === 0)
        throw new Error('Standard deviation is zero: Z-scores are undefined. All values are equal.');
    const lower =
        method === 'iqr'
            ? q1 - 1.5 * (q3 - q1)
            : method === 'z'
              ? avg - 3 * sd
              : median - (3.5 * mad) / 0.6744897501960817;
    const upper =
        method === 'iqr'
            ? q3 + 1.5 * (q3 - q1)
            : method === 'z'
              ? avg + 3 * sd
              : median + (3.5 * mad) / 0.6744897501960817;
    return { q1, q3, median, lower, upper, flags: values.map((v) => v < lower || v > upper) };
}
function fitPolynomial(x: number[], y: number[], degree: number) {
    if (new Set(x).size <= degree)
        throw new Error(`Degree ${degree} needs at least ${degree + 1} distinct training X values.`);
    const center = mean(x),
        scale = Math.sqrt(mean(x.map((v) => (v - center) ** 2)));
    const features = (v: number) => Array.from({ length: degree }, (_, j) => ((v - center) / scale) ** (j + 1));
    const model = new Linear.LinearRegression({});
    model.fit(x.map(features), y);
    const coefficients = [model.predict([Array(degree).fill(0)])[0], ...model.coef];
    if (coefficients.some((v) => !Number.isFinite(v)))
        throw new Error('The fit is numerically unstable. Reduce the degree or check the data.');
    return { center, scale, coefficients, predict: (v: number) => model.predict([features(v)])[0] };
}
export function polynomial(x: number[], y: number[], degree: number) {
    if (!Number.isInteger(degree) || degree < 1 || degree > 6) throw new Error('Degree must be between 1 and 6.');
    if (x.length < 10) throw new Error('Use at least 10 rows for the fixed 80/20 holdout comparison.');
    const train = x.map((_, i) => i).filter((i) => i % 5 !== 0),
        test = x.map((_, i) => i).filter((i) => i % 5 === 0);
    const validation = fitPolynomial(
        train.map((i) => x[i]),
        train.map((i) => y[i]),
        degree,
    );
    const model = fitPolynomial(x, y, degree),
        predictions = x.map(model.predict);
    const sse = y.reduce((s, v, i) => s + (v - predictions[i]) ** 2, 0),
        tss = y.reduce((s, v) => s + (v - mean(y)) ** 2, 0);
    return {
        ...model,
        predictions,
        residuals: y.map((v, i) => v - predictions[i]),
        r2: tss === 0 ? null : 1 - sse / tss,
        rmse: Math.sqrt(sse / x.length),
        holdoutRmse: Math.sqrt(mean(test.map((i) => (y[i] - validation.predict(x[i])) ** 2))),
        train,
        test,
    };
}
export function utcDay(value: string): number {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error('Dates must use YYYY-MM-DD.');
    const stamp = Date.parse(`${value}T00:00:00Z`);
    if (!Number.isFinite(stamp) || new Date(stamp).toISOString().slice(0, 10) !== value)
        throw new Error(`Invalid calendar date: ${value}.`);
    return stamp / 86400000;
}
export function rfm(table: Table, id: number, date: number, amount: number, reference: string) {
    const day = utcDay(reference),
        customers = new Map<string, { last: number; frequency: number; monetary: number }>();
    table.rows.forEach((r, i) => {
        if (!r[id]) throw new Error(`Row ${i + 2}: customer ID is missing.`);
        const when = utcDay(r[date]),
            money = numeric(r[amount], `Row ${i + 2}, amount`);
        if (when > day) throw new Error(`Row ${i + 2}: transaction is after the reference date.`);
        if (money < 0) throw new Error(`Row ${i + 2}: handle refunds separately; amounts must be nonnegative.`);
        const prior = customers.get(r[id]) ?? { last: when, frequency: 0, monetary: 0 };
        prior.last = Math.max(prior.last, when);
        prior.frequency++;
        prior.monetary += money;
        customers.set(r[id], prior);
    });
    return {
        ids: [...customers.keys()],
        x: [...customers.values()].map((c) => [day - c.last, c.frequency, c.monetary]),
    };
}
export function segment(ids: string[], x: number[][], k: number) {
    if (ids.some((id) => !id) || new Set(ids).size !== ids.length)
        throw new Error('Customer IDs must be nonempty and unique in feature mode.');
    if (x.length > 500) throw new Error('Use at most 500 customers for this browser tool.');
    if (!Number.isInteger(k) || k < 2 || k > 8 || k > new Set(x.map((r) => JSON.stringify(r))).size)
        throw new Error('Choose 2–8 groups, no more than the number of distinct customer profiles.');
    if (!x.length || !x[0].length || x[0].length > 12) throw new Error('Select between 1 and 12 numeric features.');
    const centers = x[0].map((_, j) => mean(x.map((r) => r[j]))),
        scales = centers.map((c, j) => Math.sqrt(mean(x.map((r) => (r[j] - c) ** 2))) || 1);
    const z = x.map((r) => r.map((v, j) => (v - centers[j]) / scales[j]));
    const model = new Clusters.KMeans({ n_clusters: k, random_state: 42, n_init: 5, max_iter: 60 });
    const labels = model.fitPredict(z);
    const groups = Array.from({ length: k }, (_, label) => {
        const members = x.filter((_, i) => labels[i] === label);
        return {
            label,
            count: members.length,
            averages: centers.map((_, j) => (members.length ? mean(members.map((r) => r[j])) : null)),
        };
    });
    return { labels, groups, centers, scales, inertia: model.getInertia() };
}
