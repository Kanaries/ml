import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';
const source = await readFile(new URL('../../components/tools/workflowMath.ts', import.meta.url), 'utf8');
const output = ts
    .transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } })
    .outputText.replace("'@kanaries/ml'", JSON.stringify(import.meta.resolve('@kanaries/ml')));
const m = await import(`data:text/javascript;base64,${Buffer.from(output).toString('base64')}`);
const near = (a, b, tol = 1e-8) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
test('CSV preserves quoted commas, escaped quotes, line breaks and text IDs', () => {
    const t = m.parseCsv('id,note,value\r\n001,"one, two",4\r\n002,"a ""quote""\nand line",5\r\n');
    assert.deepEqual(t.rows, [
        ['001', 'one, two', '4'],
        ['002', 'a "quote"\nand line', '5'],
    ]);
    assert.deepEqual(m.column(t, 2), [4, 5]);
    assert.throws(() => m.parseCsv('a,a\n1,2'), /unique/);
    assert.throws(() => m.parseCsv('a,b\n1'), /Row 2/);
    assert.throws(() => m.parseCsv('a\n"bad'), /Unclosed/);
    assert.throws(() => m.numeric('', 'row'), /missing/);
    assert.throws(() => m.numeric('Infinity', 'row'), /finite/);
    assert.throws(() => m.numeric('0x10', 'row'), /finite/);
    assert.equal(m.csv([['=SUM(A1)', '001', 3]]), '"\'=SUM(A1)","001","3"');
});
test('IQR matches linear-interpolated NumPy quartiles; Z-score masking is explicit', () => {
    const values = [10, 11, 12, 12, 13, 14, 15, 16, 17, 80];
    const r = m.outliers(values, 'iqr');
    near(r.q1, 12);
    near(r.q3, 15.75);
    near(r.lower, 6.375);
    near(r.upper, 21.375);
    assert.deepEqual(r.flags, [false, false, false, false, false, false, false, false, false, true]);
    assert.equal(m.outliers(values, 'z').flags.filter(Boolean).length, 0);
    assert.equal(m.outliers(values, 'modified').flags.filter(Boolean).length, 1);
    assert.throws(() => m.outliers([1, 1, 1, 1, 8], 'modified'), /MAD is zero/);
    assert.throws(() => m.outliers([1, 1, 1, 1], 'z'), /undefined/);
    assert.deepEqual(m.outliers([1, 1, 1, 1], 'iqr').flags, [false, false, false, false]);
});
test('polynomial recovers known coefficients and validates on training-only scaling', () => {
    const x = Array.from({ length: 15 }, (_, i) => i),
        y = x.map((v) => 2 + 2 * v * v),
        r = m.polynomial(x, y, 2);
    near(r.r2, 1);
    near(r.rmse, 0);
    near(r.holdoutRmse, 0);
    near(r.predict(3.5), 26.5);
    assert.deepEqual(r.test, [0, 5, 10]);
    assert.equal(r.train.length, 12);
    const noisy = [...y];
    noisy[0] += 20;
    const r2 = m.polynomial(x, noisy, 2);
    near(r2.holdoutRmse, 20 / Math.sqrt(3));
    assert.throws(() => m.polynomial(Array(10).fill(1), Array(10).fill(2), 2), /distinct/);
    assert.equal(m.polynomial(x, Array(15).fill(4), 2).r2, null);
    const shifted = m.polynomial(
        x.map((v) => v + 1000000),
        y,
        2,
    );
    near(shifted.predict(1000003.5), 26.5, 1e-7);
});
test('RFM uses validated UTC dates, textual customer IDs and transaction-row counts', () => {
    const table = m.parseCsv('id,date,amount\n001,2026-08-28,120\n001,2026-08-30,80\n003,2026-06-01,20');
    assert.deepEqual(m.rfm(table, 0, 1, 2, '2026-09-01'), {
        ids: ['001', '003'],
        x: [
            [2, 2, 200],
            [92, 1, 20],
        ],
    });
    assert.throws(() => m.utcDay('2026-02-30'), /Invalid calendar/);
    assert.throws(() => m.rfm(table, 0, 1, 2, '2026-08-01'), /after/);
    assert.throws(() => m.rfm(m.parseCsv('id,date,amount\na,2026-08-01,-1'), 0, 1, 2, '2026-09-01'), /refunds/);
});
test('clustering is deterministic, scale invariant and keeps obvious groups together', () => {
    const ids = ['a', 'b', 'c', 'd'],
        x = [
            [0, 1],
            [0.1, 1.1],
            [10, 11],
            [10.1, 11.1],
        ];
    const r = m.segment(ids, x, 2),
        r2 = m.segment(
            ids,
            x.map((v) => [v[0] * 1000, v[1]]),
            2,
        );
    assert.deepEqual(r.labels, m.segment(ids, x, 2).labels);
    assert.deepEqual(r.labels, r2.labels);
    assert.equal(r.labels[0], r.labels[1]);
    assert.equal(r.labels[2], r.labels[3]);
    assert.notEqual(r.labels[0], r.labels[2]);
    assert.equal(
        r.groups.reduce((s, g) => s + g.count, 0),
        4,
    );
    near(r.centers[0], 5.05);
    assert.throws(() => m.segment(['a', 'a'], [[1], [2]], 2), /unique/);
    assert.throws(() => m.segment(ids, [[1], [1], [1], [1]], 2), /distinct/);
});

test('number lists and result headers preserve identity without ambiguous fields', () => {
    assert.deepEqual(m.parseNumberList('1, 2; 3\n4').rows, [
        ['1', '1'],
        ['2', '2'],
        ['3', '3'],
        ['4', '4'],
    ]);
    assert.throws(() => m.numeric('1e150', 'row'), /1e100/);
    assert.throws(() => m.parseNumberList('1,,2'), /missing/);
    assert.equal(m.csv([['-12', '-SUM(A1)']]), '"-12","\'-SUM(A1)"');
    assert.throws(() => m.column(m.parseCsv('a,b\n,'), 1), /missing/);
    assert.deepEqual(m.resultHeaders(['analysis_outlier', 'analysis_outlier_2'], ['analysis_outlier']), [
        'analysis_outlier',
        'analysis_outlier_2',
        'analysis_outlier_3',
    ]);
});

const resultSource = await readFile(new URL('../../components/tools/workflowResult.ts', import.meta.url), 'utf8');
const mathUrl = `data:text/javascript;base64,${Buffer.from(output).toString('base64')}`;
const resultOutput = ts
    .transpileModule(resultSource, {
        compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
    })
    .outputText.replace("'./workflowMath'", JSON.stringify(mathUrl));
const { calculateWorkflow } = await import(
    `data:text/javascript;base64,${Buffer.from(resultOutput).toString('base64')}`
);
const settings = {
    a: 0,
    b: 1,
    c: 2,
    method: 'iqr',
    inputMode: 'csv',
    degree: 2,
    k: 3,
    mode: 'features',
    reference: '2026-09-01',
};
async function runCode(js) {
    const captured = [];
    // Run exactly the generated JavaScript with only module resolution and logging adapted.
    const code = js
        .replace("'@kanaries/ml'", JSON.stringify(import.meta.resolve('@kanaries/ml')))
        .replaceAll('console.log(', 'capture(');
    const module = await import(
        `data:text/javascript;base64,${Buffer.from('const results=[]; const capture=(...args)=>results.push(args);\n' + code + '\nexport default results;').toString('base64')}`
    );
    captured.push(...module.default);
    return captured;
}
test('all generated JavaScript examples reproduce exported sample rows', async () => {
    for (const kind of ['outlier', 'polynomial', 'customer']) {
        const table = m.parseCsv(await readFile(new URL(`../../public/samples/${kind}.csv`, import.meta.url), 'utf8'));
        for (const method of kind === 'outlier' ? ['iqr', 'z', 'modified'] : ['iqr']) {
            const result = calculateWorkflow(table, kind, { ...settings, a: kind === 'outlier' ? 1 : 0, method });
            const logs = await runCode(result.js),
                values = logs[0][0];
            if (kind === 'outlier')
                assert.deepEqual(
                    values.map((v) => v.outlier),
                    result.rows.map((r) => r.at(-1)),
                );
            if (kind === 'polynomial') {
                values.forEach((v, i) => near(v, result.rows[i].at(-2)));
                near(logs[1][1], Number(result.stats[2][1]));
            }
            if (kind === 'customer')
                assert.deepEqual(
                    values,
                    result.rows.map((r) => r.at(-1)),
                );
            assert.equal(result.rows.length, table.rows.length);
            assert.ok(result.points.every((p) => p.slice(0, 2).every(Number.isFinite)));
        }
    }
});
test('transaction workflow exports aggregated customers and reproducible cluster assignments', async () => {
    const table = m.parseCsv(
        'id,date,amount\n001,2026-08-28,120\n001,2026-08-30,80\n002,2026-08-29,190\n003,2026-06-01,20\n004,2026-06-15,25\n005,2026-08-10,60\n005,2026-08-15,50\n006,2026-08-12,90',
    );
    const result = calculateWorkflow(table, 'customer', { ...settings, mode: 'rfm' });
    assert.equal(result.rows.length, 6);
    assert.deepEqual(result.rows[0].slice(0, 4), ['001', 2, 2, 200]);
    assert.deepEqual(
        (await runCode(result.js))[0][0],
        result.rows.map((r) => r.at(-1)),
    );
});
