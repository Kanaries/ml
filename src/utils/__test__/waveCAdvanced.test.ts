import fs from 'fs';
import path from 'path';
import { getRegisteredEstimators, loadModel } from '../../base';
import { LinearRegression } from '../../linear';
import { partialDependence, permutationImportance } from '../inspection';
import { MultiLabelBinarizer } from '../multiLabelBinarizer';
import { SplineTransformer, TargetEncoder } from '../preprocessingAdvanced';

function expectMatrixClose(actual: number[][], expected: number[][], digits = 10): void {
    expect(actual).toHaveLength(expected.length);
    actual.forEach((row, i) => row.forEach((value, j) => expect(value).toBeCloseTo(expected[i][j], digits)));
}
const waveC = JSON.parse(fs.readFileSync(path.join(__dirname, '../../../test_data/wave_c.json'), 'utf8'));

describe('Wave C advanced preprocessing and inspection', () => {
    test('SplineTransformer matches sklearn B-spline fixture and round-trips serialization', () => {
        const fixture = waveC.advanced.spline;
        const spline = new SplineTransformer({ nKnots: 3, degree: 2, includeBias: true });
        spline.fit(fixture.X);
        const expected = fixture.transform, queries = fixture.query;
        expectMatrixClose(spline.transform(queries), expected);
        const revived = loadModel(JSON.stringify(spline)) as SplineTransformer;
        expectMatrixClose(revived.transform(queries), expected);
        const noBias = new SplineTransformer({ nKnots: 3, degree: 2, includeBias: false });
        expect(noBias.fitTransform([[0], [1], [2], [3]])[0]).toHaveLength(3);
        const continued = new SplineTransformer({ nKnots: 3, degree: 2, extrapolation: 'continue' }); continued.fit(fixture.X); expectMatrixClose(continued.transform(fixture.far_query), fixture.continue);
        const linear = new SplineTransformer({ nKnots: 3, degree: 2, extrapolation: 'linear' }); linear.fit(fixture.X); expectMatrixClose(linear.transform(fixture.far_query), fixture.linear);
        const periodic = new SplineTransformer({ nKnots: 5, degree: 3, extrapolation: 'periodic' }); periodic.fit(fixture.X); expectMatrixClose(periodic.transform(fixture.far_query), fixture.periodic); expect(periodic.transform([[0]])[0]).toHaveLength(4);
        const explicit = new SplineTransformer({ knots: fixture.explicit_knots, degree: 2 }); explicit.fit(fixture.explicit_X); expectMatrixClose(explicit.transform(fixture.explicit_X), fixture.explicit);
        expect(() => new SplineTransformer({ knots: [[0], [1], [2]], degree: 3, extrapolation: 'periodic' }).fit([[0], [1], [2]])).toThrow(/more knots than degree/);
    });

    test('TargetEncoder matches sklearn fixed-smoothing and cross-fit fixtures', () => {
        const fixture = waveC.advanced.target_encoder, X = fixture.X, y = fixture.y;
        const encoder = new TargetEncoder({ smooth: 2, cv: 3, shuffle: false, targetType: 'continuous' });
        expectMatrixClose(encoder.fitTransform(X, y), fixture.cross_fit);
        expectMatrixClose(encoder.transform(fixture.query), fixture.transform);
        const revived = loadModel(JSON.stringify(encoder)) as TargetEncoder;
        expectMatrixClose(revived.transform([['a'], ['unknown']]), [[6.15], [11.25]]);
        const automatic = new TargetEncoder({ smooth: 'auto', cv: 3, shuffle: false, targetType: 'continuous' }); automatic.fit(X, y); expectMatrixClose(automatic.transform(fixture.query), fixture.auto);
        const binary = new TargetEncoder({ smooth: 2, cv: 2, shuffle: false, targetType: 'binary' }); expectMatrixClose(binary.fitTransform(fixture.binary_X, fixture.binary_y), fixture.binary_cross_fit); expectMatrixClose(binary.transform(fixture.query), fixture.binary_transform);
        const multiclass = new TargetEncoder({ smooth: 2, cv: 2, shuffle: false, targetType: 'multiclass' }); expectMatrixClose(multiclass.fitTransform(fixture.binary_X, fixture.multiclass_y), fixture.multiclass_cross_fit); expectMatrixClose(multiclass.transform(fixture.query), fixture.multiclass_transform);
        const imbalanced = new TargetEncoder({ smooth: 2, cv: 3, shuffle: false, targetType: 'binary' }); expectMatrixClose(imbalanced.fitTransform(fixture.imbalanced_X, fixture.imbalanced_y), fixture.imbalanced_cross_fit);
        const firstOrder = new TargetEncoder({ smooth: 2, cv: 3, shuffle: false, targetType: 'multiclass' }); expectMatrixClose(firstOrder.fitTransform(fixture.first_order_X, fixture.first_order_y), fixture.first_order_cross_fit);
        const singleClass = new TargetEncoder({ smooth: 2, cv: 3, shuffle: false, targetType: 'auto' }); expectMatrixClose(singleClass.fitTransform(X, fixture.single_class_y), fixture.single_class_cross_fit); expect(singleClass.transform(fixture.query)).toEqual([[0], [0], [0], [0]]);
        const mixedCase = new TargetEncoder({ smooth: 0, targetType: 'binary' }); mixedCase.fit([['a'], ['b']], ['YES', 'no']); expect(mixedCase.transform([['a'], ['b']])).toEqual([[0], [1]]);
    });

    test('MultiLabelBinarizer matches sklearn dense and sparse fixtures', () => {
        const fixture = waveC.advanced.multi_label_binarizer, y = fixture.y;
        const dense = new MultiLabelBinarizer();
        expect(dense.fitTransform(y)).toEqual(fixture.transform); expect(dense.classes).toEqual(fixture.classes);
        expect(dense.inverseTransform([[0, 1, 1], [1, 0, 0]])).toEqual([['sci-fi', 'thriller'], ['comedy']]);
        const sparse = new MultiLabelBinarizer({ sparseOutput: true });
        expect((sparse.fitTransform(y) as any).toDense()).toEqual(fixture.transform);
        const numeric = new MultiLabelBinarizer(); numeric.fit([[10, 2, 1]]); expect(numeric.classes).toEqual([1, 2, 10]);
        const ordered = new MultiLabelBinarizer({ classes: ['z', 'a'] }); ordered.fit([['a']]); expect(ordered.classes).toEqual(['z', 'a']); expect(ordered.transform([['a', 'unseen']])).toEqual([[0, 1]]);
        const mixedCase = new MultiLabelBinarizer(); mixedCase.fit([['no', 'YES']]); expect(mixedCase.classes).toEqual(['YES', 'no']);
    });

    test('MultiLabelBinarizer follows estimator conformance', () => {
        expect(getRegisteredEstimators().get('MultiLabelBinarizer')).toBe(MultiLabelBinarizer);
        const model = new MultiLabelBinarizer({ sparseOutput: false });
        expect(model.clone().getParams()).toEqual(model.getParams());
        expect(() => model.setParams({ missing: true })).toThrow(/Invalid parameter/);
        model.fit([['a'], ['b']]);
        const revived = loadModel(JSON.stringify(model)) as MultiLabelBinarizer;
        expect(revived.transform([['a'], ['b']])).toEqual([[1, 0], [0, 1]]);
        model.setParams({}); expect(() => model.transform([['a']])).toThrow(/not fitted/);
    });

    test('inspection utilities expose deterministic permutation and partial-dependence results', () => {
        const fixture = waveC.advanced.inspection, X = fixture.X, y = fixture.y;
        const model = new LinearRegression(); model.fit(X, y);
        const importance = permutationImportance(model, X, y, { nRepeats: 4, randomState: 7 });
        expect(importance.importances).toHaveLength(2);
        importance.importancesMean.forEach((value, i) => expect(value).toBeCloseTo(fixture.permutation_mean[i], 12));
        const pd = partialDependence(model, X, [0], { gridResolution: 4, percentiles: [0, 1] });
        expectMatrixClose(pd.gridValues, fixture.partial_grid);
        expectMatrixClose(pd.average as number[][], fixture.partial_average);
    });
});
