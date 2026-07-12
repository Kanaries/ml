/**
 * Tests for the expanded metric set. Every expected value is hand-computed
 * from the sklearn formula and, where noted, matches the documented output of
 * the referenced sklearn call.
 *
 * Argument-order conventions under test (matching the existing module):
 * - label metrics:      (actual = predictions, expected = ground truth)
 * - score/probability:  (expected = ground truth, scores)
 * - regression:         (actual = predictions, expected = ground truth)
 * - clustering labels:  (labelsTrue, labelsPred)
 * - clustering X-based: (X, labels)
 */
import {
    logLoss,
    balancedAccuracyScore,
    matthewsCorrcoef,
    cohenKappaScore,
    hingeLoss,
    brierScoreLoss,
    classificationReport,
    rocAucScore,
    meanAbsoluteError,
    meanAbsolutePercentageError,
    medianAbsoluteError,
    meanSquaredLogError,
    explainedVarianceScore,
    maxError,
    rootMeanSquaredError,
    silhouetteScore,
    silhouetteSamples,
    daviesBouldinScore,
    calinskiHarabaszScore,
    mutualInfoScore,
    normalizedMutualInfoScore,
    adjustedMutualInfoScore,
    homogeneityScore,
    completenessScore,
    vMeasureScore,
    fowlkesMallowsScore,
    randScore,
    adjustedRandScore,
} from '../index';

describe('logLoss', () => {
    test('binary 1-D probabilities match sklearn docs example', () => {
        // sklearn: log_loss(["spam","ham","ham","spam"], [[.1,.9],[.9,.1],[.8,.2],[.35,.65]])
        //        = 0.21616187468057912 (ham=0, spam=1; 1-D input carries P(class 1))
        expect(logLoss([1, 0, 0, 1], [0.9, 0.1, 0.2, 0.65])).toBeCloseTo(0.21616187468057912, 10);
    });

    test('probability-matrix input matches the same value', () => {
        expect(
            logLoss([1, 0, 0, 1], [[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.35, 0.65]]),
        ).toBeCloseTo(0.21616187468057912, 10);
    });

    test('multiclass probability matrix', () => {
        // sklearn: log_loss([0,1,2], [[.8,.1,.1],[.2,.7,.1],[.2,.2,.6]])
        //        = -(ln0.8 + ln0.7 + ln0.6)/3 = 0.3635480396729776
        expect(
            logLoss([0, 1, 2], [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]),
        ).toBeCloseTo(-(Math.log(0.8) + Math.log(0.7) + Math.log(0.6)) / 3, 12);
    });

    test('normalize: false returns the summed loss', () => {
        // sklearn: log_loss(..., normalize=False) = 4 * 0.21616187468057912
        expect(logLoss([1, 0, 0, 1], [0.9, 0.1, 0.2, 0.65], { normalize: false })).toBeCloseTo(
            0.8646474987223165,
            10,
        );
    });

    test('eps clipping bounds extreme probabilities', () => {
        // perfect confident predictions: loss -> -log(1 - eps) ~ 0
        expect(logLoss([0, 1], [0, 1])).toBeCloseTo(0, 10);
        // maximally wrong confident predictions: loss = -log(eps)
        expect(logLoss([1, 0], [0, 1])).toBeCloseTo(-Math.log(Number.EPSILON), 8);
        // explicit legacy eps like sklearn<1.5's default 1e-15
        expect(logLoss([1, 0], [0, 1], { eps: 1e-15 })).toBeCloseTo(-Math.log(1e-15), 8);
    });

    test('single-label expected requires the labels option (sklearn raises too)', () => {
        expect(() => logLoss([1, 1], [0.8, 0.7])).toThrow();
        // sklearn: log_loss([1,1], [0.8,0.7], labels=[0,1]) = -(ln0.8 + ln0.7)/2
        expect(logLoss([1, 1], [0.8, 0.7], { labels: [0, 1] })).toBeCloseTo(
            -(Math.log(0.8) + Math.log(0.7)) / 2,
            12,
        );
    });

    test('rows that do not sum to one throw (sklearn >= 1.5 behaviour)', () => {
        expect(() => logLoss([0, 1], [[0.5, 0.2], [0.3, 0.7]])).toThrow();
    });

    test('validates lengths', () => {
        expect(() => logLoss([0], [0.5, 0.5])).toThrow('same length');
        expect(() => logLoss([], [])).toThrow('non-empty');
    });
});

describe('balancedAccuracyScore', () => {
    test('binary case matches sklearn docs example', () => {
        // sklearn: balanced_accuracy_score(y_true=[0,1,0,0,1,0], y_pred=[0,1,0,0,0,1]) = 0.625
        expect(balancedAccuracyScore([0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0])).toBeCloseTo(0.625, 10);
        // adjusted=True: (0.625 - 0.5) / (1 - 0.5) = 0.25
        expect(
            balancedAccuracyScore([0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0], { adjusted: true }),
        ).toBeCloseTo(0.25, 10);
    });

    test('multiclass macro recall', () => {
        // sklearn: balanced_accuracy_score([0,0,1,1,2,2], [0,1,1,2,2,2])
        // recalls: class0=1/2, class1=1/2, class2=1 -> 2/3; adjusted -> 0.5
        expect(balancedAccuracyScore([0, 1, 1, 2, 2, 2], [0, 0, 1, 1, 2, 2])).toBeCloseTo(2 / 3, 10);
        expect(
            balancedAccuracyScore([0, 1, 1, 2, 2, 2], [0, 0, 1, 1, 2, 2], { adjusted: true }),
        ).toBeCloseTo(0.5, 10);
    });

    test('classes without true samples are dropped like sklearn', () => {
        // sklearn: balanced_accuracy_score([0,0,1], [0,2,1]) = (1/2 + 1) / 2 = 0.75
        expect(balancedAccuracyScore([0, 2, 1], [0, 0, 1])).toBeCloseTo(0.75, 10);
    });
});

describe('matthewsCorrcoef', () => {
    test('binary case matches sklearn docs example', () => {
        // sklearn: matthews_corrcoef(y_true=[1,1,1,-1], y_pred=[1,-1,1,1]) = -1/3
        expect(matthewsCorrcoef([1, -1, 1, 1], [1, 1, 1, -1])).toBeCloseTo(-1 / 3, 10);
    });

    test('perfect binary prediction gives 1', () => {
        // sklearn: matthews_corrcoef([1,0,1], [1,0,1]) = 1.0
        expect(matthewsCorrcoef([1, 0, 1], [1, 0, 1])).toBeCloseTo(1, 10);
    });

    test('multiclass covariance formulation', () => {
        // sklearn: matthews_corrcoef(y_true=[0,0,1,1,2,2], y_pred=[0,0,1,2,1,2])
        // C diag=(2,1,1), t=p=(2,2,2): cov_ytyp=12, cov=24 both -> 0.5
        expect(matthewsCorrcoef([0, 0, 1, 2, 1, 2], [0, 0, 1, 1, 2, 2])).toBeCloseTo(0.5, 10);
    });

    test('returns 0 when the denominator is 0 (sklearn documented edge)', () => {
        // constant predictions: cov_ypyp = 0 -> sklearn returns 0.0
        expect(matthewsCorrcoef([1, 1, 1], [0, 1, 1])).toBe(0);
        expect(matthewsCorrcoef([1, 1], [1, 1])).toBe(0);
    });
});

describe('cohenKappaScore', () => {
    test('unweighted binary agreement', () => {
        // sklearn: cohen_kappa_score([0,1,0,1], [0,1,1,1])
        // po=0.75, pe=(2*1 + 2*3)/16=0.5 -> kappa = 0.5
        expect(cohenKappaScore([0, 1, 0, 1], [0, 1, 1, 1])).toBeCloseTo(0.5, 10);
    });

    test('multiclass with unweighted, linear, and quadratic weights', () => {
        const y1 = [0, 1, 2];
        const y2 = [0, 2, 1];
        // sklearn: cohen_kappa_score([0,1,2], [0,2,1]) = 0
        expect(cohenKappaScore(y1, y2)).toBeCloseTo(0, 10);
        // weights="linear": 1 - 2 / (8/3) = 0.25
        expect(cohenKappaScore(y1, y2, { weights: 'linear' })).toBeCloseTo(0.25, 10);
        // weights="quadratic": 1 - 2 / 4 = 0.5
        expect(cohenKappaScore(y1, y2, { weights: 'quadratic' })).toBeCloseTo(0.5, 10);
    });

    test('is symmetric', () => {
        const y1 = [0, 1, 2, 1, 0, 2];
        const y2 = [0, 2, 2, 1, 0, 1];
        expect(cohenKappaScore(y1, y2)).toBeCloseTo(cohenKappaScore(y2, y1), 12);
    });
});

describe('hingeLoss', () => {
    test('binary hinge loss with signed labels', () => {
        // sklearn: hinge_loss([-1,1,1], [-2.18,2.36,0.09])
        // losses = [0, 0, 0.91] -> 0.91/3 = 0.30333...
        expect(hingeLoss([-1, 1, 1], [-2.18, 2.36, 0.09])).toBeCloseTo(0.91 / 3, 10);
    });

    test('greater label is the positive class for {0,1} targets', () => {
        // sklearn: hinge_loss([0,1], [-0.5,0.5]) = (0.5 + 0.5)/2 = 0.5
        expect(hingeLoss([0, 1], [-0.5, 0.5])).toBeCloseTo(0.5, 10);
        // confidently correct decisions incur zero loss
        expect(hingeLoss([0, 1], [-2, 3])).toBe(0);
    });

    test('throws for non-binary targets (multiclass case)', () => {
        expect(() => hingeLoss([0, 1, 2], [1, 2, 3])).toThrow();
        expect(() => hingeLoss([1, 1], [1, 1])).toThrow();
    });
});

describe('brierScoreLoss', () => {
    test('matches sklearn docs example', () => {
        // sklearn: brier_score_loss([0,1,1,0], [0.1,0.9,0.8,0.3]) = 0.0375
        expect(brierScoreLoss([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3])).toBeCloseTo(0.0375, 10);
    });

    test('explicit positiveLabel handling', () => {
        // sklearn: brier_score_loss([2,2,3,3], [0.1,0.2,0.8,0.9], pos_label=3) = 0.025
        expect(
            brierScoreLoss([2, 2, 3, 3], [0.1, 0.2, 0.8, 0.9], { positiveLabel: 3 }),
        ).toBeCloseTo(0.025, 10);
        // pos_label cannot be inferred outside {-1, 0, 1}: sklearn raises
        expect(() => brierScoreLoss([2, 2, 3, 3], [0.1, 0.2, 0.8, 0.9])).toThrow();
    });

    test('default positive label 1 for {-1, 1} targets', () => {
        // sklearn: brier_score_loss([-1,1], [0.4,0.6]) = (0.4^2 + 0.4^2)/2 = 0.16
        expect(brierScoreLoss([-1, 1], [0.4, 0.6])).toBeCloseTo(0.16, 10);
    });

    test('rejects invalid inputs', () => {
        expect(() => brierScoreLoss([0, 1], [0.5, 1.2])).toThrow();
        expect(() => brierScoreLoss([0, 1, 2], [0.1, 0.2, 0.3])).toThrow();
    });
});

describe('classificationReport', () => {
    test('multiclass report matches sklearn classification_report(output_dict=True)', () => {
        // sklearn: classification_report(y_true=[0,2,2,1,1,0], y_pred=[0,1,2,2,1,0], output_dict=True)
        const report = classificationReport([0, 1, 2, 2, 1, 0], [0, 2, 2, 1, 1, 0]);
        expect(report.perClass['0']).toEqual({ precision: 1, recall: 1, f1Score: 1, support: 2 });
        expect(report.perClass['1']).toEqual({ precision: 0.5, recall: 0.5, f1Score: 0.5, support: 2 });
        expect(report.perClass['2']).toEqual({ precision: 0.5, recall: 0.5, f1Score: 0.5, support: 2 });
        expect(report.accuracy).toBeCloseTo(2 / 3, 10);
        expect(report.macroAvg.precision).toBeCloseTo(2 / 3, 10);
        expect(report.macroAvg.recall).toBeCloseTo(2 / 3, 10);
        expect(report.macroAvg.f1Score).toBeCloseTo(2 / 3, 10);
        expect(report.macroAvg.support).toBe(6);
        expect(report.weightedAvg.precision).toBeCloseTo(2 / 3, 10);
        expect(report.weightedAvg.support).toBe(6);
    });

    test('weighted average ignores zero-support classes', () => {
        // sklearn: classification_report(y_true=[1,1,1], y_pred=[1,1,0], output_dict=True)
        // class 0: P=0, R=0, F1=0, support=0; class 1: P=1, R=2/3, F1=0.8, support=3
        const report = classificationReport([1, 1, 0], [1, 1, 1]);
        expect(report.perClass['0']).toEqual({ precision: 0, recall: 0, f1Score: 0, support: 0 });
        expect(report.perClass['1'].precision).toBeCloseTo(1, 10);
        expect(report.perClass['1'].recall).toBeCloseTo(2 / 3, 10);
        expect(report.perClass['1'].f1Score).toBeCloseTo(0.8, 10);
        expect(report.perClass['1'].support).toBe(3);
        expect(report.accuracy).toBeCloseTo(2 / 3, 10);
        expect(report.macroAvg.precision).toBeCloseTo(0.5, 10);
        expect(report.macroAvg.recall).toBeCloseTo(1 / 3, 10);
        expect(report.macroAvg.f1Score).toBeCloseTo(0.4, 10);
        expect(report.weightedAvg.precision).toBeCloseTo(1, 10);
        expect(report.weightedAvg.recall).toBeCloseTo(2 / 3, 10);
        expect(report.weightedAvg.f1Score).toBeCloseTo(0.8, 10);
    });
});

describe('rocAucScore multiclass extension', () => {
    // columns follow sorted labels [0, 1, 2]; rows sum to 1
    const yTrue = [0, 0, 1, 2];
    const proba = [
        [0.5, 0.3, 0.2],
        [0.3, 0.5, 0.2],
        [0.2, 0.2, 0.6],
        [0.1, 0.1, 0.8],
    ];

    test('binary signature stays backward compatible', () => {
        // sklearn: roc_auc_score([0,0,1,1], [0.1,0.4,0.35,0.8]) = 0.75
        expect(rocAucScore([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])).toBeCloseTo(0.75, 10);
        // positional positive label still works
        expect(rocAucScore([1, 1, 2, 2], [0.1, 0.4, 0.35, 0.8], 2)).toBeCloseTo(0.75, 10);
    });

    test("ovr macro/weighted (sklearn multi_class='ovr')", () => {
        // per-class one-vs-rest AUCs:
        //   class0: y=[1,1,0,0], scores=[0.5,0.3,0.2,0.1] -> 1
        //   class1: y=[0,0,1,0], scores=[0.3,0.5,0.2,0.1] -> 1/3
        //   class2: y=[0,0,0,1], scores=[0.2,0.2,0.6,0.8] -> 1
        // sklearn: roc_auc_score(yTrue, proba, multi_class='ovr') = 7/9
        expect(rocAucScore(yTrue, proba, { multiClass: 'ovr' })).toBeCloseTo(7 / 9, 10);
        // sklearn: roc_auc_score(..., multi_class='ovr', average='weighted')
        //        = 0.5*1 + 0.25*(1/3) + 0.25*1 = 5/6
        expect(rocAucScore(yTrue, proba, { multiClass: 'ovr', average: 'weighted' })).toBeCloseTo(5 / 6, 10);
    });

    test("ovo macro/weighted (sklearn multi_class='ovo')", () => {
        // pair scores: (0,1)=0.5, (0,2)=1, (1,2)=1
        // sklearn: roc_auc_score(yTrue, proba, multi_class='ovo') = 5/6
        expect(rocAucScore(yTrue, proba, { multiClass: 'ovo' })).toBeCloseTo(5 / 6, 10);
        // prevalences 0.75, 0.75, 0.5 -> (0.5*0.75 + 1*0.75 + 1*0.5)/2 = 0.8125
        expect(rocAucScore(yTrue, proba, { multiClass: 'ovo', average: 'weighted' })).toBeCloseTo(0.8125, 10);
    });

    test('matrix scores without multiClass throw (sklearn multi_class="raise")', () => {
        expect(() => rocAucScore(yTrue, proba, {})).toThrow();
    });

    test('rows must be probabilities summing to one', () => {
        expect(() =>
            rocAucScore(yTrue, [
                [0.9, 0.3, 0.2],
                [0.3, 0.5, 0.2],
                [0.2, 0.2, 0.6],
                [0.1, 0.1, 0.8],
            ], { multiClass: 'ovr' }),
        ).toThrow();
    });
});

describe('regression metrics', () => {
    const actual = [2.5, 0, 2, 8]; // predictions
    const expected = [3, -0.5, 2, 7]; // ground truth

    test('meanAbsoluteError matches sklearn docs example', () => {
        // sklearn: mean_absolute_error([3,-0.5,2,7], [2.5,0,2,8]) = 0.5
        expect(meanAbsoluteError(actual, expected)).toBeCloseTo(0.5, 10);
    });

    test('meanAbsolutePercentageError matches sklearn docs example', () => {
        // sklearn: mean_absolute_percentage_error([3,-0.5,2,7], [2.5,0,2,8]) = 0.3273809523809524
        expect(meanAbsolutePercentageError(actual, expected)).toBeCloseTo(0.3273809523809524, 10);
    });

    test('meanAbsolutePercentageError clamps zero targets with machine epsilon', () => {
        // sklearn: mean_absolute_percentage_error([0], [1]) = 1 / eps ~ 4.5e15
        expect(meanAbsolutePercentageError([1], [0])).toBeCloseTo(1 / Number.EPSILON, -5);
    });

    test('medianAbsoluteError matches sklearn docs example', () => {
        // sklearn: median_absolute_error([3,-0.5,2,7], [2.5,0,2,8]) = 0.5
        expect(medianAbsoluteError(actual, expected)).toBeCloseTo(0.5, 10);
        // odd length: |diffs| = [0,0,3] -> 0
        expect(medianAbsoluteError([1, 2, 6], [1, 2, 3])).toBe(0);
    });

    test('meanSquaredLogError matches sklearn docs example', () => {
        // sklearn: mean_squared_log_error([3,5,2.5,7], [2.5,5,4,8]) = 0.03973012298459379
        expect(meanSquaredLogError([2.5, 5, 4, 8], [3, 5, 2.5, 7])).toBeCloseTo(0.0397301229845, 10);
    });

    test('meanSquaredLogError throws on negative values like sklearn', () => {
        expect(() => meanSquaredLogError([1, -2], [1, 2])).toThrow();
        expect(() => meanSquaredLogError([1, 2], [1, -2])).toThrow();
    });

    test('explainedVarianceScore matches sklearn docs example', () => {
        // sklearn: explained_variance_score([3,-0.5,2,7], [2.5,0,2,8]) = 0.9571734475374732
        expect(explainedVarianceScore(actual, expected)).toBeCloseTo(0.9571734475374732, 10);
    });

    test('explainedVarianceScore constant-target edge (force_finite default)', () => {
        expect(explainedVarianceScore([2, 2], [2, 2])).toBe(1);
        expect(explainedVarianceScore([1, 3], [2, 2])).toBe(0);
    });

    test('maxError matches sklearn docs example', () => {
        // sklearn: max_error([3,2,7,1], [4,2,7,1]) = 1
        expect(maxError([4, 2, 7, 1], [3, 2, 7, 1])).toBe(1);
    });

    test('rootMeanSquaredError is sqrt of meanSquaredError', () => {
        // sklearn: root_mean_squared_error([3,-0.5,2,7], [2.5,0,2,8]) = sqrt(0.375)
        expect(rootMeanSquaredError(actual, expected)).toBeCloseTo(Math.sqrt(0.375), 12);
    });

    test('regression metrics validate lengths', () => {
        expect(() => meanAbsoluteError([1], [1, 2])).toThrow('same length');
        expect(() => medianAbsoluteError([], [])).toThrow('non-empty');
        expect(() => maxError([], [])).toThrow('non-empty');
    });
});

describe('silhouette', () => {
    const X = [[0, 0], [0, 1], [10, 0], [10, 1]];
    const labels = [0, 0, 1, 1];
    // each sample: a=1, b=(10 + sqrt(101))/2 -> s = 1 - 2/(10 + sqrt(101)) = 0.90024876...
    const perSample = 1 - 2 / (10 + Math.sqrt(101));

    test('silhouetteScore for well-separated clusters', () => {
        // sklearn: silhouette_score([[0,0],[0,1],[10,0],[10,1]], [0,0,1,1]) = 0.9002487562...
        expect(silhouetteScore(X, labels)).toBeCloseTo(perSample, 10);
    });

    test('silhouetteSamples returns per-sample coefficients', () => {
        const samples = silhouetteSamples(X, labels);
        expect(samples).toHaveLength(4);
        samples.forEach(s => expect(s).toBeCloseTo(perSample, 10));
    });

    test('precomputed distance matrix agrees with euclidean', () => {
        const s101 = Math.sqrt(101);
        const D = [
            [0, 1, 10, s101],
            [1, 0, s101, 10],
            [10, s101, 0, 1],
            [s101, 10, 1, 0],
        ];
        expect(silhouetteScore(D, labels, { metric: 'precomputed' })).toBeCloseTo(perSample, 10);
    });

    test('singleton clusters score 0 (sklearn convention)', () => {
        // sklearn: silhouette_samples([[0,0],[0,1],[5,5]], [0,0,1])
        //        = [1 - 1/sqrt(50), 1 - 1/sqrt(41), 0]
        const samples = silhouetteSamples([[0, 0], [0, 1], [5, 5]], [0, 0, 1]);
        expect(samples[0]).toBeCloseTo(1 - 1 / Math.sqrt(50), 10);
        expect(samples[1]).toBeCloseTo(1 - 1 / Math.sqrt(41), 10);
        expect(samples[2]).toBe(0);
        expect(silhouetteScore([[0, 0], [0, 1], [5, 5]], [0, 0, 1])).toBeCloseTo(
            (1 - 1 / Math.sqrt(50) + (1 - 1 / Math.sqrt(41))) / 3,
            10,
        );
    });

    test('requires 2 <= nLabels <= nSamples - 1 like sklearn', () => {
        expect(() => silhouetteScore([[0], [1], [2]], [0, 0, 0])).toThrow('Number of labels is 1');
        expect(() => silhouetteScore([[0], [1], [2]], [0, 1, 2])).toThrow('Number of labels is 3');
    });

    test('precomputed matrices must be square with a zero diagonal', () => {
        expect(() => silhouetteScore([[0, 1], [1, 0], [2, 2]], [0, 0, 1], { metric: 'precomputed' })).toThrow();
        expect(() =>
            silhouetteScore([[1, 1, 2], [1, 0, 2], [2, 2, 0]], [0, 0, 1], { metric: 'precomputed' }),
        ).toThrow();
    });
});

describe('daviesBouldinScore', () => {
    test('well-separated clusters', () => {
        // centroids (0,0.5), (10,0.5); s=0.5 each, M=10 -> DB = (0.5+0.5)/10 = 0.1
        // sklearn: davies_bouldin_score([[0,0],[0,1],[10,0],[10,1]], [0,0,1,1]) = 0.1
        expect(daviesBouldinScore([[0, 0], [0, 1], [10, 0], [10, 1]], [0, 0, 1, 1])).toBeCloseTo(0.1, 10);
    });

    test('overlapping clusters', () => {
        // sklearn: davies_bouldin_score([[0,0],[0,1],[1,0],[5,5]], [0,0,1,1])
        // s0=0.5, s1=sqrt(10.25), M=sqrt(13) -> (0.5+sqrt(10.25))/sqrt(13)
        expect(daviesBouldinScore([[0, 0], [0, 1], [1, 0], [5, 5]], [0, 0, 1, 1])).toBeCloseTo(
            (0.5 + Math.sqrt(10.25)) / Math.sqrt(13),
            10,
        );
    });

    test('zero within-cluster scatter gives 0', () => {
        expect(daviesBouldinScore([[0, 0], [0, 0], [5, 5], [5, 5]], [0, 0, 1, 1])).toBe(0);
    });

    test('validates the number of labels', () => {
        expect(() => daviesBouldinScore([[0], [1]], [0, 0])).toThrow('Number of labels');
    });
});

describe('calinskiHarabaszScore', () => {
    test('well-separated clusters', () => {
        // BSS=100, WSS=1, n=4, k=2 -> (100/1)/(1/2) = 200
        // sklearn: calinski_harabasz_score([[0,0],[0,1],[10,0],[10,1]], [0,0,1,1]) = 200
        expect(calinskiHarabaszScore([[0, 0], [0, 1], [10, 0], [10, 1]], [0, 0, 1, 1])).toBeCloseTo(200, 8);
    });

    test('overlapping clusters', () => {
        // sklearn: calinski_harabasz_score([[0,0],[0,1],[1,0],[5,5]], [0,0,1,1])
        // BSS=13, WSS=21 -> (13/1)/(21/2) = 26/21
        expect(calinskiHarabaszScore([[0, 0], [0, 1], [1, 0], [5, 5]], [0, 0, 1, 1])).toBeCloseTo(26 / 21, 10);
    });

    test('zero within-cluster dispersion returns 1 like sklearn', () => {
        expect(calinskiHarabaszScore([[0, 0], [0, 0], [5, 5], [5, 5]], [0, 0, 1, 1])).toBe(1);
    });

    test('validates the number of labels', () => {
        expect(() => calinskiHarabaszScore([[0], [1], [2]], [0, 1, 2])).toThrow('Number of labels');
    });
});

describe('label-based clustering metrics', () => {
    // Shared non-trivial example: contingency [[2,0],[1,1],[0,2]]
    // MI = (2/3)ln2; H_true = ln3; H_pred = ln2
    const labelsTrue = [0, 0, 1, 1, 2, 2];
    const labelsPred = [0, 0, 0, 1, 1, 1];
    const mi = (2 / 3) * Math.log(2);

    test('mutualInfoScore', () => {
        // sklearn: mutual_info_score([0,0,1,1], [0,0,1,1]) = ln2 (label permutation invariant)
        expect(mutualInfoScore([0, 0, 1, 1], [0, 0, 1, 1])).toBeCloseTo(Math.log(2), 10);
        expect(mutualInfoScore([0, 0, 1, 1], [1, 1, 0, 0])).toBeCloseTo(Math.log(2), 10);
        // independent labelings: MI = 0
        expect(mutualInfoScore([0, 0, 1, 1], [0, 1, 0, 1])).toBe(0);
        // sklearn: mutual_info_score([0,0,1,1,2,2], [0,0,0,1,1,1]) = (2/3)*ln2 = 0.4620981203732969
        expect(mutualInfoScore(labelsTrue, labelsPred)).toBeCloseTo(mi, 10);
    });

    test('homogeneity, completeness, and v-measure', () => {
        // sklearn: homogeneity_score / completeness_score / v_measure_score on the shared example
        const h = mi / Math.log(3);
        const c = mi / Math.log(2); // = 2/3
        expect(homogeneityScore(labelsTrue, labelsPred)).toBeCloseTo(h, 10);
        expect(completenessScore(labelsTrue, labelsPred)).toBeCloseTo(2 / 3, 10);
        expect(vMeasureScore(labelsTrue, labelsPred)).toBeCloseTo((2 * h * c) / (h + c), 10);
        // perfect clustering
        expect(homogeneityScore([0, 0, 1, 1], [1, 1, 0, 0])).toBeCloseTo(1, 10);
        expect(completenessScore([0, 0, 1, 1], [1, 1, 0, 0])).toBeCloseTo(1, 10);
        expect(vMeasureScore([0, 0, 1, 1], [1, 1, 0, 0])).toBeCloseTo(1, 10);
        // sklearn edge: single-class truth split into singletons -> h=1, c=0, v=0
        expect(homogeneityScore([0, 0, 0, 0], [0, 1, 2, 3])).toBe(1);
        expect(completenessScore([0, 0, 0, 0], [0, 1, 2, 3])).toBe(0);
        expect(vMeasureScore([0, 0, 0, 0], [0, 1, 2, 3])).toBe(0);
        // reversed: everything merged -> h=0, c=1, v=0
        expect(homogeneityScore([0, 1, 2, 3], [0, 0, 0, 0])).toBe(0);
        expect(completenessScore([0, 1, 2, 3], [0, 0, 0, 0])).toBe(1);
    });

    test('normalizedMutualInfoScore (arithmetic default, like sklearn)', () => {
        // sklearn: normalized_mutual_info_score(labelsTrue, labelsPred)
        //        = MI / ((ln3 + ln2)/2)
        expect(normalizedMutualInfoScore(labelsTrue, labelsPred)).toBeCloseTo(
            mi / ((Math.log(3) + Math.log(2)) / 2),
            10,
        );
        // NMI with arithmetic averaging equals the v-measure (known identity)
        expect(normalizedMutualInfoScore(labelsTrue, labelsPred)).toBeCloseTo(
            vMeasureScore(labelsTrue, labelsPred),
            12,
        );
        // average_method="geometric"
        expect(
            normalizedMutualInfoScore(labelsTrue, labelsPred, { averageMethod: 'geometric' }),
        ).toBeCloseTo(mi / Math.sqrt(Math.log(3) * Math.log(2)), 10);
        // perfect and degenerate cases per sklearn
        expect(normalizedMutualInfoScore([0, 0, 1, 1], [1, 1, 0, 0])).toBeCloseTo(1, 10);
        // identical single-cluster labelings are defined as 1.0 in sklearn
        expect(normalizedMutualInfoScore([0, 0, 0], [1, 1, 1])).toBe(1);
        // zero MI -> 0
        expect(normalizedMutualInfoScore([0, 0, 1, 1], [0, 1, 0, 1])).toBe(0);
        expect(normalizedMutualInfoScore([0, 0, 0], [0, 1, 2])).toBe(0);
    });

    test('adjustedMutualInfoScore', () => {
        // EMI for the shared example (hypergeometric model, hand-computed):
        // every cell has a=2, b=3, N=6; P(nij=1)=0.6, P(nij=2)=0.2
        // EMI = 6 * (2/6)*ln2*0.2 = 0.4*ln2
        // sklearn: adjusted_mutual_info_score([0,0,1,1,2,2], [0,0,0,1,1,1])
        const emi = 0.4 * Math.log(2);
        const normalizer = (Math.log(3) + Math.log(2)) / 2;
        expect(adjustedMutualInfoScore(labelsTrue, labelsPred)).toBeCloseTo((mi - emi) / (normalizer - emi), 10);
        // sklearn: adjusted_mutual_info_score([0,0,1,1], [0,0,1,1]) = 1.0
        expect(adjustedMutualInfoScore([0, 0, 1, 1], [0, 0, 1, 1])).toBeCloseTo(1, 10);
        // identical single-cluster labelings are defined as 1.0 in sklearn
        expect(adjustedMutualInfoScore([0, 0, 0], [5, 5, 5])).toBe(1);
        // sklearn: adjusted_mutual_info_score([0,0,1,1], [0,1,0,1]) = -0.5
        // hand-derived: MI=0; each cell a=b=2, N=4, P(nij=2)=1/6 -> EMI=4*(2/4)*ln2*(1/6)=ln2/3;
        // AMI = (0 - ln2/3)/(ln2 - ln2/3) = -0.5
        expect(adjustedMutualInfoScore([0, 0, 1, 1], [0, 1, 0, 1])).toBeCloseTo(-0.5, 8);
    });

    test('fowlkesMallowsScore', () => {
        // sklearn: fowlkes_mallows_score([0,0,1,1], [0,0,1,1]) = 1
        expect(fowlkesMallowsScore([0, 0, 1, 1], [0, 0, 1, 1])).toBeCloseTo(1, 10);
        expect(fowlkesMallowsScore([0, 0, 1, 1], [1, 1, 0, 0])).toBeCloseTo(1, 10);
        // no pair co-clustered in both -> 0
        expect(fowlkesMallowsScore([0, 0, 1, 1], [0, 1, 0, 1])).toBe(0);
        // shared example: tk=4, pk=6, qk=12 -> sqrt(4/6)*sqrt(4/12) = sqrt(2)/3
        expect(fowlkesMallowsScore(labelsTrue, labelsPred)).toBeCloseTo(Math.sqrt(2) / 3, 10);
    });

    test('randScore', () => {
        // sklearn: rand_score([0,0,1,1], [1,1,0,0]) = 1.0
        expect(randScore([0, 0, 1, 1], [1, 1, 0, 0])).toBe(1);
        // sklearn docs: rand_score([0,0,1,2], [0,0,1,1]) = 0.8333...
        expect(randScore([0, 0, 1, 2], [0, 0, 1, 1])).toBeCloseTo(5 / 6, 10);
        // fully crossed: agreements = 2 of 6 pairs
        expect(randScore([0, 0, 1, 1], [0, 1, 0, 1])).toBeCloseTo(1 / 3, 10);
        // single sample: no pairs -> 1 (sklearn convention)
        expect(randScore([0], [0])).toBe(1);
    });

    test('adjustedRandScore still agrees with its documented sklearn example', () => {
        // sklearn: adjusted_rand_score([0,0,1,2], [0,0,1,1]) = 0.5714285714285714
        expect(adjustedRandScore([0, 0, 1, 2], [0, 0, 1, 1])).toBeCloseTo(4 / 7, 10);
    });

    test('label-based metrics validate input lengths', () => {
        expect(() => mutualInfoScore([0], [0, 1])).toThrow('same length');
        expect(() => normalizedMutualInfoScore([], [])).toThrow('non-empty');
        expect(() => randScore([], [])).toThrow('non-empty');
        expect(() => fowlkesMallowsScore([0], [0, 1])).toThrow('same length');
    });
});
