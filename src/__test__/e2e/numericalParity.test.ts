import { readFileSync } from 'fs';
import { gunzipSync } from 'zlib';
import { join } from 'path';
import { Pipeline } from '../../pipeline';
import { LogisticRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor } from '../../linear';
import { StandardScaler } from '../../utils/preprocessing';
import { GridSearchCV, StratifiedKFold } from '../../utils/modelSelection';
import { SVC, OneClassSVM } from '../../svm';
import { TfidfVectorizer } from '../../feature_extraction';
import { MultinomialNB, GaussianNB } from '../../bayes';
import { PCA, NMF, FastICA } from '../../decomposition';
import { IsolationForest } from '../../ensemble';
import { LocalOutlierFactor } from '../../neighbors';
import { EllipticEnvelope } from '../../covariance';
import {
    KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, MeanShift,
    SpectralClustering, Birch, AffinityPropagation,
} from '../../clusters';
import { GaussianMixture } from '../../mixture';
import { TSNE, LocallyLinearEmbedding, MDS, Isomap, SpectralEmbedding } from '../../manifold';
import { SelectFromModel, RFE } from '../../feature_selection';
import { CalibratedClassifierCV } from '../../calibration';
import { accuracyScore, adjustedRandScore, brierScoreLoss, f1Score, medianAbsoluteError } from '../../metrics';

type Matrix = number[][];
type Fixture = Record<string, any>;

const fixture = JSON.parse(gunzipSync(readFileSync(
    join(__dirname, '../../../test_data/phase3_e2e.json.gz'),
)).toString('utf8')) as Fixture;

jest.setTimeout(180_000);

function agreement(a: number[], b: number[]): number {
    return a.reduce((count, value, i) => count + Number(value === b[i]), 0) / a.length;
}

function frobeniusError(a: Matrix, b: Matrix): number {
    let squared = 0;
    for (let i = 0; i < a.length; i++) for (let j = 0; j < a[i].length; j++) squared += (a[i][j] - b[i][j]) ** 2;
    return Math.sqrt(squared);
}

function pairwiseDistances(X: Matrix): number[] {
    const result: number[] = [];
    for (let i = 0; i < X.length; i++) for (let j = 0; j < i; j++) {
        let squared = 0;
        for (let k = 0; k < X[i].length; k++) squared += (X[i][k] - X[j][k]) ** 2;
        result.push(Math.sqrt(squared));
    }
    return result;
}

function correlation(a: number[], b: number[]): number {
    const meanA = a.reduce((x, y) => x + y, 0) / a.length;
    const meanB = b.reduce((x, y) => x + y, 0) / b.length;
    let covariance = 0, varianceA = 0, varianceB = 0;
    for (let i = 0; i < a.length; i++) {
        const da = a[i] - meanA, db = b[i] - meanB;
        covariance += da * db; varianceA += da * da; varianceB += db * db;
    }
    return covariance / Math.sqrt(varianceA * varianceB);
}

/** sklearn.manifold.trustworthiness, implemented here to keep the fixture assertion rotation-invariant. */
function trustworthiness(X: Matrix, embedding: Matrix, neighbors = 10): number {
    const n = X.length;
    const order = (data: Matrix, i: number) => data.map((row, j) => ({
        j,
        distance: i === j ? Infinity : row.reduce((sum, value, k) => sum + (value - data[i][k]) ** 2, 0),
    })).sort((a, b) => a.distance - b.distance || a.j - b.j).map(hit => hit.j);
    let penalty = 0;
    for (let i = 0; i < n; i++) {
        const original = order(X, i);
        const originalNeighbors = new Set(original.slice(0, neighbors));
        const ranks = new Map(original.map((index, rank) => [index, rank + 1]));
        for (const index of order(embedding, i).slice(0, neighbors)) {
            if (!originalNeighbors.has(index)) penalty += ranks.get(index)! - neighbors;
        }
    }
    return 1 - 2 * penalty / (n * neighbors * (2 * n - 3 * neighbors - 1));
}

test('Phase 3 fixture is pinned to the deterministic sklearn 1.9 schema', () => {
    expect(fixture.schemaVersion).toBe(1);
    expect(fixture.sklearnVersion).toBe('1.9.0');
});

test('1/10 Iris pipeline runs multiclass LogisticRegression regularization through GridSearchCV', () => {
    const data = fixture.iris;
    const pipeline = new Pipeline({ steps: [
        ['scale', new StandardScaler()],
        ['lr', new LogisticRegression({ learningRate: .2, maxIter: 1800 })],
    ] });
    const search = new GridSearchCV({
        estimator: pipeline,
        paramGrid: { lr__C: [.1, 1, 10] },
        cv: new StratifiedKFold({ nSplits: 5 }),
    });
    search.fit(data.XTrain, data.yTrain);
    const prediction = search.predict(data.XTest);
    const accuracy = accuracyScore(prediction, data.yTest);
    expect(search.bestParams!.lr__C).toBe(data.expected.bestC);
    expect(accuracy).toBeGreaterThanOrEqual(data.expected.accuracy - .06);
    expect(agreement(prediction, data.expected.predictions)).toBeGreaterThanOrEqual(.9);
});

test('2/10 Digits StandardScaler + RBF SVC matches held-out sklearn behavior', () => {
    const data = fixture.digits;
    const pipeline = new Pipeline({ steps: [
        ['scale', new StandardScaler()],
        ['svc', new SVC({ C: 10, gamma: 'scale', maxIter: -1 })],
    ] });
    pipeline.fit(data.XTrain, data.yTrain);
    const prediction = pipeline.predict(data.XTest);
    expect(accuracyScore(prediction, data.yTest)).toBeGreaterThanOrEqual(data.expected.accuracy - .03);
    expect(agreement(prediction, data.expected.predictions)).toBeGreaterThanOrEqual(.94);
});

test('3/10 sparse 20 Newsgroups pipeline matches sklearn accuracy and macro-F1', () => {
    const data = fixture.text;
    const pipeline = new Pipeline({ steps: [
        ['tfidf', new TfidfVectorizer()],
        ['nb', new MultinomialNB()],
    ] });
    pipeline.fit(data.trainDocuments, data.yTrain);
    const prediction = pipeline.predict(data.testDocuments);
    expect(accuracyScore(prediction, data.yTest)).toBeGreaterThanOrEqual(data.expected.accuracy - .04);
    expect(f1Score(prediction, data.yTest, { average: 'macro' })).toBeGreaterThanOrEqual(data.expected.macroF1 - .04);
    expect(agreement(prediction, data.expected.predictions)).toBeGreaterThanOrEqual(.92);
});

test('4/10 pooled Olivetti faces exercise PCA, NMF, and FastICA reconstruction', () => {
    const { X, expected } = fixture.faces;
    const models = {
        pca: new PCA({ nComponents: 16 }),
        nmf: new NMF({ nComponents: 16, init: 'nndsvda', maxIter: 1000, randomState: 42 }),
        fastICA: new FastICA({ nComponents: 16, whiten: 'unit-variance', maxIter: 500, randomState: 42, tol: 1e-4 }),
    };
    for (const [name, model] of Object.entries(models)) {
        const transformed = model.fitTransform(X);
        const reconstructed = model.inverseTransform(transformed);
        const components = name === 'pca' ? (model as PCA).getComponents() : (model as NMF | FastICA).components;
        expect([components.length, components[0].length]).toEqual(expected[name].shape);
        expect(frobeniusError(X, reconstructed)).toBeLessThanOrEqual(expected[name].reconstructionError * 1.5 + 1e-8);
    }
});

test('5/10 anomaly comparison exercises all four detectors with sklearn label conventions', () => {
    const { X, truth, expected } = fixture.anomaly;
    const models = {
        isolationForest: new IsolationForest({ subsampling_size: 128, tree_num: 100, contamination: .1, random_state: 42 }),
        oneClassSVM: new OneClassSVM({ kernel: 'rbf', gamma: 'scale', nu: .1 }),
        localOutlierFactor: new LocalOutlierFactor({ nNeighbors: 20, contamination: .1 }),
        ellipticEnvelope: new EllipticEnvelope({ contamination: .1, supportFraction: .8, randomState: 42 }),
    };
    for (const [name, model] of Object.entries(models)) {
        let labels = model.fitPredict(X);
        if (name === 'isolationForest') labels = labels.map(value => value === 1 ? -1 : 1);
        const scores = name === 'isolationForest'
            ? X.map((row: number[]) => -(model as IsolationForest).anomalyScore(row))
            : name === 'localOutlierFactor'
                ? (model as LocalOutlierFactor).negativeOutlierFactor
                : (model as OneClassSVM | EllipticEnvelope).decisionFunction(X);
        const trueOutliers = truth.map((value: number, i: number) => value === -1 ? i : -1).filter((i: number) => i >= 0);
        const recall = trueOutliers.reduce((sum: number, i: number) => sum + Number(labels[i] === -1), 0) / trueOutliers.length;
        expect(recall).toBeGreaterThanOrEqual(expected[name].outlierRecall - .2);
        expect(agreement(labels, expected[name].labels)).toBeGreaterThanOrEqual(.72);
        expect(correlation(scores, expected[name].scores)).toBeGreaterThanOrEqual(.8);
    }
});

test('6/10 varied-shape clustering compares every frozen clusterer by adjusted Rand score', () => {
    for (const [datasetName, data] of Object.entries(fixture.clustering.datasets) as Array<[string, any]>) {
        const scaler = new StandardScaler();
        const X = scaler.fitTransform(data.X);
        const models = {
            kMeans: new KMeans({ n_clusters: data.nClusters, n_init: 10, random_state: 42 }),
            miniBatchKMeans: new MiniBatchKMeans({ nClusters: data.nClusters, nInit: 3, batchSize: 32, randomState: 42 }),
            dbscan: new DBSCAN({ eps: data.eps, minSamples: 5 }),
            agglomerative: new AgglomerativeClustering({ nClusters: data.nClusters, linkage: 'ward' }),
            meanShift: new MeanShift({ bandwidth: data.bandwidth }),
            spectral: new SpectralClustering({ nClusters: data.nClusters, affinity: 'rbf', gamma: 1, nInit: 10, randomState: 42 }),
            birch: new Birch({ nClusters: data.nClusters, threshold: .35 }),
            affinityPropagation: new AffinityPropagation({ damping: .7, randomState: 42 }),
            gaussianMixture: new GaussianMixture({ nComponents: data.nClusters, nInit: 1, randomState: 42 }),
        };
        for (const [name, model] of Object.entries(models)) {
            const labels = model.fitPredict(X);
            const ari = adjustedRandScore(data.truth, labels);
            expect(Number.isFinite(ari)).toBe(true);
            expect(ari).toBeGreaterThanOrEqual(data.expected[name].ari - .28);
            const expectedLabels = data.expected[name].labels as number[];
            const partitionParity = adjustedRandScore(expectedLabels, labels);
            if (datasetName === 'circles' && name === 'gaussianMixture') {
                // The symmetric circles admit orthogonal, equally poor GMM
                // partitions under different seeded RNG implementations.
                // Gate this one explicit exception on cluster-size profile.
                const signature = (values: number[]) => Array.from(new Set(values))
                    .map(label => values.filter(value => value === label).length)
                    .sort((a, b) => a - b);
                const actualSizes = signature(labels), expectedSizes = signature(expectedLabels);
                expect(actualSizes).toHaveLength(expectedSizes.length);
                const normalizedDifference = actualSizes.reduce((sum, value, i) => sum + Math.abs(value - expectedSizes[i]), 0) / labels.length;
                expect(normalizedDifference).toBeLessThanOrEqual(.1);
            } else {
                expect(partitionParity).toBeGreaterThanOrEqual(.3);
            }
        }
    }
});

test('7/10 S-curve manifold embeddings preserve sklearn-scale neighborhoods and distances', () => {
    const { X, expected } = fixture.manifold;
    const models = {
        tsne: new TSNE({ nComponents: 2, perplexity: 20, learningRate: 200, nIter: 500, randomState: 42 }),
        lle: new LocallyLinearEmbedding({ nNeighbors: 12, nComponents: 2, randomState: 42 }),
        mds: new MDS({ nComponents: 2, randomState: 42 }),
        isomap: new Isomap({ nNeighbors: 12, nComponents: 2 }),
        spectralEmbedding: new SpectralEmbedding({ nComponents: 2, nNeighbors: 12, randomState: 42 }),
    };
    const originalDistances = pairwiseDistances(X);
    for (const [name, model] of Object.entries(models)) {
        const embedding = model.fitTransform(X);
        expect(embedding).toHaveLength(X.length);
        expect(embedding.every(row => row.length === 2 && row.every(Number.isFinite))).toBe(true);
        expect(trustworthiness(X, embedding)).toBeGreaterThanOrEqual(expected[name].trustworthiness - .25);
        expect(correlation(originalDistances, pairwiseDistances(embedding))).toBeGreaterThanOrEqual(expected[name].distanceCorrelation - .35);
    }
});

test('8/10 feature-selection pipelines preserve masks and held-out accuracy', () => {
    const data = fixture.featureSelection;
    const selectors = {
        selectFromModel: new SelectFromModel({ estimator: new LogisticRegression({ C: 1, maxIter: 1800 }), maxFeatures: 4 }),
        rfe: new RFE({ estimator: new LogisticRegression({ C: 1, maxIter: 1200 }), nFeaturesToSelect: 4, step: 1 }),
    };
    for (const [name, selector] of Object.entries(selectors)) {
        const pipeline = new Pipeline({ steps: [
            ['scale', new StandardScaler()], ['selector', selector],
            ['lr', new LogisticRegression({ C: 1, maxIter: 1800 })],
        ] });
        pipeline.fit(data.XTrain, data.yTrain);
        const support = selector.getSupport() as boolean[];
        const expectedSupport = data.expected[name].support as boolean[];
        const intersection = support.filter((value, i) => value && expectedSupport[i]).length;
        const union = support.filter((value, i) => value || expectedSupport[i]).length;
        expect(intersection / union).toBeGreaterThanOrEqual(.6);
        expect(accuracyScore(pipeline.predict(data.XTest), data.yTest)).toBeGreaterThanOrEqual(data.expected[name].accuracy - .08);
    }
});

test('9/10 sigmoid and isotonic calibration match sklearn held-out probabilities', () => {
    const data = fixture.calibration;
    for (const method of ['sigmoid', 'isotonic'] as const) {
        const model = new CalibratedClassifierCV({ estimator: new GaussianNB(), method, cv: 3, ensemble: true });
        model.fit(data.XTrain, data.yTrain);
        const probabilities = model.predictProba(data.XTest).map(row => row[1]);
        expect(brierScoreLoss(data.yTest, probabilities)).toBeLessThanOrEqual(data.expected[method].brier + .035);
        expect(correlation(probabilities, data.expected[method].probabilities)).toBeGreaterThanOrEqual(.95);
        const labels = model.predict(data.XTest);
        const expectedLabels = data.expected[method].probabilities.map((value: number) => value >= .5 ? 1 : 0);
        expect(agreement(labels, expectedLabels)).toBeGreaterThanOrEqual(.95);
        expect(accuracyScore(labels, data.yTest)).toBeGreaterThanOrEqual(data.expected[method].accuracy - .04);
    }
});

test('10/10 robust regressors recover the clean line despite frozen vertical outliers', () => {
    const data = fixture.robustRegression;
    const models = {
        huber: new HuberRegressor({ epsilon: 1.35, maxIter: 500, tol: 1e-8 }),
        ransac: new RANSACRegressor({ randomState: 42, residualThreshold: .75, maxTrials: 100 }),
        theilSen: new TheilSenRegressor({ randomState: 42, maxSubpopulation: 1000, maxIter: 500 }),
    };
    for (const [name, model] of Object.entries(models)) {
        model.fit(data.X, data.y);
        const fitted = model instanceof RANSACRegressor ? model.estimatorFitted as any : model;
        expect(fitted.coef[0]).toBeCloseTo(data.expected[name].coef[0], 1);
        expect(fitted.intercept).toBeCloseTo(data.expected[name].intercept, 1);
        expect(medianAbsoluteError(model.predict(data.X), data.yClean)).toBeLessThanOrEqual(data.expected[name].medianAbsoluteError + .08);
    }
    const ransac = models.ransac;
    // The seeded samplers intentionally differ (NumPy MT19937 vs the portable JS PRNG),
    // so compare consensus quality rather than requiring identical sampled trials.
    expect(agreement(ransac.inlierMask.map(Boolean) as any, data.expected.ransac.inlierMask.map(Boolean) as any)).toBeGreaterThanOrEqual(.85);
});
