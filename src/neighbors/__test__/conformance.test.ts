import { runEstimatorConformance } from '../../__test__/conformance/harness';
import { KNearestNeighbors } from '../knn';
import { KNeighborsRegressor } from '../kneighborsRegressor';
import { RadiusNeighborsClassifier } from '../radiusNeighborsClassifier';
import { RadiusNeighborsRegressor } from '../radiusNeighborsRegressor';
import { NearestCentroid } from '../nearestCentroid';
import { LocalOutlierFactor } from '../localOutlierFactor';

runEstimatorConformance([
    {
        name: 'KNearestNeighbors',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new KNearestNeighbors({ kNeighbors: 3 }),
    },
    {
        name: 'KNeighborsRegressor',
        kind: 'regressor',
        dataset: 'regression',
        create: () => new KNeighborsRegressor({ nNeighbors: 3 }),
    },
    {
        name: 'RadiusNeighborsClassifier',
        kind: 'classifier',
        dataset: 'multiclass',
        // multiclass blobs have spread 1 around 3-D centers; radius 3 covers
        // every within-blob neighbor while staying below the inter-blob gap
        create: () => new RadiusNeighborsClassifier({ radius: 3 }),
    },
    {
        name: 'RadiusNeighborsRegressor',
        kind: 'regressor',
        dataset: 'regression',
        // regression features are uniform in [-2, 2]^3, so radius 3 always
        // finds neighbors
        create: () => new RadiusNeighborsRegressor({ radius: 3 }),
    },
    {
        name: 'NearestCentroid',
        kind: 'classifier',
        dataset: 'multiclass',
        create: () => new NearestCentroid(),
    },
    {
        name: 'LocalOutlierFactor', kind: 'outlier', dataset: 'blobs',
        create: () => new LocalOutlierFactor({ nNeighbors: 5, contamination: 0.1, novelty: true }),
    },
]);
