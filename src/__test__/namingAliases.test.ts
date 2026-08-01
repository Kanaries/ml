import { Clusters, DiscriminantAnalysis, Manifold, Neighbors } from '../index';
import type { BaggingRegressorProps, ExtraTreesClassifierProps, ExtraTreesRegressorProps } from '../ensemble';
import type { IsomapProps } from '../manifold';
import type { LocalOutlierFactorProps } from '../neighbors';

test('sklearn naming aliases preserve legacy constructors', () => {
    expect(Clusters.DBSCAN).toBe(Clusters.DBScan);
    expect(Clusters.HDBSCAN).toBe(Clusters.HDBScan);
    expect(Neighbors.KNeighborsClassifier).toBe(Neighbors.KNearestNeighbors);
    expect(DiscriminantAnalysis.LDA).toBe(DiscriminantAnalysis.LinearDiscriminantAnalysis);
    expect(DiscriminantAnalysis.QDA).toBe(DiscriminantAnalysis.QuadraticDiscriminantAnalysis);
    expect(Manifold.LLE).toBe(Manifold.LocallyLinearEmbedding);
});

test('Wave A public prop types are exported by their module namespaces', () => {
    const bagging: BaggingRegressorProps = { nEstimators: 2 };
    const extraClassifier: ExtraTreesClassifierProps = { nEstimators: 2, max_features: 'sqrt' };
    const extraRegressor: ExtraTreesRegressorProps = { nEstimators: 2, max_features: 'all' };
    const isomap: IsomapProps = { nNeighbors: 2 };
    const lof: LocalOutlierFactorProps = { novelty: true };
    expect([bagging, extraClassifier, extraRegressor, isomap, lof]).toHaveLength(5);
});
