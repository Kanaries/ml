import { KMeans } from './kmeans';
import { DBScan } from './dbscan';
import { kmeansPlusPlus } from './kmeans_plusplus';
import { OPTICS } from './optics';
import { MeanShift } from './meanShift';
import { HDBScan } from './hdbscan';
import { AgglomerativeClustering } from './agglomerativeClustering';
import { SpectralClustering } from './spectralClustering';
import { MiniBatchKMeans } from './miniBatchKMeans';

export { KMeans, DBScan, HDBScan, MeanShift, OPTICS, kmeansPlusPlus, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans };
/** sklearn-compatible names; legacy camel-cased exports remain supported. */
export const DBSCAN = DBScan;
export type DBSCAN = DBScan;
export const HDBSCAN = HDBScan;
export type HDBSCAN = HDBScan;
export type { AgglomerativeClusteringProps, AgglomerativeLinkage } from './agglomerativeClustering';
export type { SpectralClusteringProps, SpectralAffinity } from './spectralClustering';
export type { MiniBatchKMeansProps } from './miniBatchKMeans';
