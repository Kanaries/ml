import { makeBlobs } from './makeBlobs';
import { makeClassification } from './makeClassification';
import { makeRegression, makeLowRankMatrix } from './makeRegression';
import { makeMoons } from './makeMoons';
import { makeCircles } from './makeCircles';
import { createGaussianGenerator } from './common';

export { makeBlobs, makeClassification, makeRegression, makeLowRankMatrix, makeMoons, makeCircles, createGaussianGenerator };

export type { Dataset } from './common';
export type { MakeBlobsProps } from './makeBlobs';
export type { MakeClassificationProps } from './makeClassification';
export type { MakeRegressionProps, MakeRegressionResult } from './makeRegression';
export type { MakeMoonsProps } from './makeMoons';
export type { MakeCirclesProps } from './makeCircles';
