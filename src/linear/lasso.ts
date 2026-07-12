import { registerEstimator } from '../base/estimator';
import { LassoRegression, LassoRegressionProps } from './lassoRegression';

export interface LassoProps extends LassoRegressionProps {}

export class Lasso extends LassoRegression {}
registerEstimator('Lasso', Lasso);
