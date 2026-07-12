import { registerEstimator } from '../base/estimator';
import { RidgeRegression, RidgeRegressionProps } from './ridgeRegression';

export interface RidgeProps extends RidgeRegressionProps {}

export class Ridge extends RidgeRegression {}
registerEstimator('Ridge', Ridge);
