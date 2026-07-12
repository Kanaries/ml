import { GaussianMixture, GaussianMixtureProps } from './gaussianMixture';
import {
    BayesianGaussianMixture,
    BayesianGaussianMixtureProps,
    BayesianCovarianceType,
    WeightConcentrationPriorType,
} from './bayesianGaussianMixture';
import { CovarianceType, Covariances } from './common';

export { GaussianMixture, BayesianGaussianMixture };
export type {
    GaussianMixtureProps,
    BayesianGaussianMixtureProps,
    CovarianceType,
    Covariances,
    BayesianCovarianceType,
    WeightConcentrationPriorType,
};
