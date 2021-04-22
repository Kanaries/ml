import {
    gaussianElimination,
    identityMatrix,
    transpose,
    augmentMatrix
} from './basic';

import * as Inverse from './inverse';

export * from './determinant';
export * as Regression from './regression';

export {
    augmentMatrix,
    gaussianElimination,
    transpose,
    identityMatrix,
    Inverse
}
