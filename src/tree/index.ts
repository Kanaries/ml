import { DecisionTreeClassifier } from './decisionTreeClassifier';
import { DecisionTreeRegressor } from './decisionTreeRegressor';
import { ExtraTreeRegressor } from './extraTreeRegressor';
import { ExtraTreeClassifier } from './extraTreeClassifier';
export type { DecisionTreeProps } from './decisionTreeClassifier';

export { DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor };
export { treeImportances, normalizedTreeImportances, averageImportances, weightedImportances } from './featureImportances';
