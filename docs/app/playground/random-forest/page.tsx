import type { Metadata } from 'next';
import Link from 'next/link';
import { RandomForestPlayground } from '@/components/playground/RandomForestPlayground';
import { CodeTabs } from '@/components/tools/CodeTabs';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');
export const metadata: Metadata = {
  title: { absolute: 'Random Forest Visualization — Interactive JavaScript Playground' },
  description: 'Compare a Random Forest with one decision tree. Change tree count, depth, noise, and data while watching the decision boundary and holdout score update.',
  keywords: ['random forest visualization', 'random forest JavaScript', 'random forest playground', 'bagging visualization', 'decision forest'],
  alternates: { canonical: `${siteUrl}/playground/random-forest` },
  openGraph: { title: 'Interactive Random Forest Visualization', description: 'See how bootstrap aggregation stabilizes decision trees on noisy nonlinear data.', url: `${siteUrl}/playground/random-forest` },
};

const js = `import { Ensemble } from '@kanaries/ml';

const model = new Ensemble.RandomForestClassifier({
  nEstimators: 50,
  max_depth: 6,
  maxFeatures: 'sqrt',
  bootstrap: true,
  randomState: 42,
});
model.fit(X, y);
const predictions = model.predict(testX);`;
const py = `from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=6,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
model.fit(X, y)
predictions = model.predict(test_X)`;

const faq: ToolFaq[] = [
  { question: 'How does a Random Forest work?', answer: 'A Random Forest fits many decision trees on bootstrap samples and usually gives each split a random subset of features. Classification combines their votes, reducing the instability of one deeply fitted tree.' },
  { question: 'How many trees should a Random Forest use?', answer: 'Add trees until validation quality and predictions stabilize within your latency and memory budget. More trees generally reduce Monte Carlo variance but have diminishing returns and do not fix biased features or leakage.' },
  { question: 'Why does Random Forest use random features?', answer: 'Feature subsampling prevents the same dominant predictor from controlling every tree. Less-correlated trees provide a larger variance reduction when their votes are aggregated.' },
  { question: 'Can a Random Forest overfit?', answer: 'It is usually more resistant than a single unpruned tree, but leakage, noisy labels, extreme class imbalance, inappropriate depth, and repeated tuning against one validation set can still produce overfitting.' },
  { question: 'Does this visualization fit a real forest?', answer: 'Yes. The left boundary and holdout score come from Ensemble.RandomForestClassifier in @kanaries/ml. The right side fits a real DecisionTreeClassifier on the identical split for comparison.' },
];

export default function RandomForestPage() {
  return <ToolPageLayout name="Random Forest visualization" description="Compare dozens of bootstrapped, feature-randomized trees with one decision tree. Change the data and complexity to see when aggregation produces a steadier boundary." pathname="/playground/random-forest" sectionName="Playground" sectionPath="/playground" eyebrow="Interactive ensemble learning playground" activityLabel="Live Random Forest" tool={<><RandomForestPlayground /><div style={{ padding: '0 1rem 1rem' }}><CodeTabs javascript={js} python={py} /></div></>} faq={faq} related={[
    { href: '/docs/apis/ensemble/randomForestClassifier', title: 'Random Forest JavaScript API', description: 'Fit, predict, bootstrap, depth, feature sampling, and reproducibility.' },
    { href: '/playground/decision-tree', title: 'Decision tree visualization', description: 'Inspect the individual rule learner used by the ensemble.' },
    { href: '/docs/guides/isolation-forest', title: 'Isolation Forest guide', description: 'Use randomized trees for unsupervised anomaly detection.' },
  ]}>
    <h2>What a Random Forest classifier does</h2>
    <p>A Random Forest is an ensemble of decision trees. Instead of trusting one hierarchy fitted to one sample, it trains many related but deliberately different trees and combines their predictions. Each tree receives a bootstrap sample drawn with replacement from the training rows. At each split it can consider only a random subset of features. The final class is the forest’s majority vote.</p>
    <p>This design targets a weakness of decision trees: variance. Small changes in data can change early splits and produce a different tree. Averaging many imperfect, partly independent learners preserves nonlinear modeling while stabilizing the result. Random forests are strong baselines for tabular classification, feature screening, risk models, quality prediction, and datasets containing thresholds and interactions.</p>

    <h2>Use the Random Forest visualization</h2>
    <p>The left plot is predicted by <code>Ensemble.RandomForestClassifier</code> from <code>@kanaries/ml</code>. The right plot uses one <code>Tree.DecisionTreeClassifier</code> with the same maximum depth and training split. Change the number of trees, depth, dataset, or noise and compare boundaries and holdout accuracy directly.</p>
    <p>Boundary disagreement measures how often the two fitted models assign different classes across the grid. On noisy moons or XOR data, a single tree can create brittle rectangular pockets. The forest vote often removes isolated pockets while retaining a nonlinear outline. Every grid prediction runs locally in JavaScript, so the page is also an end-to-end browser test of the public API.</p>

    <h2>Bootstrap aggregation reduces variance</h2>
    <p>A bootstrap dataset has the same row count as the training set but includes duplicates and leaves some rows out. Each tree therefore sees a different empirical problem. If individual errors are not perfectly correlated, voting cancels part of their variation. This is bagging: bootstrap aggregating.</p>
    <p>Bagging is most effective for unstable learners such as trees. It does not automatically reduce systematic bias. A forest of shallow trees can still underfit a complex relationship, and a forest trained with leakage will repeat that shortcut very reliably. Aggregation improves an estimator; it cannot repair the definition of the learning problem.</p>

    <h2>Why random feature subsets matter</h2>
    <p>If one feature is overwhelmingly predictive, ordinary bagged trees may all choose it near the root and remain highly correlated. Restricting candidate features makes some trees discover alternative signals. An individual tree may become slightly weaker, but the collection becomes more diverse, and diversity is what lets averaging reduce variance.</p>
    <p><code>maxFeatures: 'sqrt'</code> is a common classification default. Larger subsets strengthen each split but correlate trees; smaller subsets increase diversity but may hide useful predictors too often. Treat feature sampling, depth, minimum samples, and class weighting as validation choices.</p>

    <h2>Tree count, depth, and computation</h2>
    <p>Adding trees generally makes predictions converge rather than overfit suddenly, but training time, memory, and inference cost grow. Increase the count until validation metrics and repeated-seed predictions stabilize. The exact number depends on data size, feature count, latency, and how uncertain the operational decision can be.</p>
    <p>Depth controls the bias and variance of each member. Deep trees can capture interactions and narrow regions; bootstrap and feature randomness then smooth their aggregate. Shallow members are faster and easier to constrain but may share the same underfitting. Monitor class-specific metrics instead of optimizing only overall accuracy.</p>

    <h2>Random Forest versus one decision tree</h2>
    <p>A single tree offers a compact global rule diagram and straightforward prediction paths. A forest usually predicts better and changes less when samples move, but hundreds of paths are not a simple explanation. Permutation importance, partial dependence, accumulated local effects, and local attribution can summarize behavior, each with assumptions and failure modes.</p>
    <p>Use a tree when a small auditable rule set is central. Use a forest when predictive stability matters more than a single hierarchy. The side-by-side chart makes this tradeoff concrete without claiming that a smooth-looking boundary is necessarily correct.</p>

    <h2>JavaScript and scikit-learn workflow</h2>
    <p>The code tabs align estimator concepts across environments: tree count, depth, feature sampling, bootstrapping, and seed. Exact trees can differ because random number generators and tie rules differ, but both workflows fit a matrix and labels, then predict new rows. Compare fixed-dataset metrics rather than serialized tree identity.</p>
    <p>Reproducibility requires more than a constructor seed. Preserve the data snapshot, row order, feature schema, package version, preprocessing state, and validation split. If training happens in a UI, move heavier forests to a worker so rendering stays responsive; in Node.js, benchmark concurrent scoring under realistic traffic. Serialize only through a supported model format or keep deterministic training inputs, because private tree fields are not a stable interchange contract.</p>
    <p>Split data before tuning, keep preprocessing inside cross-validation, and preserve time or group boundaries. Evaluate calibration and minority-class recall when votes drive risk decisions. Read the <Link href="/docs/apis/ensemble/randomForestClassifier">Random Forest JavaScript API</Link>, inspect a member in the <Link href="/playground/decision-tree">Decision Tree visualization</Link>, or learn how randomized partitions isolate outliers in the <Link href="/docs/guides/isolation-forest">Isolation Forest guide</Link>.</p>
  </ToolPageLayout>;
}
