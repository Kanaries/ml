import type { Metadata } from 'next';
import Link from 'next/link';
import { DecisionTreePlayground } from '@/components/decisionTreePlayground';
import { CodeTabs } from '@/components/tools/CodeTabs';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export const metadata: Metadata = {
  title: { absolute: 'Decision Tree Visualization — Interactive Classifier & Tree Diagram' },
  description: 'Build and visualize a decision tree in JavaScript. Change depth, criterion, noise, and samples while inspecting the fitted boundary and tree diagram.',
  keywords: ['decision tree visualization', 'visualize decision tree', 'decision tree JavaScript', 'sklearn tree visualization', 'interactive decision tree'],
  alternates: { canonical: `${siteUrl}/playground/decision-tree` },
  openGraph: { title: 'Interactive Decision Tree Visualization in JavaScript', description: 'Change the data and hyperparameters, then inspect the live decision surface and fitted tree.', url: `${siteUrl}/playground/decision-tree` },
};

const javascriptCode = `import { Tree } from '@kanaries/ml';

const classifier = new Tree.DecisionTreeClassifier({
  max_depth: 4,
  min_samples_split: 4,
  criterion: 'gini',
  randomState: 42,
});

classifier.fit(X, y);
const predictions = classifier.predict(testX);`;

const pythonCode = `from sklearn.tree import DecisionTreeClassifier, plot_tree

classifier = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=4,
    criterion='gini',
    random_state=42
)

classifier.fit(X, y)
predictions = classifier.predict(test_X)
plot_tree(classifier, filled=True)`;

const faq: ToolFaq[] = [
  { question: 'How do I visualize a decision tree?', answer: 'Fit the estimator, inspect its root and child nodes, and draw each split as a node connected to its children. The playground also plots the classifier’s predictions across feature space so rules and geometric regions can be read together.' },
  { question: 'How can I visualize a scikit-learn decision tree?', answer: 'In Python, use sklearn.tree.plot_tree for Matplotlib output or export_graphviz for Graphviz. The JavaScript page shows the corresponding @kanaries/ml workflow and renders the live tree directly in the browser.' },
  { question: 'What does max depth do?', answer: 'Max depth limits how many split levels a tree can grow. Shallow trees are easier to explain and often generalize better; deep trees can capture fine interactions but may memorize noise.' },
  { question: 'What is the difference between Gini and entropy?', answer: 'Both measure node impurity. Gini uses squared class proportions, while entropy uses information content. They often choose similar splits, so depth and minimum-sample constraints usually have a larger practical effect.' },
  { question: 'Is the visualization using a real model?', answer: 'Yes. The boundary, holdout score, and exposed tree structure come from Tree.DecisionTreeClassifier in @kanaries/ml, fitted locally whenever a control or observation changes.' },
];

export default function DecisionTreePage() {
  return (
    <ToolPageLayout
      name="Decision tree visualization"
      description="Grow a decision tree in your browser and connect every rule to the region it creates. Tune depth, split constraints, criterion, and noise, then add your own observations."
      pathname="/playground/decision-tree"
      sectionName="Playground"
      sectionPath="/playground"
      eyebrow="Interactive tree learning playground"
      activityLabel="Live decision tree"
      tool={<><DecisionTreePlayground task="classification" variant="decision" /><div style={{ padding: '0 1rem 1rem' }}><CodeTabs javascript={javascriptCode} python={pythonCode} /></div></>}
      faq={faq}
      related={[
        { href: '/docs/apis/tree/decisionTreeClassifier', title: 'Decision Tree JavaScript API', description: 'Constructor options, split behavior, fit, and predict.' },
        { href: '/playground/random-forest', title: 'Random Forest visualization', description: 'See how bootstrap aggregation stabilizes many randomized trees.' },
        { href: '/playground/knn', title: 'KNN visualization', description: 'Compare axis-aligned rules with a local distance-based boundary.' },
      ]}
    >
      <h2>What a decision tree classifier does</h2>
      <p>A decision tree learns a sequence of if/else rules from labeled data. At each internal node it chooses a feature and threshold that make the child groups purer than the parent. A sample travels left or right according to the rule until it reaches a leaf, where the stored majority class becomes the prediction. The result is nonlinear but remains readable as a hierarchy.</p>
      <p>Trees are useful when feature interactions matter, relationships contain thresholds, and stakeholders need an explanation closer to business rules than coefficients. They require little distributional modeling and do not need standardized numeric scales. Common applications include eligibility logic, risk triage, churn signals, quality control, and interpretable baselines for tabular classification.</p>

      <h2>Use the interactive decision tree visualization</h2>
      <p>The left chart shows predictions across two-dimensional feature space. Each rectangular color region is produced by axis-aligned splits. The right diagram exposes the fitted nodes. Change the dataset, maximum depth, minimum samples, noise, or impurity criterion and both views refit with <code>Tree.DecisionTreeClassifier</code> from <code>@kanaries/ml</code>.</p>
      <p>Click inside the surface to add a sample from the selected class. Training and holdout accuracy update independently, which makes overfitting visible. If training accuracy rises while holdout accuracy falls, the extra rules are fitting peculiarities of the sample rather than a reusable relationship. Everything is calculated in the browser without a Python service.</p>

      <h2>How to read a tree diagram</h2>
      <p>Begin at the root. A rule such as <code>feature 1 ≤ 0.42</code> sends matching observations to the left child and the rest to the right. Repeat until a leaf displays a class. Early nodes affect many samples and usually describe the broadest separation; later nodes refine smaller subgroups. A path from root to leaf can be translated into a conjunction of rules for an individual prediction.</p>
      <p>The surface provides a complementary interpretation. Every vertical boundary comes from a split on the horizontal feature, and every horizontal boundary comes from a split on the vertical feature. More levels create smaller rectangles. Curved patterns such as moons therefore require a staircase of regions, revealing both the flexibility and inefficiency of axis-aligned trees.</p>

      <h2>Depth, minimum samples, and overfitting</h2>
      <p>Maximum depth is the most visible complexity control. A depth-one stump makes one split. Increasing depth allows interactions and local corrections, but an unrestricted tree can create leaves for isolated observations. Minimum samples per split prevents small nodes from dividing further. Leaf-size constraints and pruning serve related purposes in other implementations.</p>
      <p>Choose these settings with cross-validation or a representative holdout set. Accuracy is not enough for imbalanced problems; review per-class recall, precision, a confusion matrix, and the operational cost of mistakes. The playground’s holdout metric is a teaching signal, while a production evaluation should preserve time, user, or group boundaries found in the real application.</p>

      <h2>Gini impurity versus entropy</h2>
      <p>Gini impurity is one minus the sum of squared class proportions. Entropy is the negative sum of each proportion times its logarithm. Both equal zero for a pure node and increase as classes mix. The algorithm evaluates candidate thresholds and prefers the split with the largest impurity reduction, weighted by child size.</p>
      <p>The criteria often produce similar trees, although entropy may react slightly differently near small class proportions. Treat the choice as a hyperparameter rather than a philosophical commitment. Dataset quality, leakage prevention, depth, and minimum-sample settings usually influence generalization more strongly.</p>

      <h2>How to visualize sklearn and JavaScript trees</h2>
      <p>Scikit-learn users commonly call <code>plot_tree</code> for a Matplotlib figure or <code>export_graphviz</code> for Graphviz. Those functions draw the fitted Python estimator. In JavaScript, this page renders the node structure exposed by <code>@kanaries/ml</code> and simultaneously evaluates a prediction grid, making it suitable for interactive browser lessons and product explainability views.</p>
      <p>The code tabs show parallel fitting workflows. The important comparison is semantic: identical preprocessing, split constraints, criterion, and validation design. Exact structures can differ when thresholds tie or implementations apply deterministic tie breaking differently. Compare predictions and metrics on fixed samples rather than assuming node identities must match.</p>

      <h2>A responsible deployment workflow</h2>
      <p>Define features available at decision time, split data before tuning, and check for leakage. Fit several depth and minimum-sample candidates, select with relevant validation metrics, then inspect paths for implausible shortcuts. Trees can encode sensitive proxies and sharp threshold discontinuities, so explanation does not automatically imply fairness or causality.</p>
      <p>Test boundary cases on both sides of important thresholds. A tiny measurement change can send two otherwise similar people to different leaves, so verify that precision, rounding, missing-value handling, and upstream units remain consistent. Record the model version and the complete decision path when predictions affect support, risk, or eligibility. That trace helps distinguish a model rule from a feature-pipeline problem and gives reviewers concrete evidence to challenge.</p>
      <p>Monitor feature ranges, leaf traffic, and outcome quality after release. New observations outside training ranges still reach a leaf, but that does not make the extrapolation reliable. Read the <Link href="/docs/apis/tree/decisionTreeClassifier">Decision Tree JavaScript API</Link>, compare variance reduction in the <Link href="/playground/random-forest">Random Forest playground</Link>, or contrast local voting in the <Link href="/playground/knn">KNN visualization</Link>.</p>
    </ToolPageLayout>
  );
}
