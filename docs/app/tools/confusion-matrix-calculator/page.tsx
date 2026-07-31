import type { Metadata } from 'next';
import Link from 'next/link';
import { ConfusionMatrixCalculator } from '@/components/tools/ConfusionMatrixCalculator';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');
const pathname = '/tools/confusion-matrix-calculator';

const title = 'Confusion Matrix Calculator — F1, Precision, Recall (2-class & Multiclass)';
const description =
  'Calculate accuracy, precision, recall, specificity, F1, MCC, Cohen’s kappa, and multiclass averages from a confusion matrix or pasted y_true and y_pred labels.';

export const metadata: Metadata = {
  title: { absolute: title },
  description,
  keywords: [
    'confusion matrix calculator',
    'F1 score calculator',
    'precision recall calculator',
    'multiclass confusion matrix',
    'accuracy calculator',
  ],
  alternates: { canonical: `${siteUrl}${pathname}` },
  openGraph: {
    type: 'website',
    title,
    description,
    url: `${siteUrl}${pathname}`,
  },
  twitter: { card: 'summary_large_image', title, description },
};

const faq: ToolFaq[] = [
  {
    question: 'How do I calculate F1 score from a confusion matrix?',
    answer:
      'For a binary matrix, first calculate precision as TP divided by TP plus FP and recall as TP divided by TP plus FN. F1 is their harmonic mean: 2 × precision × recall divided by precision plus recall. Equivalently, F1 = 2TP / (2TP + FP + FN).',
  },
  {
    question: 'What is a good F1 score?',
    answer:
      'F1 ranges from 0 to 1, and higher is better, but there is no universal good threshold. Compare it with a simple baseline, prior model versions, and the cost of false positives versus false negatives in your application. A score of 0.8 can be excellent for one noisy problem and unacceptable for another.',
  },
  {
    question: 'What is the difference between micro, macro, and weighted F1?',
    answer:
      'Macro F1 gives every class equal weight, weighted F1 weights class scores by their true support, and micro F1 combines all class decisions before computing the score. For single-label multiclass classification, micro F1 equals accuracy. Macro F1 is especially useful when minority classes matter.',
  },
  {
    question: 'Which axis is true and which axis is predicted?',
    answer:
      'This calculator uses the common scikit-learn orientation: rows are true labels and columns are predicted labels. A cell therefore answers: among samples whose true class is this row, how many were predicted as this column?',
  },
  {
    question: 'Does this confusion matrix calculator support multiclass data?',
    answer:
      'Yes. Choose the y_true / y_pred input, paste two numeric columns, and the tool builds an n × n matrix. It reports per-class precision, recall, and F1 together with macro, micro, and support-weighted summaries, MCC, and Cohen’s kappa.',
  },
  {
    question: 'Is my data uploaded to a server?',
    answer:
      'No. Parsing, matrix construction, metric calculation, charts, and exports run locally in your browser with @kanaries/ml. The page does not need a backend calculation request.',
  },
];

export default function ConfusionMatrixCalculatorPage() {
  return (
    <ToolPageLayout
      name="Confusion Matrix & F1 Calculator"
      description={description}
      pathname={pathname}
      tool={<ConfusionMatrixCalculator />}
      faq={faq}
      related={[
        {
          href: '/tools/logistic-regression-calculator',
          title: 'Logistic regression calculator',
          description: 'Fit a binary classifier, inspect odds ratios, and visualize its decision boundary.',
        },
        {
          href: '/docs/apis/metrics',
          title: 'JavaScript Metrics API',
          description: 'Use the same accuracy, precision, recall, F1, and confusion-matrix functions in your app.',
        },
        {
          href: '/docs/apis/linear/logisticRegression',
          title: 'Logistic regression in JavaScript',
          description: 'Learn the estimator API behind the related interactive calculator.',
        },
      ]}
    >
      <h2>What is a confusion matrix?</h2>
      <p>
        A confusion matrix is a table that compares a classifier’s predicted labels with the labels that were actually
        observed. Instead of reducing model performance to one number, it preserves the direction of every mistake. In a
        binary problem, the four cells are true positives, false positives, false negatives, and true negatives. In a
        multiclass problem, the same idea expands to an n × n grid: rows represent true classes, columns represent predicted
        classes, and the diagonal contains correct predictions.
      </p>
      <p>
        That structure makes the matrix a better starting point than accuracy alone. A fraud model can be 99% accurate by
        predicting “not fraud” for nearly every transaction, yet still miss most fraud cases. The matrix exposes those false
        negatives immediately. It also lets you calculate the family of metrics used to discuss different error costs:
        precision, recall, specificity, F1, Matthews correlation coefficient, and Cohen’s kappa.
      </p>
      <p>
        Use the direct 2 × 2 mode when you already know TP, FP, FN, and TN. Use the two-column mode when you have raw
        <code>y_true</code> and <code>y_pred</code> values. The latter automatically discovers numeric class labels and works
        for multiclass classification. Rows in the heatmap follow true labels and columns follow predicted labels, matching
        the orientation returned by scikit-learn and the <Link href="/docs/apis/metrics">@kanaries/ml Metrics API</Link>.
      </p>

      <h2>How the main classification metrics are calculated</h2>
      <h3>Accuracy: the overall hit rate</h3>
      <p>
        Accuracy is the number of correct predictions divided by all predictions. For a binary matrix, that is
        <code>(TP + TN) / (TP + TN + FP + FN)</code>. It is easy to communicate and useful when classes are reasonably
        balanced and different mistakes have similar costs. It becomes misleading when one class dominates. Always inspect
        support per class and compare accuracy with recall, precision, or macro F1 before declaring a model successful.
      </p>

      <h3>Precision: how trustworthy positive predictions are</h3>
      <p>
        Precision answers: “Of everything predicted as positive, how much was truly positive?” Its binary formula is
        <code>TP / (TP + FP)</code>. Choose precision as a primary metric when false alarms are expensive. Examples include a
        system that automatically blocks legitimate payments, removes lawful content, or sends scarce sales leads to a human
        team. High precision means the positive queue contains relatively little noise, although the model may still fail to
        find many real positives.
      </p>

      <h3>Recall: how many real positives were found</h3>
      <p>
        Recall, also called sensitivity or the true-positive rate, answers: “Of all real positives, how many did the model
        detect?” The formula is <code>TP / (TP + FN)</code>. Recall matters when missing a positive is dangerous or costly,
        such as disease screening, safety-event detection, or fraud review. Raising recall often lowers precision because a
        more permissive decision threshold captures more positives and more false positives. The correct trade-off depends on
        the action triggered by a prediction, not on a universal target.
      </p>

      <h3>Specificity: how well negatives are rejected</h3>
      <p>
        Specificity is the true-negative rate: <code>TN / (TN + FP)</code>. It complements recall by measuring performance on
        the negative class. A medical screening test, for example, can have high sensitivity but poor specificity, catching
        most cases while also producing many unnecessary follow-ups. For multiclass input this calculator treats each class
        as positive in turn, computes the one-vs-rest specificity, and reports the unweighted macro average.
      </p>

      <h3>F1 score: balancing precision and recall</h3>
      <p>
        F1 is the harmonic mean of precision and recall. Unlike an arithmetic mean, the harmonic mean stays low when either
        component is low, so a model cannot compensate for terrible recall with excellent precision or vice versa. The
        compact matrix formula is <code>2TP / (2TP + FP + FN)</code>. F1 ignores true negatives, which is useful when the
        positive class is rare but means it does not describe every classification problem. If correct rejection of negatives
        matters independently, read F1 alongside specificity or MCC.
      </p>

      <div className="tool-note">
        A metric does not select your operating threshold for you. Use domain costs to decide whether the next improvement
        should reduce false positives, reduce false negatives, or improve both classes more evenly.
      </div>

      <h2>Micro, macro, and weighted F1 for multiclass models</h2>
      <p>
        Multiclass evaluation produces one precision, recall, and F1 value per class. An average is needed when a dashboard or
        experiment table requires a single summary. Macro averaging calculates every class score independently and then gives
        each class equal weight. A class with 20 examples therefore matters as much as a class with 20,000 examples. This is a
        strong default when minority-class quality is a product requirement.
      </p>
      <p>
        Weighted F1 also calculates each class separately but weights the results by true support. It reflects the class mix
        in the evaluated dataset and can be easier to compare with overall accuracy. However, a very large majority class can
        hide weak minority performance. Micro F1 pools all per-sample decisions before calculating the metric. In ordinary
        single-label multiclass classification, every error creates one false positive and one false negative, so micro F1 is
        numerically equal to accuracy.
      </p>
      <p>
        There is no universally best average. Report macro F1 when every class deserves equal attention, weighted F1 when the
        observed class distribution is operationally meaningful, and micro F1 when each individual prediction should carry
        equal weight. The per-class bars above remain essential because two models can share the same average while failing on
        different classes.
      </p>

      <h2>Why MCC and Cohen’s kappa are included</h2>
      <p>
        Matthews correlation coefficient summarizes the entire confusion matrix and remains informative when class sizes are
        very different. It behaves like a correlation between true and predicted labels: 1 means perfect agreement, 0 means
        no better than the relevant chance structure, and -1 represents complete disagreement in the binary extreme. The
        multiclass formulation uses all rows and columns rather than averaging a collection of binary F1 scores.
      </p>
      <p>
        Cohen’s kappa measures observed agreement after subtracting the agreement expected from the label marginals. It is
        often used for annotator agreement but also helps describe classifiers when the frequency of predicted classes differs
        from the frequency of true classes. Kappa and MCC answer different statistical questions, so agreement between them is
        reassuring while a large difference is a reason to inspect the matrix and class supports more closely.
      </p>

      <h2>JavaScript implementation and reproducible Python comparison</h2>
      <p>
        Every result above is calculated locally with the JavaScript implementation in <code>@kanaries/ml</code>. The code
        panel emits the exact label arrays currently in the calculator and calls <code>confusionMatrix</code>,
        <code>accuracyScore</code>, <code>precisionScore</code>, <code>recallScore</code>, and <code>f1Score</code>; the page
        derives weighted F1, MCC, and kappa from that matrix with the corresponding sklearn formulas. The Python tab sends
        the same data to scikit-learn. This side-by-side form makes the tool useful both for a quick answer and for moving a
        verified calculation into a browser application, Node.js service, notebook, or test suite.
      </p>
      <p>
        The CSV export includes the displayed matrix and headline metrics. The PNG export captures the heatmap for reports and
        presentations. Because the calculation happens in the browser, sensitive labels are not transmitted to a calculation
        API. For a production evaluation pipeline, keep class order explicit, record the positive label and decision
        threshold, and save the dataset version alongside the exported scores so later comparisons remain reproducible.
      </p>
    </ToolPageLayout>
  );
}
