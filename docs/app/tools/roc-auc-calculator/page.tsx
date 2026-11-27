import type { Metadata } from 'next';
import Link from 'next/link';
import { RocAucCalculator } from '@/components/tools/RocAucCalculator';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';
import { SITE_URL } from '@/lib/seo';

const siteUrl = SITE_URL;
const pathname = '/tools/roc-auc-calculator';

const title = 'ROC-AUC & PR Curve Calculator — Threshold, Youden’s J & AP, Free';
const description =
  'Plot ROC and Precision-Recall curves from pasted y_true and y_score CSV data in your browser. Tune thresholds, find Youden’s J & best F1, and export code.';

export const metadata: Metadata = {
  title: { absolute: title },
  description,
  keywords: [
    'roc auc calculator',
    'roc curve generator online',
    'precision recall curve calculator',
    'average precision calculator',
    'youden j threshold calculator',
  ],
  alternates: { canonical: `${siteUrl}${pathname}` },
  openGraph: {
    type: 'website',
    siteName: '@kanaries/ml',
    title,
    description,
    url: `${siteUrl}${pathname}`,
    images: [{ url: `${siteUrl}${pathname}/opengraph-image`, width: 1200, height: 630, alt: title }],
  },
  twitter: {
    card: 'summary_large_image',
    title,
    description,
    images: [`${siteUrl}${pathname}/opengraph-image`],
  },
};

const faq: ToolFaq[] = [
  {
    question: 'What is ROC-AUC and how should I interpret it?',
    answer:
      'ROC-AUC (Area Under the Receiver Operating Characteristic curve) measures how well a classifier ranks positive instances above negative instances across every possible decision threshold. A score of 0.5 equals random guessing, 1.0 indicates perfect separation, and an AUC of 0.85 means a randomly chosen positive receives a higher score than a randomly chosen negative 85% of the time.',
  },
  {
    question: 'When should I use a Precision-Recall (PR) curve instead of ROC?',
    answer:
      'Use a Precision-Recall curve and Average Precision (AP) whenever the positive class is rare—such as fraud detection, churn alerts, or rare medical screening. ROC curves can look optimistically high on imbalanced datasets because a large true-negative count keeps the false-positive rate small even when false alarms overwhelm precision.',
  },
  {
    question: 'How is Average Precision (AP) calculated here?',
    answer:
      'This calculator computes exact scikit-learn Average Precision: the weighted sum of precisions at each threshold, with the increase in recall from the previous threshold used as the weight. Unlike trapezoidal interpolation on PR curves, step-wise Average Precision avoids optimistic linear interpolation between sparse recall points.',
  },
  {
    question: 'What is Youden’s J statistic for choosing an optimal threshold?',
    answer:
      'Youden’s J equals Sensitivity + Specificity − 1, or equivalently TPR − FPR. Geometrically, it maximizes the vertical distance between the ROC curve and the random diagonal line. It is a clean default when false positives and false negatives carry comparable weight; for asymmetric business costs, compare it against the Best F1 threshold or set a custom cost cutoff.',
  },
  {
    question: 'Does this calculator accept raw decision scores or only probabilities?',
    answer:
      'Both work for ROC-AUC, Precision-Recall curves, Youden’s J, and F1 because ranking metrics only depend on score order. When all pasted scores fall in [0, 1], the tool also reports the Brier score to evaluate probability calibration quality.',
  },
  {
    question: 'Is my pasted CSV data uploaded to a server?',
    answer:
      'No. All curve computation, threshold evaluation, PNG rendering, and CSV exports run locally inside your browser using @kanaries/ml. You can also copy a shareable link or embeddable iframe without sending data to a backend.',
  },
];

export default function RocAucCalculatorPage() {
  return (
    <ToolPageLayout
      name="ROC-AUC & PR Curve Calculator"
      description={description}
      pathname={pathname}
      tool={<RocAucCalculator />}
      faq={faq}
      related={[
        {
          href: '/tools/confusion-matrix-calculator',
          title: 'Confusion Matrix & F1 Calculator',
          description: 'Evaluate multiclass or 2×2 counts with accuracy, precision, recall, F1, MCC, and kappa.',
        },
        {
          href: '/tools/logistic-regression-calculator',
          title: 'Logistic Regression Calculator',
          description: 'Fit a binary probabilistic classifier directly on CSV data and inspect odds ratios.',
        },
        {
          href: '/docs/apis/calibration',
          title: 'Probability Calibration in JS',
          description: 'Calibrate raw classifier scores with Platt scaling or isotonic regression.',
        },
        {
          href: '/docs/apis/metrics',
          title: 'JavaScript Metrics API',
          description: 'Run rocCurve, rocAucScore, and precisionRecallCurve programmatically in browser or Node.js.',
        },
      ]}
    >
      <h2>What is an ROC curve and why does ROC-AUC matter?</h2>
      <p>
        Binary classifiers rarely output hard 0/1 decisions natively. Instead, models such as{' '}
        <Link href="/docs/apis/linear/logisticRegression">logistic regression</Link>,{' '}
        <Link href="/docs/apis/ensemble/randomForestClassifier">random forests</Link>, gradient boosting, and neural
        networks output continuous probabilities or decision scores. Turning those scores into operational actions
        requires choosing a decision threshold—for example flagging every transaction above <code>0.50</code> as fraud.
      </p>
      <p>
        Evaluating a model at a single threshold hides how well the underlying ranking behaves across stricter or more
        permissive cutoffs. The <strong>Receiver Operating Characteristic (ROC) curve</strong> sweeps the threshold
        across every distinct score in your dataset and plots the <strong>True Positive Rate (Recall / Sensitivity)</strong>{' '}
        on the vertical axis against the <strong>False Positive Rate (1 − Specificity)</strong> on the horizontal axis.
        The area underneath that curve—<strong>ROC-AUC</strong>—compresses ranking quality into a single threshold-free
        number between <code>0</code> and <code>1</code>.
      </p>
      <p>
        ROC-AUC has a direct probabilistic interpretation: if you pick one true positive observation and one true
        negative observation at random, ROC-AUC is the probability that your model assigned a higher score to the
        positive case. A random model traces the diagonal baseline and scores <code>0.50</code>; a perfect separator
        hugs the top-left corner and scores <code>1.00</code>.
      </p>

      <h2>ROC curve vs Precision–Recall (PR) curve on imbalanced datasets</h2>
      <p>
        While ROC-AUC is scale-invariant and unaffected by class prevalence, that same property can create a blind spot
        when positives are rare. Suppose only 2% of events are fraudulent. Raising false positives from 20 to 200 barely
        nudges the False Positive Rate because the denominator contains tens of thousands of true negatives—so the ROC
        curve still looks near-perfect. Meanwhile, an investigation team reviewing those alerts sees precision plummet
        from 50% to under 10%.
      </p>
      <p>
        The <strong>Precision–Recall (PR) curve</strong> eliminates true negatives from both axes, plotting{' '}
        <strong>Precision</strong> (positive predictive value) against <strong>Recall</strong> (true positive rate).
        Its baseline is not a fixed diagonal at <code>0.50</code>, but a horizontal line equal to the dataset’s positive
        class prevalence (shown as the dashed line in the PR chart above). Comparing the <strong>Imbalanced Fraud</strong>{' '}
        preset against the <strong>Well-Separated</strong> preset demonstrates this difference immediately: even when
        ROC-AUC stays high, Average Precision reacts sharply to false positives near the top of the score ranking.
      </p>

      <div className="tool-note">
        Rule of thumb: report <strong>ROC-AUC</strong> when both classes matter equally or you want a prevalence-invariant
        ranking metric; report <strong>Average Precision (PR-AUC)</strong> alongside operating-point precision and recall
        whenever positive events are rare and false alarms carry direct review cost.
      </div>

      <h2>Choosing the right decision threshold: Youden’s J vs Best F1</h2>
      <p>
        Ranking metrics tell you which model orders examples better, but shipping a classifier still requires selecting
        a concrete cutoff. Dragging the threshold slider in the tool above moves the red operating marker simultaneously
        along both the ROC and PR curves and updates the exact TP, FP, FN, and TN counts in real time.
      </p>
      <h3>Youden’s J statistic (TPR − FPR)</h3>
      <p>
        <strong>Youden’s J</strong> equals <code>Sensitivity + Specificity − 1</code> and identifies the point on the
        ROC curve farthest above the random diagonal. It treats a 1% gain in true-positive rate as equally valuable to a
        1% reduction in false-positive rate—making it a principled default for balanced diagnostic tests and balanced
        binary classification problems.
      </p>
      <h3>Best F1 threshold</h3>
      <p>
        When class imbalance is severe or true negatives are abundant, maximizing Youden’s J can still produce too many
        false alarms. Jumping to the <strong>Best F1</strong> threshold maximizes the harmonic mean of precision and
        recall instead, focusing strictly on positive-class detection quality. For asymmetric business problems (such as
        automated payment blocking vs manual review queues), start from Best F1 or Youden’s J and fine-tune until
        precision or recall meets your operational SLA.
      </p>

      <h2>Checking probability calibration with the Brier score</h2>
      <p>
        Notice an important subtlety: ROC-AUC and PR curves are purely rank-based. Multiplying every predicted
        probability by <code>0.1</code> leaves ROC-AUC and Average Precision completely unchanged—even though the raw
        numbers are now wildly miscalibrated. Whenever your pasted scores lie in <code>[0, 1]</code>, this calculator
        also computes the <strong>Brier score</strong> (mean squared error between predicted probabilities and binary
        outcomes, where lower is better). If your model achieves strong ROC-AUC yet poor Brier score, apply Platt
        scaling or isotonic calibration using{' '}
        <Link href="/docs/apis/calibration">Calibration.CalibratedClassifierCV</Link>.
      </p>

      <h2>Common ROC-AUC &amp; Precision–Recall pitfalls in practice</h2>
      <p>
        Even experienced machine learning teams misread ranking curves when moving from a notebook to production.
        Watch out for four frequent failure modes whenever you interpret the plots above:
      </p>
      <ul>
        <li>
          <strong>Tuning thresholds on the test set:</strong> Selecting the Youden’s J or Best F1 threshold on your
          final evaluation split leaks information and inflates reported F1. Always choose the operating threshold on a
          dedicated validation split or out-of-fold cross-validation predictions, then freeze that cutoff before scoring
          the held-out test set.
        </li>
        <li>
          <strong>Tied scores and coarse model outputs:</strong> Shallow decision trees, k-NN with small <code>k</code>,
          and rule-based scorecards often assign identical scores to many rows. On the ROC curve, tied scores create
          diagonal line segments rather than pure horizontal/vertical steps, because all tied rows cross the decision
          boundary together. Use continuous probabilistic estimators or ensemble averaging if you need fine-grained
          threshold control.
        </li>
        <li>
          <strong>Comparing PR curves across different datasets:</strong> Because the Precision–Recall baseline equals
          positive prevalence, an Average Precision of <code>0.45</code> on a 1% fraud sample represents a massive
          45× lift over chance, whereas <code>0.45</code> on a 50/50 balanced dataset is worse than random guessing.
          Never compare AP across datasets with different positive rates without reporting prevalence alongside it.
        </li>
        <li>
          <strong>Ignoring high-precision or low-FPR operating regions:</strong> Two models can share an identical
          overall ROC-AUC of <code>0.88</code> while behaving very differently in the far-left region (FPR &lt; 2%)
          where automated fraud blocks or email spam filters actually operate. Always inspect the left tail of the ROC
          curve and the right tail of the PR curve rather than relying solely on the aggregate area.
        </li>
      </ul>

      <h2>JavaScript &amp; Python parity with @kanaries/ml</h2>
      <p>
        Every ROC point, AUC area, and Precision–Recall threshold rendered in this calculator runs directly in your
        browser via <code>Metrics.rocCurve</code>, <code>Metrics.rocAucScore</code>, and{' '}
        <code>Metrics.precisionRecallCurve</code> from <code>@kanaries/ml</code>, with Average Precision verified
        against <code>sklearn.metrics.average_precision_score</code>. Copy the generated JavaScript or Python tabs to
        reproduce the exact curves in unit tests, Node.js evaluation scripts, or Jupyter notebooks, or click{' '}
        <strong>Share link</strong> / <strong>Embed</strong> in the header bar to share this threshold experiment with
        your team.
      </p>
    </ToolPageLayout>
  );
}
