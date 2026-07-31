import type { Metadata } from 'next';
import Link from 'next/link';
import { LogisticRegressionCalculator } from '@/components/tools/LogisticRegressionCalculator';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');
const pathname = '/tools/logistic-regression-calculator';
const title = 'Logistic Regression Calculator — Odds Ratios & Decision Boundary, Free';
const description =
  'Fit logistic regression in your browser from pasted CSV data. Get coefficients, odds ratios, probabilities, classification metrics, a sigmoid curve, and a 2D decision boundary.';

export const metadata: Metadata = {
  title: { absolute: title },
  description,
  keywords: [
    'logistic regression calculator',
    'odds ratio calculator logistic regression',
    'logistic regression decision boundary',
    'binary logistic regression online',
    'sigmoid curve calculator',
  ],
  alternates: { canonical: `${siteUrl}${pathname}` },
  openGraph: { type: 'website', title, description, url: `${siteUrl}${pathname}` },
  twitter: { card: 'summary_large_image', title, description },
};

const faq: ToolFaq[] = [
  {
    question: 'What does a logistic regression coefficient mean?',
    answer:
      'A coefficient is the change in log odds of the positive outcome for a one-unit increase in that feature, holding the other features constant. A positive coefficient raises the estimated probability; a negative coefficient lowers it. The size only has a useful interpretation together with the feature’s unit and scale.',
  },
  {
    question: 'How do I interpret an odds ratio?',
    answer:
      'The odds ratio is exp(coefficient). A value of 1 means no change in odds for a one-unit feature increase, a value above 1 means higher odds, and a value below 1 means lower odds. For example, 1.5 means the odds are multiplied by 1.5, not that probability rises by 50 percentage points.',
  },
  {
    question: 'What is the difference between logistic and linear regression?',
    answer:
      'Linear regression predicts an unbounded numeric outcome, while logistic regression models the probability of a class and keeps it between 0 and 1 with the sigmoid function. Logistic regression also uses a classification threshold, commonly 0.5, to turn probability into a label.',
  },
  {
    question: 'Can this calculator use more than one predictor?',
    answer:
      'Yes. Put any number of numeric feature columns before the final binary target column. The calculator reports a coefficient and odds ratio for every feature. With two or more features, the chart displays the decision boundary for the first two while holding the interpretation of all fitted coefficients in the table.',
  },
  {
    question: 'Why are the features standardized during fitting?',
    answer:
      'Standardization puts features on comparable numeric scales so gradient descent converges more reliably. The displayed coefficients and intercept are transformed back to the original feature units, so the reported odds ratios still describe a one-unit change in the values you pasted.',
  },
  {
    question: 'Is this a replacement for statistical inference software?',
    answer:
      'No. This tool is designed for interactive fitting, prediction, visualization, and model understanding. It does not currently report standard errors, confidence intervals, p-values, survey weights, clustered errors, or the diagnostics required for formal inferential claims.',
  },
];

export default function LogisticRegressionCalculatorPage() {
  return (
    <ToolPageLayout
      name="Logistic Regression Calculator"
      description={description}
      pathname={pathname}
      tool={<LogisticRegressionCalculator />}
      faq={faq}
      related={[
        {
          href: '/tools/confusion-matrix-calculator',
          title: 'Confusion matrix calculator',
          description: 'Inspect F1, precision, recall, MCC, kappa, and multiclass errors in detail.',
        },
        {
          href: '/docs/apis/linear/logisticRegression',
          title: 'Logistic regression in JavaScript',
          description: 'Read the browser and Node.js API guide for the estimator used on this page.',
        },
        {
          href: '/docs/apis/metrics',
          title: 'JavaScript Metrics API',
          description: 'Calculate precision, recall, F1, confusion matrices, and binary ranking curves.',
        },
      ]}
    >
      <h2>What logistic regression calculates</h2>
      <p>
        Logistic regression is a supervised classification algorithm that estimates the probability of one of two outcomes.
        A model begins with a linear score: the intercept plus every feature multiplied by its coefficient. It then passes that
        score through the sigmoid function, turning any real number into a probability between zero and one. The familiar
        S-shaped sigmoid is why a one-feature fit bends smoothly toward 0 and 1 instead of producing the unbounded line used
        by ordinary linear regression.
      </p>
      <p>
        This calculator fits the model from numeric CSV data entirely in your browser. Put predictor columns first and the
        binary target in the final column. The greater numeric target label is treated as the positive class. The result
        includes coefficients, an intercept, odds ratios, fitted probabilities, predicted classes, accuracy, F1, log loss,
        Matthews correlation coefficient, and a training confusion matrix. With one feature you see the sigmoid curve; with
        two or more features you see the decision boundary across the first two features.
      </p>
      <p>
        Logistic regression is a strong baseline when you want interpretable direction and magnitude, probability estimates,
        fast fitting, and a roughly linear decision boundary after feature engineering. Common applications include churn,
        lead qualification, credit risk, conversion, medical screening, quality control, and any yes/no outcome where a
        probability is more useful than a label alone.
      </p>

      <h2>How to use the calculator</h2>
      <ol>
        <li>Paste comma-separated numeric data or load one of the built-in examples.</li>
        <li>Keep one or more feature columns on the left and the binary target as the last column.</li>
        <li>Review the fitted metrics, chart, coefficient table, and confusion matrix.</li>
        <li>Move the new-sample sliders to see probability and predicted class update immediately.</li>
        <li>Download predictions, save the chart, or copy equivalent JavaScript and Python code.</li>
      </ol>
      <p>
        The preview table is editable, so you can correct individual observations without rebuilding the CSV. Every accepted
        edit triggers a fresh model fit. Feature columns are standardized internally because gradient descent is sensitive to
        very different scales. The tool converts fitted weights back into your original units before displaying coefficients,
        odds ratios, and the equation used for live prediction.
      </p>
      <div className="tool-note">
        The reported accuracy and confusion matrix describe the same data used for fitting. They are useful for understanding
        the model, but they are not an unbiased estimate of future performance. Use a held-out test set or cross-validation
        before selecting a model for production.
      </div>

      <h2>Reading coefficients and odds ratios</h2>
      <p>
        A logistic coefficient operates on log odds. If a coefficient is positive, increasing that feature raises the model’s
        estimated odds of the positive class while the other features stay fixed. If it is negative, increasing the feature
        lowers the odds. A coefficient of zero means the feature does not change the linear score. Comparing raw coefficient
        magnitudes across differently measured features can be misleading: one year, one dollar, and one percentage point are
        not comparable changes.
      </p>
      <p>
        Exponentiating a coefficient produces an odds ratio, which is usually easier to explain. An odds ratio of 1.30 means a
        one-unit increase multiplies the odds by 1.30, holding other predictors constant. An odds ratio of 0.70 multiplies the
        odds by 0.70, a 30% reduction in odds. Odds are not probability. Moving from odds of 1:4 to 1.3:4 changes probability
        differently than applying the same multiplier when the starting odds are 4:1. Use the new-sample controls to see that
        nonlinear probability effect at realistic feature values.
      </p>
      <p>
        The intercept is the log odds when every feature equals zero in its original unit. It can be meaningful when zero is a
        plausible baseline, but it may simply anchor the fitted equation when zero falls outside the data. Centering features
        before analysis can make the intercept easier to interpret. This calculator standardizes for optimization and then
        transforms the parameters back, so the displayed intercept refers to the unstandardized values shown in the input.
      </p>

      <h2>Understanding the sigmoid curve and decision boundary</h2>
      <p>
        In one dimension, the chart plots the positive-class probability against the feature. The midpoint of the curve is the
        location where probability equals 0.5 and the model changes its predicted class. A large coefficient creates a steep
        transition; a small coefficient produces a gradual transition. Points at the top and bottom show observed positive and
        negative labels, while the highlighted point follows the live prediction slider.
      </p>
      <p>
        With two predictors, all combinations assigned probability 0.5 form a straight decision boundary. One side is
        classified as the lower label and the other as the higher label. The colored background indicates the predicted side
        and grows slightly stronger as probability moves away from 0.5. If your classes curve around each other or form
        disconnected islands, a straight boundary will underfit unless you add nonlinear transformations such as interactions
        or polynomial features.
      </p>
      <p>
        When more than two features are present, the fitted model still uses every column and the coefficient table reports all
        of them. A flat screen cannot directly display a high-dimensional hyperplane, so the plot shows the first two feature
        axes. Treat it as a projection rather than a complete picture. Use the live controls, metrics, and exported probability
        table to examine the influence of the remaining features.
      </p>

      <h2>Logistic regression versus linear regression</h2>
      <p>
        Linear regression minimizes numeric prediction error and can return values below zero or above one. Applying it to a
        binary target creates invalid probabilities and assumes the wrong error distribution. Logistic regression instead
        models log odds and uses a classification loss, so its output remains within the probability range. Both models are
        linear in their parameters, but they answer different questions: linear regression predicts how much; logistic
        regression predicts the probability of which class.
      </p>
      <p>
        A probability does not become a decision until you choose a threshold. This page uses 0.5 for clarity, but production
        systems often move the threshold. Lowering it generally increases recall and false positives; raising it generally
        increases precision and false negatives. After exporting probabilities, evaluate candidate thresholds with the
        <Link href="/tools/confusion-matrix-calculator">confusion matrix and F1 calculator</Link>, or use ROC and
        precision-recall curves from the <Link href="/docs/apis/metrics">Metrics API</Link>.
      </p>

      <h2>When the model is a good fit—and when it is not</h2>
      <p>
        Logistic regression is attractive when interpretability, speed, and calibrated ranking are important. It works well
        with numeric or encoded categorical features, benefits from sensible scaling, and can remain competitive on many
        tabular datasets. Because every coefficient has a direction, product and domain teams can inspect whether the learned
        relationship is plausible. It is also easy to reproduce in JavaScript, Python, SQL-like scoring systems, and edge or
        browser environments.
      </p>
      <p>
        It is less suitable when the true boundary is strongly nonlinear and feature engineering cannot represent it, when
        observations are not independent, or when perfect separation drives coefficients toward extreme values. Highly
        correlated predictors can also make individual coefficient interpretations unstable even when predictions remain
        useful. Sparse categories, missing values, outliers, and data leakage should be handled before fitting. For causal or
        inferential work, you additionally need standard errors, confidence intervals, design assumptions, and diagnostics not
        provided by this interactive calculator.
      </p>

      <h2>JavaScript implementation in the browser</h2>
      <p>
        The estimator on this page is <code>Linear.LogisticRegression</code> from <code>@kanaries/ml</code>. It follows the
        familiar fit-and-predict workflow used by scikit-learn while running in JavaScript or TypeScript. All parsing, scaling,
        gradient-descent fitting, metrics, visualization, and exports happen on the client. That makes the calculator a useful
        reference for browser-native analysis, privacy-sensitive prototypes, frontend teaching tools, and Node.js workflows
        that should not depend on a Python service.
      </p>
      <p>
        The code tabs preserve the current data and show equivalent starting points for <code>@kanaries/ml</code> and
        scikit-learn. Their predictions should be compared on a held-out dataset rather than assuming coefficients will be
        numerically identical: optimizers, regularization defaults, stopping rules, and scaling choices can differ across
        implementations. For a deeper API explanation and a minimal runnable example, continue to the
        <Link href="/docs/apis/linear/logisticRegression"> Logistic Regression in JavaScript guide</Link>.
      </p>
    </ToolPageLayout>
  );
}
