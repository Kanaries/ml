import type { Metadata } from 'next';
import Link from 'next/link';
import { LinearRegressionCalculator } from '@/components/tools/LinearRegressionCalculator';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';
import { SITE_URL } from '@/lib/seo';

const siteUrl = SITE_URL;
const pathname = '/tools/linear-regression-calculator';

const title = 'Linear & Ridge Regression Calculator — CSV, R² & Residuals';
const description =
  'Fit OLS linear or L2 Ridge regression on pasted CSV data in your browser. Inspect R², RMSE, coefficients, residual plots, and export JS or Python code.';

export const metadata: Metadata = {
  title: { absolute: title },
  description,
  keywords: [
    'linear regression calculator csv',
    'multiple linear regression calculator',
    'ridge regression calculator online',
    'residual plot calculator',
    'r squared and rmse calculator',
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
    question: 'Does this linear regression calculator support multiple features (multivariate CSV)?',
    answer:
      'Yes. Paste any numeric CSV or TSV table with two or more columns. Every column before the last is treated as a predictor (X) and the final column is treated as the continuous target (y). For single-feature data, the chart draws the fitted line; for multi-feature data, it switches to an Actual vs Predicted scatter plot alongside residuals.',
  },
  {
    question: 'When should I switch from OLS Linear Regression to Ridge Regression?',
    answer:
      'Switch to Ridge Regression (L2 penalty α > 0) when predictors are strongly correlated (multicollinearity), when sample size is close to or smaller than feature count, or when OLS coefficients swing dramatically after adding or removing a few rows. Try the Collinear (Ridge Demo) preset to see L2 shrinkage stabilize coefficients.',
  },
  {
    question: 'Why should predictors be standardized before applying Ridge regularization?',
    answer:
      'Ridge penalizes the sum of squared coefficients (α Σ βⱼ²). If one feature is measured in thousands (such as square feet) and another in single digits (bedrooms), an unstandardized penalty shrinks the small-scale feature far more aggressively. Standardizing puts every predictor on equal footing, and this tool automatically unscales the fitted weights back into original units for prediction.',
  },
  {
    question: 'What is the difference between R² and Adjusted R²?',
    answer:
      'R² measures the share of target variance explained by the model, but adding any random feature never lowers training R². Adjusted R² penalizes extra predictors relative to sample size (n − p − 1), so adding an uninformative column causes Adjusted R² to drop—making it a safer summary for multiple regression.',
  },
  {
    question: 'How do I read the residual plot?',
    answer:
      'A healthy linear model produces residuals scattered randomly around zero with roughly constant vertical spread across predicted values. A curved U-shape signals nonlinear relationships (try polynomial regression), a funnel shape signals heteroscedasticity, and isolated spikes flag potential outliers.',
  },
  {
    question: 'Is my CSV dataset sent to an external server?',
    answer:
      'No. Parsing, QR least-squares solving, L2 Ridge fitting, residual diagnostics, and PNG/CSV exports execute entirely inside your browser using @kanaries/ml.',
  },
];

export default function LinearRegressionCalculatorPage() {
  return (
    <ToolPageLayout
      name="Linear & Ridge Regression Calculator"
      description={description}
      pathname={pathname}
      tool={<LinearRegressionCalculator />}
      faq={faq}
      related={[
        {
          href: '/tools/polynomial-regression-calculator',
          title: 'Polynomial Regression Calculator',
          description: 'Fit curved trends when residual plots show systematic nonlinearity.',
        },
        {
          href: '/tools/outlier-calculator',
          title: 'Outlier Calculator',
          description: 'Flag extreme CSV values before they distort ordinary least squares slopes.',
        },
        {
          href: '/docs/apis/linear/linearRegression',
          title: 'LinearRegression API in JS',
          description: 'Use QR least-squares linear regression programmatically in browser or Node.js.',
        },
        {
          href: '/docs/apis/linear/ridgeRegression',
          title: 'RidgeRegression API in JS',
          description: 'Apply L2 regularization inside JavaScript Pipelines and GridSearchCV.',
        },
      ]}
    >
      <h2>How OLS Linear Regression and Ridge Regression work</h2>
      <p>
        <strong>Ordinary Least Squares (OLS) linear regression</strong> models a continuous target <code>y</code> as a
        weighted linear combination of numeric features <code>x₁, x₂, …, xₚ</code> plus an intercept <code>β₀</code>,
        choosing weights that minimize the sum of squared residuals <code>Σ (yᵢ − ŷᵢ)²</code>. Because OLS has a
        closed-form solution via QR factorization, it fits deterministically in milliseconds and produces directly
        interpretable coefficients: holding other features constant, a one-unit increase in <code>xⱼ</code> shifts the
        expected prediction by <code>βⱼ</code>.
      </p>
      <p>
        In real-world tables, however, predictors are often correlated—such as website sessions and pageviews, or home
        square footage and room count. When columns move together (multicollinearity), ordinary least squares can assign
        wildly opposing positive and negative weights that cancel on training rows yet explode on slightly perturbed
        inputs. <strong>Ridge Regression</strong> resolves this instability by adding an <strong>L2 penalty</strong>{' '}
        <code>α Σ βⱼ²</code> to the squared-error objective (leaving the intercept unpenalized). Raising{' '}
        <code>α</code> smoothly shrinks large coefficients toward zero, trading a small amount of training bias for
        lower variance and far more reliable out-of-sample predictions.
      </p>

      <h2>Interpreting raw coefficients vs standardized β weights</h2>
      <p>
        A common mistake in multiple linear regression is ranking feature importance by raw coefficient magnitude. In
        the <strong>Multivariate Housing</strong> preset above, <code>bedrooms</code> carries a much larger raw
        coefficient than <code>sqft</code> simply because one extra bedroom is a large jump whereas one extra square
        foot is tiny.
      </p>
      <p>
        To compare predictors fairly on a common scale, inspect the <strong>Standardized β</strong> column and relative
        magnitude bars in the table above. Standardized β equals <code>βⱼ × (σₓⱼ / σᵧ)</code>: the predicted shift in
        standard deviations of <code>y</code> for a one-standard-deviation increase in predictor <code>xⱼ</code>. Use
        raw coefficients for algebraic prediction equations and business unit explanations, and use standardized β to
        spot which columns drive the majority of target variation.
      </p>

      <div className="tool-note">
        Always leave <strong>Standardize predictors before L2 penalty</strong> enabled when fitting Ridge Regression on
        mixed-unit CSV columns. Otherwise, columns with small numeric ranges get over-penalized while large-scale
        columns escape shrinkage.
      </div>

      <h2>Reading R², Adjusted R², RMSE, and MAE together</h2>
      <p>
        No single regression metric tells the entire story. Combine scale-free goodness-of-fit summaries with error
        metrics measured in your target’s original units:
      </p>
      <ul>
        <li>
          <strong>R² (Coefficient of Determination):</strong> Fraction of variance in <code>y</code> explained by the
          model relative to predicting the constant mean. Useful for high-level communication, but remember that adding
          noise columns never decreases training R².
        </li>
        <li>
          <strong>Adjusted R²:</strong> Corrects R² for degrees of freedom <code>(n − p − 1)</code>. If you paste an
          irrelevant column into a small CSV, Adjusted R² drops—warning you against overfitting.
        </li>
        <li>
          <strong>RMSE (Root Mean Squared Error):</strong> Square root of average squared error, expressed in the exact
          units of <code>y</code> (for example thousands of dollars). Because errors are squared before averaging, RMSE
          penalizes occasional large misses heavily.
        </li>
        <li>
          <strong>MAE (Mean Absolute Error):</strong> Average absolute prediction error. Comparing RMSE against MAE
          immediately reveals tail behavior: when RMSE is much larger than MAE, a handful of high-leverage outliers are
          dominating squared loss.
        </li>
      </ul>

      <h2>Diagnosing model assumptions with residual plots</h2>
      <p>
        A high R² does not guarantee that a linear model is appropriate. Inspect the{' '}
        <strong>Residuals (y − ŷ) vs Predicted</strong> chart every time you fit a CSV:
      </p>
      <h3>1. Curved or U-shaped residuals</h3>
      <p>
        Systematic curvature above and below zero means a straight hyperplane misses nonlinear structure. Try the{' '}
        <Link href="/tools/polynomial-regression-calculator">Polynomial Regression Calculator</Link> or add spline basis
        features with <Link href="/docs/apis/utils/advancedPreprocessing">SplineTransformer</Link>.
      </p>
      <h3>2. Funnel-shaped spread (heteroscedasticity)</h3>
      <p>
        When residual variance expands as predicted values grow—common in prices, revenue, and latency—consider modeling
        a log-transformed target with{' '}
        <Link href="/docs/apis/compose">Compose.TransformedTargetRegressor</Link> or fitting a{' '}
        <Link href="/docs/apis/linear/generalizedLinearModels">Gamma / Tweedie GLM</Link>.
      </p>
      <h3>3. Isolated extreme residuals</h3>
      <p>
        Because OLS squares residuals, one corrupted row can tilt the entire hyperplane. Flag bad rows first with the{' '}
        <Link href="/tools/outlier-calculator">Outlier Calculator</Link>, or switch to Huber, RANSAC, or Theil–Sen
        estimators documented in <Link href="/docs/apis/linear/robustRegressors">Robust Regression in JS</Link>.
      </p>

      <h2>Tuning Ridge α and choosing between OLS, Ridge, Lasso, and ElasticNet</h2>
      <p>
        Choosing the right regularization strength <code>α</code> is a classic bias–variance tradeoff. At{' '}
        <code>α = 0</code>, Ridge Regression reduces mathematically to Ordinary Least Squares: zero shrinkage bias,
        lowest possible training MSE, but maximum sensitivity to collinearity and small-sample noise. As you sweep the{' '}
        <code>α</code> slider upward from <code>0.01</code> toward <code>50</code>, correlated weights share credit
        smoothly rather than canceling each other with giant opposite signs. Pushing <code>α</code> too far eventually
        underfits, pulling every slope toward zero until predictions flatten near the sample mean of <code>y</code>.
      </p>
      <p>
        In production code, never pick <code>α</code> purely from training R²—because training R² always peaks at{' '}
        <code>α = 0</code>. Instead, wrap <code>StandardScaler</code> and <code>Linear.RidgeRegression</code> inside a{' '}
        <Link href="/docs/apis/pipeline">Pipeline</Link> and evaluate a logarithmic grid (for example{' '}
        <code>[0.01, 0.1, 1, 10, 100]</code>) via <code>GridSearchCV</code> in{' '}
        <Link href="/docs/apis/utils/modelSelection">utils.ModelSelection</Link>. That way scaling statistics fit
        strictly inside each training fold without leaking validation information.
      </p>
      <ul>
        <li>
          <strong>Use OLS Linear Regression</strong> when sample count comfortably exceeds feature count, predictors
          are low-correlation, and you want unbiased classical parameter estimates.
        </li>
        <li>
          <strong>Use Ridge Regression (L2)</strong> when predictors are correlated or noisy and you expect most
          features to carry small real contributions.
        </li>
        <li>
          <strong>Use <Link href="/docs/apis/linear/lassoRegression">Lasso Regression (L1)</Link></strong> when you
          suspect most columns are irrelevant and want automatic sparse feature selection that drives weak weights to
          exact zero.
        </li>
        <li>
          <strong>Use <Link href="/docs/apis/linear/elasticNet">ElasticNet (L1 + L2)</Link></strong> when your dataset
          combines both challenges—correlated predictor groups plus irrelevant noise columns.
        </li>
      </ul>

      <h2>Moving from browser exploration to JavaScript or Python code</h2>
      <p>
        Every fit above runs locally in pure TypeScript using <code>Linear.LinearRegression</code> (QR least squares)
        and <code>Linear.RidgeRegression</code> from <code>@kanaries/ml</code>, with numerical results verified against
        scikit-learn. Use the <strong>JavaScript</strong> and <strong>Python</strong> tabs to copy a runnable pipeline
        with your exact CSV values, export predictions and residuals for spreadsheets, or share your configured model
        via <strong>Share link</strong> or <strong>Embed</strong> in the header bar.
      </p>
      <p>
        Once your model passes validation on held-out data, serialize the entire preprocessing and regression{' '}
        <Link href="/docs/apis/pipeline">Pipeline</Link> with <code>JSON.stringify(model)</code> and revive it inside a
        browser tab, Web Worker, Cloudflare Worker, or Node.js API using{' '}
        <Link href="/docs/apis/serialization">loadModel()</Link>. Because linear models store only one weight per
        feature plus an intercept, serialized artifacts stay under a few kilobytes and score thousands of rows in
        sub-millisecond latency without maintaining a Python backend.
      </p>
    </ToolPageLayout>
  );
}
