import type { Metadata } from 'next';
import Link from 'next/link';
import { GradientDescentPlayground } from '@/components/playground/GradientDescentPlayground';
import { ToolPageLayout, type ToolFaq } from '@/components/tools/ToolPageLayout';

const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.kanaries.net').replace(/\/$/, '');

export const metadata: Metadata = {
  title: { absolute: 'Gradient Descent Visualization — Compare SGD, Momentum & Adam' },
  description: 'Compare SGD, Momentum, and Adam in an interactive gradient descent visualization. Change the loss surface, learning rate, and starting point in your browser.',
  keywords: ['gradient descent visualization', 'SGD visualization', 'Adam optimizer visualization', 'Momentum gradient descent', 'machine learning optimizer'],
  alternates: { canonical: `${siteUrl}/playground/gradient-descent` },
  openGraph: { title: 'Gradient Descent Visualization: SGD vs Momentum vs Adam', description: 'Animate three optimizers from the same point across convex and non-convex loss surfaces.', url: `${siteUrl}/playground/gradient-descent` },
};

const faq: ToolFaq[] = [
  { question: 'What is gradient descent?', answer: 'Gradient descent is an iterative optimization method. At each step it computes the gradient of an objective and moves parameters in the opposite direction, which locally decreases the loss when the learning rate is appropriate.' },
  { question: 'How is Momentum different from ordinary SGD?', answer: 'Ordinary SGD follows the current gradient. Momentum also carries a velocity built from earlier gradients, which can accelerate progress along consistent directions and reduce back-and-forth oscillation across narrow valleys.' },
  { question: 'What does Adam add?', answer: 'Adam tracks exponential moving averages of gradients and squared gradients. Bias correction and per-coordinate scaling let it adapt step sizes, often making it effective when gradients differ greatly by direction or training data is noisy.' },
  { question: 'What happens when the learning rate is too high?', answer: 'Updates can overshoot a minimum, oscillate, or diverge. A very low rate is usually stable but slow. Schedules, warmup, normalization, and adaptive optimizers help, but the rate still needs validation.' },
  { question: 'Are these paths internal @kanaries/ml training logs?', answer: 'No. The playground openly implements the textbook update equations for teaching and uses @kanaries/ml numerical utilities for the objective. It does not present simulated points as hidden estimator logs.' },
];

export default function GradientDescentPage() {
  return (
    <ToolPageLayout
      name="Gradient descent visualization"
      description="Put SGD, Momentum, and Adam on the same loss surface. Drag their starting point, tune the learning rate, and animate how optimizer state changes the route to a minimum."
      pathname="/playground/gradient-descent"
      sectionName="Playground"
      sectionPath="/playground"
      eyebrow="Interactive optimization playground"
      activityLabel="Live optimizer simulation"
      tool={<GradientDescentPlayground />}
      faq={faq}
      related={[
        { href: '/tools/logistic-regression-calculator', title: 'Logistic regression calculator', description: 'See how optimization produces usable classification probabilities.' },
        { href: '/playground/pca', title: 'PCA visualization', description: 'Explore an eigenvector-based alternative to iterative loss minimization.' },
        { href: '/docs/apis/linear/logisticRegression', title: 'Logistic regression JavaScript API', description: 'Train and predict with a familiar estimator workflow.' },
      ]}
    >
      <h2>What gradient descent does</h2>
      <p>Machine-learning training often means finding parameters that minimize a loss function. Gradient descent approaches that problem through local slope information. The gradient is a vector of partial derivatives pointing toward the steepest increase in loss. Subtracting a fraction of that vector moves parameters downhill. Repeating the update turns derivatives into a sequence of candidate solutions.</p>
      <p>The idea is simple enough to write in a few lines, yet its behavior depends on the objective geometry, starting point, learning rate, gradient noise, and optimizer state. A round convex bowl is easy. A narrow curved valley can produce slow zigzags. A non-convex surface adds saddle points and local minima. This visualization makes those differences visible rather than hiding them inside a training call.</p>

      <h2>Compare SGD, Momentum, and Adam interactively</h2>
      <p>Choose a loss surface, drag the shared start marker, and press Play. The chart advances three update rules from identical initial coordinates. Plain SGD follows the current gradient. Momentum combines the gradient with a decaying velocity. Adam estimates both the first moment of the gradient and its uncentered second moment, then uses those estimates to adapt the update in each coordinate.</p>
      <p>The simulation runs entirely in JavaScript. Its objectives use <code>KMath.sum</code> from <code>@kanaries/ml</code>, while the optimizer equations are intentionally written in the component so learners can inspect them. This distinction matters: the lines are a transparent educational simulation, not undocumented internal traces extracted from a model.</p>

      <h2>Reading the optimizer trajectories</h2>
      <p>On the convex bowl, SGD usually takes a direct path but may move faster along the steep axis than the shallow one. Momentum builds speed where gradients agree and can cross the minimum before damping out. Adam normalizes updates using recent squared gradients, so its path may look more balanced across differently scaled directions.</p>
      <p>The Rosenbrock objective contains a long, curved valley. Reaching the valley is not the same as following it to the minimum, which exposes oscillation and coordinate imbalance. The rippled surface adds several local basins. No first-order optimizer can promise the global minimum on every non-convex objective; initialization changes which basin the path encounters.</p>

      <h2>Learning rate is part of the algorithm</h2>
      <p>The learning rate multiplies every update. Raise it and optimization covers more distance per iteration, but overly aggressive steps overshoot or bounce between sides of a valley. Lower it and the path becomes stable but may make negligible progress within the iteration budget. Try several rates on each surface and compare loss values rather than judging motion alone.</p>
      <p>Real training commonly changes the rate over time. Decay schedules reduce it as training approaches a solution, warmup starts cautiously, and adaptive methods scale coordinates based on gradient history. Batch size changes gradient noise as well. The clean surfaces here isolate update behavior before those production considerations are introduced.</p>

      <h2>From equations to machine-learning models</h2>
      <p>For a model with millions of parameters, the chart would be impossible to draw directly, but the update logic is the same. Automatic differentiation computes gradients, the optimizer updates tensors, and a validation metric checks whether lower training loss improves generalization. Regularization adds terms or constraints so the chosen parameters do not merely memorize training examples.</p>
      <p>Gradient clipping in this playground caps very steep gradients so all paths remain visible. Clipping is also used in neural-network training, especially for exploding gradients, but it changes the effective update and should be monitored. Likewise, finite precision, stopping criteria, and reproducible random seeds matter when an experiment becomes production software.</p>

      <h2>JavaScript and Python implementation</h2>
      <p>The code tabs implement the same convex objective and SGD loop with <code>@kanaries/ml</code> in JavaScript and NumPy in Python. Momentum would add a velocity vector; Adam would add first- and second-moment vectors plus bias correction. Keeping that state explicit is a useful way to understand what an optimizer contributes beyond the raw gradient.</p>
      <p>Continue with the <Link href="/tools/logistic-regression-calculator">logistic regression calculator</Link> to connect optimization to classification probabilities, inspect the <Link href="/docs/apis/linear/logisticRegression">JavaScript logistic regression API</Link> for a fitted estimator, or contrast iterative optimization with the projection geometry in the <Link href="/playground/pca">PCA visualization</Link>.</p>

      <h2>Diagnosing optimization in practice</h2>
      <p>Record training loss, validation loss, gradient norm, update norm, and learning rate together. A flat training loss with tiny updates can indicate a rate that is too low, saturated activations, poor feature scaling, or a coding error. Exploding loss and non-finite numbers point toward an excessive rate, unstable arithmetic, or gradients that need clipping. Falling training loss paired with worsening validation performance is an overfitting signal, not an optimizer victory.</p>
      <p>Optimizer comparisons should use the same initialization, batches, preprocessing, stopping budget, and random seeds. One run is rarely enough when minibatch order or initialization is stochastic. Repeat experiments and report the distribution of the metric that matters to the application. Checkpoints should include optimizer state as well as model parameters; restoring weights without Momentum velocity or Adam moments changes the subsequent trajectory. Finally, remember that faster reduction of the training objective does not guarantee better generalization. The optimizer is one part of a system that includes the objective, data, regularization, schedule, architecture, and evaluation protocol.</p>
    </ToolPageLayout>
  );
}
